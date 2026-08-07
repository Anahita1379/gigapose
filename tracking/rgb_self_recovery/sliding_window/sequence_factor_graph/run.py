"""Optimize full RGB/CAD candidate sequences with a sparse SE(3) factor graph."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np

from tracking.geometry import pose_from_rt, rotation_error_deg, translation_error_m
from tracking.io import pose_to_csv_row, write_tracking_csv
from tracking.rgb_self_recovery.sliding_window.gru_selector.dataset import CandidateBundle

from .optimizer import (
    FactorGraphConfig,
    normalized_visual_cost,
    optimize_sequence,
    visual_initialization,
)


FORMAT = "rgb_self_recovery_sequence_factor_graph_v1"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--data", type=Path, required=True, help="RGB/CAD candidate bundle")
    value.add_argument(
        "--mode", choices=("candidates", "prior", "causal"), required=True,
        help=(
            "candidates: graph only; prior: generic external trajectory plus candidates; "
            "causal: backward-compatible alias for prior"
        ),
    )
    value.add_argument(
        "--prior-predictions", type=Path,
        help="Existing cascade or transformer tracking CSV for --mode prior.",
    )
    value.add_argument("--prior-translation-unit", choices=("mm", "m"), default="mm")
    value.add_argument(
        "--causal-predictions", type=Path,
        help="Deprecated alias for --prior-predictions.",
    )
    value.add_argument("--causal-translation-unit", choices=("mm", "m"), default="mm")
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--measurement-translation-sigma-m", type=float, default=2.0)
    value.add_argument("--measurement-rotation-sigma-deg", type=float, default=20.0)
    value.add_argument("--prior-translation-sigma-m", type=float, default=5.0)
    value.add_argument("--prior-rotation-sigma-deg", type=float, default=45.0)
    value.add_argument("--acceleration-sigma-mps2", type=float, default=20.0)
    value.add_argument("--angular-acceleration-sigma-radps2", type=float, default=3.0)
    value.add_argument("--jerk-sigma-mps3", type=float, default=100.0)
    value.add_argument("--visual-assignment-weight", type=float, default=0.5)
    value.add_argument("--assignment-current-weight", type=float, default=0.25)
    value.add_argument("--assignment-motion-weight", type=float, default=1.5)
    value.add_argument("--assignment-translation-scale-m", type=float, default=3.0)
    value.add_argument("--assignment-rotation-scale-deg", type=float, default=30.0)
    value.add_argument("--outer-iterations", type=int, default=3)
    value.add_argument("--maximum-nfev", type=int, default=100)
    value.add_argument("--maximum-translation-step-m", type=float, default=20.0)
    value.add_argument("--maximum-rotation-step-deg", type=float, default=90.0)
    value.add_argument(
        "--robust-loss", choices=("linear", "soft_l1", "huber", "cauchy", "arctan"),
        default="huber",
    )
    value.add_argument("--robust-scale", type=float, default=1.0)
    value.add_argument("--solver-tolerance", type=float, default=1e-4)
    value.add_argument("--finite-difference-step", type=float, default=1e-4)
    value.add_argument("--max-segments", type=int)
    value.add_argument("--overwrite", action="store_true")
    return value


def _parse_values(text: str, count: int) -> np.ndarray:
    values = np.fromstring(str(text), sep=" ", dtype=float)
    if values.size != count:
        raise ValueError(f"Expected {count} pose values, found {values.size}")
    return values


def _load_prior(path: Path, unit: str) -> dict[tuple[int, int], np.ndarray]:
    scale = 0.001 if unit == "mm" else 1.0
    output = {}
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"scene_id", "im_id", "R", "t"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Causal CSV is missing columns: {sorted(missing)}")
        for row in reader:
            key = (int(row["scene_id"]), int(row["im_id"]))
            if key in output:
                raise ValueError(f"Prior CSV has multiple poses for frame {key}")
            output[key] = pose_from_rt(
                _parse_values(row["R"], 9).reshape(3, 3),
                _parse_values(row["t"], 3) * scale,
            )
    return output


def _prior_array(bundle: CandidateBundle, path: Path, unit: str) -> np.ndarray:
    lookup = _load_prior(path, unit)
    arrays = bundle.arrays
    values, missing = [], []
    for scene_id, im_id in zip(arrays["scene_id"], arrays["im_id"]):
        key = (int(scene_id), int(im_id))
        if key not in lookup:
            missing.append(key)
        else:
            values.append(lookup[key])
    if missing:
        raise ValueError(
            f"Prior CSV is missing {len(missing)} candidate-bundle frames; first={missing[:5]}"
        )
    return np.asarray(values, dtype=float)


def _segment_time(arrays: dict, indices: np.ndarray, frame_feature_names: list[str]) -> np.ndarray:
    raw = np.asarray(arrays["time_s"][indices], dtype=float)
    output = np.zeros(len(indices), dtype=float)
    delta_index = (
        frame_feature_names.index("delta_time_s")
        if "delta_time_s" in frame_feature_names else None
    )
    for local in range(1, len(indices)):
        raw_dt = raw[local] - raw[local - 1]
        if np.isfinite(raw_dt) and raw_dt > 0:
            dt = raw_dt
        elif delta_index is not None:
            dt = float(arrays["frame_features"][indices[local], delta_index])
        else:
            dt = 0.1
        output[local] = output[local - 1] + float(np.clip(dt, 0.02, 1.0))
    return output


def _stats(values) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "count": int(values.size),
        "mean": float(values.mean()) if values.size else None,
        "rmse": float(np.sqrt(np.mean(values**2))) if values.size else None,
        "median": float(np.median(values)) if values.size else None,
        "p90": float(np.percentile(values, 90)) if values.size else None,
    }


def _pose_metrics(poses: np.ndarray, ground_truth: np.ndarray) -> dict:
    translation = [translation_error_m(a, b) for a, b in zip(poses, ground_truth)]
    rotation = [rotation_error_deg(a, b) for a, b in zip(poses, ground_truth)]
    return {
        "translation_error_m": _stats(translation),
        "rotation_error_deg": _stats(rotation),
    }


def _config(args) -> FactorGraphConfig:
    return FactorGraphConfig(
        measurement_translation_sigma_m=args.measurement_translation_sigma_m,
        measurement_rotation_sigma_deg=args.measurement_rotation_sigma_deg,
        prior_translation_sigma_m=args.prior_translation_sigma_m,
        prior_rotation_sigma_deg=args.prior_rotation_sigma_deg,
        acceleration_sigma_mps2=args.acceleration_sigma_mps2,
        angular_acceleration_sigma_radps2=args.angular_acceleration_sigma_radps2,
        jerk_sigma_mps3=args.jerk_sigma_mps3,
        visual_assignment_weight=args.visual_assignment_weight,
        assignment_current_weight=args.assignment_current_weight,
        assignment_motion_weight=args.assignment_motion_weight,
        assignment_translation_scale_m=args.assignment_translation_scale_m,
        assignment_rotation_scale_deg=args.assignment_rotation_scale_deg,
        outer_iterations=args.outer_iterations,
        maximum_nfev=args.maximum_nfev,
        maximum_translation_step_m=args.maximum_translation_step_m,
        maximum_rotation_step_deg=args.maximum_rotation_step_deg,
        robust_loss=args.robust_loss,
        robust_scale=args.robust_scale,
        solver_tolerance=args.solver_tolerance,
        finite_difference_step=args.finite_difference_step,
    )


def main() -> None:
    args = parser().parse_args()
    prior_path = args.prior_predictions or args.causal_predictions
    if args.prior_predictions is not None and args.causal_predictions is not None:
        raise ValueError("Pass only one of --prior-predictions and --causal-predictions")
    uses_prior = args.mode in {"prior", "causal"}
    if uses_prior and prior_path is None:
        raise ValueError("--prior-predictions is required for prior-assisted mode")
    if not uses_prior and prior_path is not None:
        raise ValueError("A prior prediction CSV is invalid with --mode candidates")
    prior_unit = (
        args.causal_translation_unit
        if args.causal_predictions is not None
        else args.prior_translation_unit
    )
    config = _config(args)
    config.validate()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle = CandidateBundle(args.data)
    arrays = bundle.arrays
    feature_names = list(bundle.manifest["candidate_feature_names"])
    if "total_error" not in feature_names:
        raise ValueError("Candidate bundle lacks the RGB/CAD total_error feature")
    total_error = arrays["features"][..., feature_names.index("total_error")]
    visual_cost = normalized_visual_cost(total_error, arrays["valid"])
    visual_initial, visual_selected = visual_initialization(
        arrays["poses"], arrays["valid"], visual_cost
    )
    prior = (
        _prior_array(bundle, prior_path, prior_unit)
        if uses_prior else None
    )
    initial = prior.copy() if prior is not None else visual_initial
    final = initial.copy()
    selected = visual_selected.copy()
    segment_reports, diagnostics = [], []
    started = time.perf_counter()
    segment_ids = np.unique(arrays["segment_id"])
    if args.max_segments is not None:
        segment_ids = segment_ids[:args.max_segments]
    optimized_mask = np.zeros(len(bundle), dtype=bool)
    for number, segment_id in enumerate(segment_ids, 1):
        indices = np.flatnonzero(arrays["segment_id"] == segment_id)
        times = _segment_time(arrays, indices, list(bundle.manifest["frame_feature_names"]))
        result = optimize_sequence(
            arrays["poses"][indices],
            arrays["valid"][indices],
            visual_cost[indices],
            times,
            initial_poses=initial[indices],
            prior_poses=None if prior is None else prior[indices],
            config=config,
        )
        final[indices] = result.poses
        selected[indices] = result.selected_candidate
        optimized_mask[indices] = True
        segment_reports.append({
            "segment_id": int(segment_id),
            "frames": int(len(indices)),
            "initial_objective": result.initial_objective,
            "final_objective": result.final_objective,
            "outer_iterations": result.outer_iterations_completed,
            "optimizer_success": result.optimizer_success,
            "optimizer_message": result.optimizer_message,
            "optimizer_nfev": result.optimizer_nfev,
            "assignment_changes": result.assignment_changes,
            "solution_accepted": result.solution_accepted,
        })
        print(
            f"segment {number}/{len(segment_ids)} id={segment_id} frames={len(indices)} "
            f"objective={result.initial_objective:.4f}->{result.final_objective:.4f} "
            f"nfev={result.optimizer_nfev} success={result.optimizer_success}",
            flush=True,
        )

    output_rows, initial_rows = [], []
    for index in np.flatnonzero(optimized_mask):
        candidate_index = int(selected[index])
        source = str(arrays["sources"][index, candidate_index])
        confidence = float(np.exp(-visual_cost[index, candidate_index]))
        common = dict(
            scene_id=int(arrays["scene_id"][index]),
            im_id=int(arrays["im_id"][index]),
            track_id=0,
            obj_id=1,
            confidence=confidence,
            elapsed_s=0.0,
        )
        output_rows.append(pose_to_csv_row(
            **common, pose_m=final[index], mode=f"factor_graph_{args.mode}",
            source=f"sequence_factor_graph|{'prior' if uses_prior else 'candidates'}|{source}",
        ))
        initial_rows.append(pose_to_csv_row(
            **common, pose_m=initial[index], mode=f"factor_graph_{args.mode}_initial",
            source=f"sequence_factor_graph|{args.mode}|initial",
        ))
        diagnostics.append({
            "global_index": int(index),
            "scene_id": int(arrays["scene_id"][index]),
            "im_id": int(arrays["im_id"][index]),
            "segment_id": int(arrays["segment_id"][index]),
            "selected_candidate": candidate_index,
            "selected_source": source,
            "selected_visual_cost": float(visual_cost[index, candidate_index]),
            "translation_change_from_initial_m": translation_error_m(final[index], initial[index]),
            "rotation_change_from_initial_deg": rotation_error_deg(final[index], initial[index]),
        })
    write_tracking_csv(args.output_dir / "tracked_predictions.csv", output_rows)
    write_tracking_csv(args.output_dir / "initial_predictions.csv", initial_rows)
    with (args.output_dir / "diagnostics.jsonl").open("w") as handle:
        for row in diagnostics:
            handle.write(json.dumps(row) + "\n")

    selected_indices = np.flatnonzero(optimized_mask)
    has_gt = "ground_truth_pose" in arrays
    report = {
        "format": FORMAT,
        "mode": "prior" if uses_prior else "candidates",
        "requested_mode": args.mode,
        "mode_interpretation": (
            "RGB/CAD candidates only" if not uses_prior
            else "generic external trajectory soft prior plus RGB/CAD candidates"
        ),
        "prior_predictions": str(prior_path) if prior_path is not None else None,
        "ground_truth_used_by_optimizer": False,
        "frame_count": int(len(selected_indices)),
        "segment_count": int(len(segment_reports)),
        "candidate_assignment_changes": int(sum(
            item["assignment_changes"] for item in segment_reports
        )),
        "optimizer_success_fraction": float(np.mean([
            item["optimizer_success"] for item in segment_reports
        ])) if segment_reports else None,
        "solution_accepted_fraction": float(np.mean([
            item["solution_accepted"] for item in segment_reports
        ])) if segment_reports else None,
        "configuration": config.as_dict(),
        "initial_metrics": (
            _pose_metrics(initial[selected_indices], arrays["ground_truth_pose"][selected_indices])
            if has_gt else None
        ),
        "final_metrics": (
            _pose_metrics(final[selected_indices], arrays["ground_truth_pose"][selected_indices])
            if has_gt else None
        ),
        "segment_reports": segment_reports,
        "elapsed_s": time.perf_counter() - started,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "run_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "segment_reports"}, indent=2))


if __name__ == "__main__":
    main()
