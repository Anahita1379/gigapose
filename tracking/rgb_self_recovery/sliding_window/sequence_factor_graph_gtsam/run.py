"""Run isolated batch GTSAM optimization over RGB/CAD pose candidates."""

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
    GTSAMGraphConfig,
    normalized_visual_cost,
    optimize_sequence,
    require_gtsam,
    visual_initialization,
)


FORMAT = "rgb_self_recovery_sequence_factor_graph_gtsam_v1"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--data", type=Path, required=True, help="RGB/CAD candidate bundle")
    value.add_argument("--mode", choices=("candidates", "prior"), required=True)
    value.add_argument(
        "--prior-predictions", type=Path,
        help="Translation-only cascade, full-pose cascade, or transformer CSV.",
    )
    value.add_argument("--prior-translation-unit", choices=("mm", "m"), default="mm")
    value.add_argument(
        "--prior-components", choices=("translation", "full"), default="full",
        help=(
            "translation ignores the prior CSV orientation; full adds both "
            "translation and rotation prior factors."
        ),
    )
    value.add_argument(
        "--world-camera-transforms", type=Path,
        help=(
            "Optional JSONL containing T_world_camera/T_map_camera, or "
            "T_map_lidar and T_lidar_camera, keyed by scene_id and im_id."
        ),
    )
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
    value.add_argument("--maximum-iterations", type=int, default=100)
    value.add_argument("--maximum-translation-step-m", type=float, default=20.0)
    value.add_argument("--maximum-rotation-step-deg", type=float, default=180.0)
    value.add_argument(
        "--robust-loss", choices=("none", "huber", "cauchy", "tukey"), default="huber"
    )
    value.add_argument("--robust-scale", type=float, default=1.0)
    value.add_argument("--solver-tolerance", type=float, default=1e-5)
    value.add_argument("--numerical-derivative-step", type=float, default=1e-5)
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
        missing = {"scene_id", "im_id", "R", "t"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Prior CSV is missing columns: {sorted(missing)}")
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
    values, missing = [], []
    for scene_id, im_id in zip(bundle.arrays["scene_id"], bundle.arrays["im_id"]):
        key = (int(scene_id), int(im_id))
        if key not in lookup:
            missing.append(key)
        else:
            values.append(lookup[key])
    if missing:
        raise ValueError(
            f"Prior CSV is missing {len(missing)} bundle frames; first={missing[:5]}"
        )
    return np.asarray(values, dtype=float)


def _matrix(row: dict, names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        if name not in row or row[name] is None:
            continue
        value = row[name]
        if isinstance(value, str):
            parsed = np.fromstring(value, sep=" ", dtype=float)
        else:
            parsed = np.asarray(value, dtype=float)
        if parsed.size != 16:
            raise ValueError(f"{name} must contain a 4x4 matrix")
        return parsed.reshape(4, 4)
    return None


def _load_world_camera(path: Path) -> dict[tuple[int, int], np.ndarray]:
    output = {}
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["scene_id"]), int(row["im_id"]))
            transform = _matrix(row, ("T_world_camera", "T_map_camera"))
            if transform is None:
                world_lidar = _matrix(row, ("T_world_lidar", "T_map_lidar"))
                lidar_camera = _matrix(row, ("T_lidar_camera",))
                if world_lidar is not None and lidar_camera is not None:
                    transform = world_lidar @ lidar_camera
            if transform is None:
                raise ValueError(
                    f"No usable world-camera transform at {path}:{line_number}"
                )
            if key in output:
                raise ValueError(f"Duplicate world-camera transform for frame {key}")
            output[key] = transform
    return output


def _world_camera_array(bundle: CandidateBundle, path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    lookup = _load_world_camera(path)
    values, missing = [], []
    for scene_id, im_id in zip(bundle.arrays["scene_id"], bundle.arrays["im_id"]):
        key = (int(scene_id), int(im_id))
        if key not in lookup:
            missing.append(key)
        else:
            values.append(lookup[key])
    if missing:
        raise ValueError(
            f"World-camera JSONL is missing {len(missing)} bundle frames; first={missing[:5]}"
        )
    return np.asarray(values, dtype=float)


def _transform_poses(left: np.ndarray, poses: np.ndarray) -> np.ndarray:
    return np.matmul(left[:, None], poses) if poses.ndim == 4 else np.matmul(left, poses)


def _segment_time(arrays: dict, indices: np.ndarray, feature_names: list[str]) -> np.ndarray:
    raw = np.asarray(arrays["time_s"][indices], dtype=float)
    output = np.zeros(len(indices), dtype=float)
    delta_index = feature_names.index("delta_time_s") if "delta_time_s" in feature_names else None
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
    return {
        "translation_error_m": _stats([
            translation_error_m(a, b) for a, b in zip(poses, ground_truth)
        ]),
        "rotation_error_deg": _stats([
            rotation_error_deg(a, b) for a, b in zip(poses, ground_truth)
        ]),
    }


def _config(args) -> GTSAMGraphConfig:
    names = set(GTSAMGraphConfig.__dataclass_fields__)
    return GTSAMGraphConfig(**{name: getattr(args, name) for name in names})


def main() -> None:
    args = parser().parse_args()
    require_gtsam()
    if args.mode == "prior" and args.prior_predictions is None:
        raise ValueError("--prior-predictions is required for --mode prior")
    if args.mode == "candidates" and args.prior_predictions is not None:
        raise ValueError("--prior-predictions is invalid for --mode candidates")
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
    visual_cost = normalized_visual_cost(
        arrays["features"][..., feature_names.index("total_error")], arrays["valid"]
    )
    visual_initial, visual_selected = visual_initialization(
        arrays["poses"], arrays["valid"], visual_cost
    )
    prior = (
        _prior_array(bundle, args.prior_predictions, args.prior_translation_unit)
        if args.prior_predictions is not None else None
    )
    initial_camera = prior.copy() if prior is not None else visual_initial
    world_camera = _world_camera_array(bundle, args.world_camera_transforms)
    candidate_working = np.asarray(arrays["poses"], dtype=float)
    initial_working = initial_camera.copy()
    prior_working = None if prior is None else prior.copy()
    if world_camera is not None:
        candidate_working = _transform_poses(world_camera, candidate_working)
        initial_working = _transform_poses(world_camera, initial_working)
        prior_working = None if prior is None else _transform_poses(world_camera, prior)

    final_working = initial_working.copy()
    selected = visual_selected.copy()
    segment_reports, diagnostics = [], []
    optimized_mask = np.zeros(len(bundle), dtype=bool)
    segment_ids = np.unique(arrays["segment_id"])
    if args.max_segments is not None:
        segment_ids = segment_ids[: args.max_segments]
    started = time.perf_counter()
    for number, segment_id in enumerate(segment_ids, 1):
        indices = np.flatnonzero(arrays["segment_id"] == segment_id)
        times = _segment_time(
            arrays, indices, list(bundle.manifest["frame_feature_names"])
        )
        result = optimize_sequence(
            candidate_working[indices], arrays["valid"][indices], visual_cost[indices], times,
            initial_poses=initial_working[indices],
            prior_poses=None if prior_working is None else prior_working[indices],
            use_prior_rotation=args.prior_components == "full",
            config=config,
        )
        final_working[indices] = result.poses
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
            "optimizer_iterations": result.optimizer_iterations,
            "assignment_changes": result.assignment_changes,
            "solution_accepted": result.solution_accepted,
        })
        print(
            f"segment {number}/{len(segment_ids)} id={segment_id} frames={len(indices)} "
            f"objective={result.initial_objective:.4f}->{result.final_objective:.4f} "
            f"iterations={result.optimizer_iterations} success={result.optimizer_success} "
            f"accepted={result.solution_accepted}", flush=True,
        )

    if world_camera is None:
        final_camera = final_working
    else:
        final_camera = _transform_poses(np.linalg.inv(world_camera), final_working)
    output_rows, initial_rows = [], []
    for index in np.flatnonzero(optimized_mask):
        candidate_index = int(selected[index])
        source = str(arrays["sources"][index, candidate_index])
        confidence = float(np.exp(-visual_cost[index, candidate_index]))
        common = dict(
            scene_id=int(arrays["scene_id"][index]), im_id=int(arrays["im_id"][index]),
            track_id=0, obj_id=1, confidence=confidence, elapsed_s=0.0,
        )
        output_rows.append(pose_to_csv_row(
            **common, pose_m=final_camera[index], mode=f"gtsam_factor_graph_{args.mode}",
            source=f"gtsam_sequence_graph|{args.mode}|{source}",
        ))
        initial_rows.append(pose_to_csv_row(
            **common, pose_m=initial_camera[index], mode=f"gtsam_factor_graph_{args.mode}_initial",
            source=f"gtsam_sequence_graph|{args.mode}|initial",
        ))
        diagnostics.append({
            "global_index": int(index),
            "scene_id": int(arrays["scene_id"][index]),
            "im_id": int(arrays["im_id"][index]),
            "segment_id": int(arrays["segment_id"][index]),
            "selected_candidate": candidate_index,
            "selected_source": source,
            "selected_visual_cost": float(visual_cost[index, candidate_index]),
            "translation_change_from_initial_m": translation_error_m(
                final_camera[index], initial_camera[index]
            ),
            "rotation_change_from_initial_deg": rotation_error_deg(
                final_camera[index], initial_camera[index]
            ),
        })
    write_tracking_csv(args.output_dir / "tracked_predictions.csv", output_rows)
    write_tracking_csv(args.output_dir / "initial_predictions.csv", initial_rows)
    with (args.output_dir / "diagnostics.jsonl").open("w") as handle:
        for row in diagnostics:
            handle.write(json.dumps(row) + "\n")

    indices = np.flatnonzero(optimized_mask)
    has_gt = "ground_truth_pose" in arrays
    report = {
        "format": FORMAT,
        "mode": args.mode,
        "mode_interpretation": (
            "RGB/CAD candidates only" if prior is None
            else "external trajectory soft prior plus RGB/CAD candidates"
        ),
        "prior_predictions": str(args.prior_predictions) if args.prior_predictions else None,
        "prior_components": args.prior_components if prior is not None else None,
        "optimization_coordinate_frame": "world" if world_camera is not None else "camera",
        "world_camera_transforms": str(args.world_camera_transforms) if args.world_camera_transforms else None,
        "ground_truth_used_by_optimizer": False,
        "ground_truth_available_for_report": has_gt,
        "frame_count": int(len(indices)),
        "segment_count": int(len(segment_reports)),
        "candidate_assignment_changes": int(sum(x["assignment_changes"] for x in segment_reports)),
        "optimizer_success_fraction": float(np.mean([
            x["optimizer_success"] for x in segment_reports
        ])) if segment_reports else None,
        "solution_accepted_fraction": float(np.mean([
            x["solution_accepted"] for x in segment_reports
        ])) if segment_reports else None,
        "configuration": config.as_dict(),
        "initial_metrics": _pose_metrics(
            initial_camera[indices], arrays["ground_truth_pose"][indices]
        ) if has_gt else None,
        "final_metrics": _pose_metrics(
            final_camera[indices], arrays["ground_truth_pose"][indices]
        ) if has_gt else None,
        "segment_reports": segment_reports,
        "elapsed_s": time.perf_counter() - started,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "run_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "segment_reports"}, indent=2))


if __name__ == "__main__":
    main()
