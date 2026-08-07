"""Run GRU-Kalman followed by a five-frame translation-only fixed-lag smoother."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from tracking.geometry import rotation_error_deg, translation_error_m
from tracking.io import pose_to_csv_row, write_tracking_csv
from tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.dataset import MeasurementBundle
from tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.train import load_filter

from .inference import run_filter_segments
from .smoother import FixedLagConfig, smooth_segment


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--data", type=Path, required=True)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--device", default="cuda")
    value.add_argument("--window-size", type=int, default=5)
    value.add_argument("--lag", type=int, default=2)
    value.add_argument("--minimum-gate-m", type=float, default=1.5)
    value.add_argument("--depth-gate-fraction", type=float, default=0.02)
    value.add_argument("--mad-multiplier", type=float, default=3.5)
    value.add_argument("--maximum-neighbor-fit-residual-m", type=float, default=4.0)
    value.add_argument("--minimum-candidate-improvement-m", type=float, default=1.0)
    value.add_argument("--candidate-acceptance-gate-multiplier", type=float, default=1.5)
    value.add_argument("--selector-penalty-m", type=float, default=0.15)
    value.add_argument("--gigapose-penalty-m", type=float, default=0.25)
    value.add_argument("--allow-motion-only-repair", action="store_true")
    value.add_argument("--motion-only-penalty-m", type=float, default=1.0)
    value.add_argument("--motion-blend", type=float, default=0.10)
    value.add_argument("--maximum-repair-m", type=float, default=30.0)
    value.add_argument("--acceleration-regularization", type=float, default=0.05)
    value.add_argument("--irls-iterations", type=int, default=4)
    value.add_argument("--overwrite", action="store_true")
    return value


def _summary(values) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0, "mean": None, "rmse": None, "median": None, "p90": None}
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    }


def _configuration(args) -> FixedLagConfig:
    return FixedLagConfig(
        window_size=args.window_size,
        lag=args.lag,
        minimum_gate_m=args.minimum_gate_m,
        depth_gate_fraction=args.depth_gate_fraction,
        mad_multiplier=args.mad_multiplier,
        maximum_neighbor_fit_residual_m=args.maximum_neighbor_fit_residual_m,
        minimum_candidate_improvement_m=args.minimum_candidate_improvement_m,
        candidate_acceptance_gate_multiplier=args.candidate_acceptance_gate_multiplier,
        selector_penalty_m=args.selector_penalty_m,
        gigapose_penalty_m=args.gigapose_penalty_m,
        allow_motion_only_repair=args.allow_motion_only_repair,
        motion_only_penalty_m=args.motion_only_penalty_m,
        motion_blend=args.motion_blend,
        maximum_repair_m=args.maximum_repair_m,
        acceleration_regularization=args.acceleration_regularization,
        irls_iterations=args.irls_iterations,
    )


def main() -> None:
    args = parser().parse_args()
    config = _configuration(args)
    config.validate()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle = MeasurementBundle(args.data)
    arrays = bundle.arrays
    model, checkpoint = load_filter(args.checkpoint, args.device)
    if checkpoint["context_feature_names"] != bundle.manifest["context_feature_names"]:
        raise ValueError("Checkpoint and measurement context schemas differ")

    count = len(bundle)
    causal, gains = run_filter_segments(model, arrays, args.device)

    final = causal.copy()
    repaired = np.zeros(count, dtype=bool)
    selected_source = np.full(count, "gru_kalman", dtype="<U32")
    motion_prediction = np.full((count, 3), np.nan)
    estimated_velocity = np.full((count, 3), np.nan)
    estimated_acceleration = np.full((count, 3), np.nan)
    causal_residual = np.full(count, np.nan)
    selected_residual = np.full(count, np.nan)
    adaptive_gate = np.full(count, np.nan)
    neighbor_fit_residual = np.full(count, np.nan)
    repair_magnitude = np.zeros(count)
    replay_start = np.full(count, -1, dtype=np.int64)

    for segment_id in np.unique(arrays["segment_id"]):
        indices = np.flatnonzero(arrays["segment_id"] == segment_id)
        times = arrays["time_s"][indices].astype(float)
        if len(times) > 1 and np.any(np.diff(times) <= 0):
            times = np.cumsum(arrays["delta_time_s"][indices].astype(float))
        result = smooth_segment(
            causal[indices, :3, 3],
            arrays["measurement_pose"][indices, :3, 3],
            arrays["baseline_pose"][indices, :3, 3],
            times,
            config,
        )
        final[indices, :3, 3] = result.positions
        repaired[indices] = result.repair_mask
        selected_source[indices] = result.selected_source
        motion_prediction[indices] = result.motion_prediction
        estimated_velocity[indices] = result.estimated_velocity
        estimated_acceleration[indices] = result.estimated_acceleration
        causal_residual[indices] = result.causal_residual_m
        selected_residual[indices] = result.selected_residual_m
        adaptive_gate[indices] = result.adaptive_gate_m
        neighbor_fit_residual[indices] = result.neighbor_fit_residual_m
        repair_magnitude[indices] = result.repair_magnitude_m
        valid_replay = result.replay_start_local_index >= 0
        replay_start[indices[valid_replay]] = indices[
            result.replay_start_local_index[valid_replay]
        ]

    output_rows = []
    causal_rows = []
    diagnostics = []
    metric_values = {
        name: [] for name in (
            "measurement_t", "measurement_r", "gigapose_t", "gigapose_r",
            "causal_t", "causal_r", "final_t", "final_r",
        )
    }
    has_ground_truth = "ground_truth_pose" in arrays
    for index in range(count):
        output_rows.append(pose_to_csv_row(
            scene_id=int(arrays["scene_id"][index]),
            im_id=int(arrays["im_id"][index]),
            track_id=0,
            obj_id=1,
            confidence=float(arrays["measurement_score"][index]),
            pose_m=final[index],
            mode=("fixed_lag_translation_repair" if repaired[index] else "gru_kalman"),
            source=f"gru_kalman|fixed_lag_translation|{selected_source[index]}",
            elapsed_s=0.0,
        ))
        causal_rows.append(pose_to_csv_row(
            scene_id=int(arrays["scene_id"][index]),
            im_id=int(arrays["im_id"][index]),
            track_id=0,
            obj_id=1,
            confidence=float(arrays["measurement_score"][index]),
            pose_m=causal[index],
            mode="gru_kalman_causal_first_pass",
            source="gru_kalman|causal_first_pass",
            elapsed_s=0.0,
        ))
        row = {
            "scene_id": int(arrays["scene_id"][index]),
            "im_id": int(arrays["im_id"][index]),
            "source_run": str(arrays["source_run"][index]),
            "camera_id": str(arrays["camera_id"][index]),
            "source_frame": int(arrays["source_frame"][index]),
            "segment_id": int(arrays["segment_id"][index]),
            "fixed_lag_repaired": int(repaired[index]),
            "selected_source": str(selected_source[index]),
            "causal_motion_residual_m": causal_residual[index],
            "selected_motion_residual_m": selected_residual[index],
            "adaptive_gate_m": adaptive_gate[index],
            "neighbor_fit_residual_m": neighbor_fit_residual[index],
            "repair_magnitude_m": repair_magnitude[index],
            "estimated_speed_mps": float(np.linalg.norm(estimated_velocity[index])),
            "estimated_acceleration_mps2": float(np.linalg.norm(estimated_acceleration[index])),
            "replay_start_global_index": int(replay_start[index]),
            "mean_pose_gain": float(gains[index, :6].mean()),
            "mean_velocity_gain": float(gains[index, 6:].mean()),
        }
        if has_ground_truth:
            gt = arrays["ground_truth_pose"][index]
            poses = {
                "measurement": arrays["measurement_pose"][index],
                "gigapose": arrays["baseline_pose"][index],
                "causal": causal[index],
                "final": final[index],
            }
            for name, pose in poses.items():
                translation = translation_error_m(pose, gt)
                rotation = rotation_error_deg(pose, gt)
                row[f"{name}_translation_error_m"] = translation
                row[f"{name}_rotation_error_deg"] = rotation
                metric_values[f"{name}_t"].append(translation)
                metric_values[f"{name}_r"].append(rotation)
        diagnostics.append(row)

    write_tracking_csv(args.output_dir / "tracked_predictions.csv", output_rows)
    write_tracking_csv(args.output_dir / "causal_predictions.csv", causal_rows)
    with (args.output_dir / "fixed_lag_diagnostics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)

    replay_measurements = arrays["measurement_pose"].copy()
    replay_measurements[repaired, :3, 3] = final[repaired, :3, 3]
    np.savez_compressed(
        args.output_dir / "replay_inputs.npz",
        scene_id=arrays["scene_id"],
        im_id=arrays["im_id"],
        segment_id=arrays["segment_id"],
        corrected_measurement_pose=replay_measurements,
        repair_mask=repaired,
        replay_start_global_index=replay_start,
        selected_source=selected_source,
    )
    repaired_translation_improvements = (
        np.asarray([
            row["causal_translation_error_m"] - row["final_translation_error_m"]
            for row in diagnostics if row["fixed_lag_repaired"]
        ], dtype=float)
        if has_ground_truth else np.asarray([], dtype=float)
    )
    report = {
        "format": "gru_kalman_fixed_lag_translation_v1",
        "frame_count": count,
        "segment_count": int(len(np.unique(arrays["segment_id"]))),
        "repair_count": int(repaired.sum()),
        "repair_fraction": float(repaired.mean()),
        "selected_source_counts": dict(Counter(selected_source[repaired].tolist())),
        "mean_repair_magnitude_m": (
            float(repair_magnitude[repaired].mean()) if repaired.any() else 0.0
        ),
        "maximum_repair_magnitude_m": float(repair_magnitude.max()),
        "repaired_translation_better_fraction": (
            float(np.mean(repaired_translation_improvements > 0))
            if len(repaired_translation_improvements) else None
        ),
        "repaired_translation_improvement_m": _summary(
            repaired_translation_improvements
        ),
        "rotation_modified": False,
        "ground_truth_used_by_smoother": False,
        "state_replay_performed": False,
        "replay_inputs_written": True,
        "model_had_fallback_heads_but_they_were_ignored": bool(model.use_fallback_heads),
        "configuration": config.__dict__,
        "metrics": (
            {name: _summary(values) for name, values in metric_values.items()}
            if has_ground_truth else None
        ),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "run_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
