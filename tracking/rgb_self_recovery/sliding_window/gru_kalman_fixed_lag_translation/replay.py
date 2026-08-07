"""Replay GRU segments after fixed-lag translation measurement repairs."""

from __future__ import annotations

import argparse
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


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--data", type=Path, required=True)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--fixed-lag-run", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--device", default="cuda")
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


def _rows(arrays, poses, mode, source):
    return [
        pose_to_csv_row(
            scene_id=int(arrays["scene_id"][index]),
            im_id=int(arrays["im_id"][index]),
            track_id=0,
            obj_id=1,
            confidence=float(arrays["measurement_score"][index]),
            pose_m=poses[index],
            mode=mode,
            source=source,
            elapsed_s=0.0,
        )
        for index in range(len(poses))
    ]


def main() -> None:
    args = parser().parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle = MeasurementBundle(args.data)
    arrays = bundle.arrays
    replay_path = args.fixed_lag_run / "replay_inputs.npz"
    if not replay_path.is_file():
        raise FileNotFoundError(f"Fixed-lag replay inputs are missing: {replay_path}")
    with np.load(replay_path, allow_pickle=False) as payload:
        corrected_measurement = np.asarray(
            payload["corrected_measurement_pose"], dtype=np.float32
        )
        repair_mask = np.asarray(payload["repair_mask"], dtype=bool)
        replay_start = np.asarray(payload["replay_start_global_index"], dtype=np.int64)
        selected_source = np.asarray(payload["selected_source"])
        for key in ("scene_id", "im_id", "segment_id"):
            if key in payload and not np.array_equal(payload[key], arrays[key]):
                raise ValueError(f"Replay inputs and data disagree on {key}")
    if corrected_measurement.shape != arrays["measurement_pose"].shape:
        raise ValueError("Replay inputs and data contain different frame counts")
    if repair_mask.shape != (len(bundle),) or replay_start.shape != (len(bundle),):
        raise ValueError("Replay masks have the wrong shape")

    model, checkpoint = load_filter(args.checkpoint, args.device)
    if checkpoint["context_feature_names"] != bundle.manifest["context_feature_names"]:
        raise ValueError("Checkpoint and measurement context schemas differ")
    causal, _causal_gains = run_filter_segments(model, arrays, args.device)

    replay_arrays = dict(arrays)
    replay_context = arrays["context_features"].copy()
    feature_names = list(bundle.manifest["context_feature_names"])
    if "measurement_depth_m" in feature_names:
        depth_index = feature_names.index("measurement_depth_m")
        replay_context[:, depth_index] = corrected_measurement[:, 2, 3]
    replay_arrays["context_features"] = replay_context
    replay_full, replay_gains = run_filter_segments(
        model,
        replay_arrays,
        args.device,
        measurement_pose=corrected_measurement,
    )
    replay_translation_only = replay_full.copy()
    replay_translation_only[:, :3, :3] = causal[:, :3, :3]

    write_tracking_csv(
        args.output_dir / "causal_predictions.csv",
        _rows(arrays, causal, "gru_kalman_causal_first_pass", "gru_kalman|causal"),
    )
    write_tracking_csv(
        args.output_dir / "replayed_full_pose_predictions.csv",
        _rows(arrays, replay_full, "gru_state_replay_full_pose", "gru_kalman|state_replay"),
    )
    translation_rows = _rows(
        arrays,
        replay_translation_only,
        "gru_state_replay_translation_only",
        "gru_kalman|state_replay|causal_rotation",
    )
    write_tracking_csv(args.output_dir / "replayed_translation_only_predictions.csv", translation_rows)
    write_tracking_csv(args.output_dir / "tracked_predictions.csv", translation_rows)

    diagnostics = []
    metric_values = {
        f"{name}_{component}": []
        for name in ("causal", "replay_full", "replay_translation_only")
        for component in ("t", "r")
    }
    has_ground_truth = "ground_truth_pose" in arrays
    for index in range(len(bundle)):
        row = {
            "scene_id": int(arrays["scene_id"][index]),
            "im_id": int(arrays["im_id"][index]),
            "source_run": str(arrays["source_run"][index]),
            "camera_id": str(arrays["camera_id"][index]),
            "source_frame": int(arrays["source_frame"][index]),
            "segment_id": int(arrays["segment_id"][index]),
            "measurement_repaired_before_replay": int(repair_mask[index]),
            "selected_source": str(selected_source[index]),
            "replay_start_global_index": int(replay_start[index]),
            "causal_to_replay_translation_change_m": translation_error_m(
                replay_translation_only[index], causal[index]
            ),
            "causal_to_replay_rotation_change_deg": rotation_error_deg(
                replay_full[index], causal[index]
            ),
            "mean_replay_pose_gain": float(replay_gains[index, :6].mean()),
            "mean_replay_velocity_gain": float(replay_gains[index, 6:].mean()),
        }
        if has_ground_truth:
            gt = arrays["ground_truth_pose"][index]
            for name, poses in (
                ("causal", causal),
                ("replay_full", replay_full),
                ("replay_translation_only", replay_translation_only),
            ):
                translation = translation_error_m(poses[index], gt)
                rotation = rotation_error_deg(poses[index], gt)
                row[f"{name}_translation_error_m"] = translation
                row[f"{name}_rotation_error_deg"] = rotation
                metric_values[f"{name}_t"].append(translation)
                metric_values[f"{name}_r"].append(rotation)
        diagnostics.append(row)
    with (args.output_dir / "replay_diagnostics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)

    report = {
        "format": "gru_kalman_fixed_lag_state_replay_v1",
        "frame_count": len(bundle),
        "segment_count": int(len(np.unique(arrays["segment_id"]))),
        "repaired_measurement_count": int(repair_mask.sum()),
        "state_replay_scope": "complete_segment",
        "recommended_output": "replayed_translation_only_predictions.csv",
        "tracked_predictions_alias": "replayed_translation_only_predictions.csv",
        "full_pose_replay_is_diagnostic": True,
        "ground_truth_used_by_replay": False,
        "model_had_fallback_heads_but_they_were_ignored": bool(model.use_fallback_heads),
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
