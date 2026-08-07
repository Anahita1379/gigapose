"""Export sequence-safe raw GigaPose measurements and GT filter targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from tracking.association import assign_prediction_groups
from tracking.generate_recovery_dataset import iter_gt_frames
from tracking.geometry import pose_from_rt, rotation_error_deg, translation_error_m
from tracking.io import PredictionCSVProvider, WebDatasetSequence
from tracking.rgb_self_recovery.sliding_window.assetto_validation_eval.common import (
    resolve_predictions,
)
from tracking.rgb_self_recovery.sliding_window.sequence import (
    SequenceStamp,
    discontinuity_reason,
    order_sequence_rows,
)

from .features import CONTEXT_FEATURE_NAMES, context_features


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--dataset-dir", type=Path, required=True)
    value.add_argument(
        "--ground-truth-dataset-dir",
        type=Path,
        help=(
            "Optional regenerated-pose dataset. Ground truth is matched by "
            "source run, camera, and source frame rather than scene numbering."
        ),
    )
    value.add_argument("--split", default="test")
    value.add_argument("--predictions", type=Path, required=True)
    value.add_argument(
        "--fallback-baseline-predictions",
        type=Path,
        help=(
            "Original GigaPose predictions used by the learned translation and "
            "rotation fallback heads. If omitted, the input measurement is used."
        ),
    )
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--prediction-translation-unit", choices=("mm", "m"), default="mm")
    value.add_argument("--sequence-max-frame-gap", type=int, default=1)
    value.add_argument("--sequence-max-time-gap-s", type=float, default=0.5)
    value.add_argument("--max-frames", type=int)
    value.add_argument("--overwrite", action="store_true")
    return value


def _ground_truth(dataset_dir: Path, split: str) -> dict[tuple[int, int], np.ndarray]:
    output = {}
    for key, _image, _K, _depth, gt, _extra in iter_gt_frames(
        dataset_dir, split, load_depth=False
    ):
        if len(gt) != 1:
            continue
        scene_id, im_id = (int(part) for part in key.split("_"))
        output[(scene_id, im_id)] = pose_from_rt(
            np.asarray(gt[0]["cam_R_m2c"], dtype=float).reshape(3, 3),
            np.asarray(gt[0]["cam_t_m2c"], dtype=float) * 0.001,
        ).astype(np.float32)
    return output


def _ground_truth_by_source(
    dataset_dir: Path, split: str
) -> dict[tuple[str, str, int], np.ndarray]:
    by_scene = _ground_truth(dataset_dir, split)
    sequence = WebDatasetSequence(dataset_dir, split, load_depth=False)
    output = {}
    for row in sequence.rows:
        scene_key = (int(row["scene_id"]), int(row["im_id"]))
        pose = by_scene.get(scene_key)
        if pose is None:
            continue
        source_key = (
            str(row["source_run"]),
            str(row["camera_id"]),
            int(row["source_frame"]),
        )
        output[source_key] = pose
    return output


def main() -> None:
    args = parser().parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = WebDatasetSequence(args.dataset_dir, args.split, load_depth=False)
    source.rows = list(order_sequence_rows(source.rows))
    predictions_path = resolve_predictions(args.predictions)
    predictions = PredictionCSVProvider(
        predictions_path, args.prediction_translation_unit, "gigapose"
    )
    baseline_path = (
        None
        if args.fallback_baseline_predictions is None
        else resolve_predictions(args.fallback_baseline_predictions)
    )
    baseline_predictions = (
        None
        if baseline_path is None
        else PredictionCSVProvider(
            baseline_path, args.prediction_translation_unit, "gigapose_fallback"
        )
    )
    ground_truth = _ground_truth(args.dataset_dir, args.split)
    regenerated_ground_truth = (
        None
        if args.ground_truth_dataset_dir is None
        else _ground_truth_by_source(args.ground_truth_dataset_dir, args.split)
    )

    records = []
    skipped = []
    previous_saved_stamp = None
    previous_saved_time = None
    segment_id = -1
    for frame_index, frame in enumerate(source):
        if args.max_frames is not None and frame_index >= args.max_frames:
            break
        stamp = SequenceStamp.from_frame(frame)
        key = (int(frame.scene_id), int(frame.im_id))
        gt_pose = ground_truth.get(key)
        if regenerated_ground_truth is not None:
            gt_pose = regenerated_ground_truth.get(
                (str(stamp.key[0]), str(stamp.key[1]), int(stamp.source_frame))
            )
        if gt_pose is None or len(frame.detections) != 1:
            skipped.append({"scene_id": key[0], "im_id": key[1], "reason": "requires one GT and detection"})
            continue
        groups = predictions.groups_for_frame(*key)
        assigned = assign_prediction_groups(
            groups, frame.detections, frame.K, frame.image.shape[:2]
        )
        hypotheses = assigned.get(0, [])
        if not hypotheses:
            skipped.append({"scene_id": key[0], "im_id": key[1], "reason": "no assigned GigaPose pose"})
            continue
        measurement = hypotheses[0]
        baseline = measurement
        if baseline_predictions is not None:
            baseline_groups = baseline_predictions.groups_for_frame(*key)
            baseline_assigned = assign_prediction_groups(
                baseline_groups, frame.detections, frame.K, frame.image.shape[:2]
            )
            baseline_hypotheses = baseline_assigned.get(0, [])
            if not baseline_hypotheses:
                skipped.append({
                    "scene_id": key[0],
                    "im_id": key[1],
                    "reason": "no assigned fallback GigaPose pose",
                })
                continue
            baseline = baseline_hypotheses[0]
        reason = discontinuity_reason(
            previous_saved_stamp,
            stamp,
            max_frame_gap=args.sequence_max_frame_gap,
            max_time_gap_s=args.sequence_max_time_gap_s,
        )
        if previous_saved_stamp is None or reason is not None:
            segment_id += 1
            delta_time_s = 0.0
        else:
            delta_time_s = (
                0.0 if previous_saved_time is None or stamp.time_s is None
                else max(float(stamp.time_s - previous_saved_time), 0.0)
            )
        pose = np.asarray(measurement.pose, dtype=np.float32)
        baseline_pose = np.asarray(baseline.pose, dtype=np.float32)
        records.append({
            "scene_id": key[0],
            "im_id": key[1],
            "source_run": stamp.key[0],
            "camera_id": stamp.key[1],
            "source_frame": stamp.source_frame,
            "time_s": np.nan if stamp.time_s is None else stamp.time_s,
            "delta_time_s": delta_time_s,
            "segment_id": segment_id,
            "measurement_pose": pose,
            "baseline_pose": baseline_pose,
            "ground_truth_pose": gt_pose,
            "context_features": context_features(frame, frame.detections[0], measurement),
            "measurement_score": float(measurement.measurement_score),
            "baseline_score": float(baseline.measurement_score),
            "raw_translation_error_m": translation_error_m(pose, gt_pose),
            "raw_rotation_error_deg": rotation_error_deg(pose, gt_pose),
        })
        previous_saved_stamp = stamp
        previous_saved_time = stamp.time_s
        if len(records) % 250 == 0:
            print(f"exported {len(records)}/{len(source)} measurements", flush=True)
    if not records:
        raise RuntimeError("No measurements were exported")

    scalar_names = (
        "scene_id", "im_id", "source_frame", "time_s", "delta_time_s",
        "segment_id", "measurement_score", "baseline_score", "raw_translation_error_m",
        "raw_rotation_error_deg",
    )
    arrays = {name: np.asarray([row[name] for row in records]) for name in scalar_names}
    for name in (
        "measurement_pose", "baseline_pose", "ground_truth_pose", "context_features"
    ):
        arrays[name] = np.asarray([row[name] for row in records])
    arrays["source_run"] = np.asarray([row["source_run"] for row in records], dtype="<U256")
    arrays["camera_id"] = np.asarray([row["camera_id"] for row in records], dtype="<U64")
    np.savez_compressed(args.output_dir / "measurements.npz", **arrays)
    report = {
        "format": "rgb_self_recovery_gru_kalman_measurements_v2",
        "dataset_dir": str(args.dataset_dir),
        "ground_truth_dataset_dir": (
            None
            if args.ground_truth_dataset_dir is None
            else str(args.ground_truth_dataset_dir)
        ),
        "split": args.split,
        "predictions": str(predictions_path),
        "fallback_baseline_predictions": (
            None if baseline_path is None else str(baseline_path)
        ),
        "frame_count": len(records),
        "run_count": len(set(arrays["source_run"].tolist())),
        "sequence_count": len(set(zip(arrays["source_run"].tolist(), arrays["camera_id"].tolist()))),
        "segment_count": len(set(arrays["segment_id"].tolist())),
        "context_feature_names": list(CONTEXT_FEATURE_NAMES),
        "skipped_count": len(skipped),
        "skipped": skipped,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
