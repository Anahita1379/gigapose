"""Validate a prepared Assetto development-evaluation dataset end to end."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tracking.generate_recovery_dataset import iter_gt_frames
from tracking.io import WebDatasetSequence


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--dataset-dir", type=Path, required=True)
    value.add_argument("--split", default="test")
    value.add_argument("--output", type=Path)
    return value


def validate(dataset_dir: Path, split: str = "test") -> dict[str, object]:
    frame_map_path = dataset_dir / "frame_map.json"
    targets_path = dataset_dir / "test_targets_bop19.json"
    metadata_path = dataset_dir / "evaluation_metadata.json"
    detection_path = (
        dataset_dir.parent
        / "cnos-fastsam"
        / f"cnos-fastsam_{dataset_dir.name}-test.json"
    )
    template_path = dataset_dir.parent / "templates" / dataset_dir.name
    for path in (frame_map_path, targets_path, metadata_path, detection_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not template_path.is_dir():
        raise FileNotFoundError(template_path)

    rows = json.loads(frame_map_path.read_text())
    targets = json.loads(targets_path.read_text())
    detections = json.loads(detection_path.read_text())
    expected_keys = {(int(row["scene_id"]), int(row["im_id"])) for row in rows}
    target_keys = {(int(row["scene_id"]), int(row["im_id"])) for row in targets}
    detection_keys = {
        (int(row["scene_id"]), int(row["image_id"])) for row in detections
    }
    if expected_keys != target_keys or expected_keys != detection_keys:
        raise ValueError(
            "frame_map, targets, and detections do not describe identical frame keys"
        )
    timestamps = np.asarray([float(row["timestamp_ms"]) for row in rows])
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("frame_map timestamps are not strictly increasing")

    sequence = WebDatasetSequence(dataset_dir, split, load_depth=False)
    sequence_count = detection_count = mask_pixels = 0
    sequence_keys = set()
    for frame in sequence:
        sequence_count += 1
        sequence_keys.add((frame.scene_id, frame.im_id))
        if len(frame.detections) != 1:
            raise ValueError(
                f"Expected one frame_map detection for {frame.scene_id}/{frame.im_id}; "
                f"found {len(frame.detections)}"
            )
        detection_count += len(frame.detections)
        pixels = int(np.asarray(frame.detections[0].mask, dtype=bool).sum())
        if pixels <= 0:
            raise ValueError(f"Empty decoded mask for {frame.scene_id}/{frame.im_id}")
        mask_pixels += pixels
    if sequence_keys != expected_keys:
        raise ValueError("WebDatasetSequence did not load every frame_map key")

    gt_count = gt_instance_count = 0
    gt_keys = set()
    for key, image, K, _depth, gt, extra in iter_gt_frames(
        dataset_dir, split, load_depth=False
    ):
        gt_count += 1
        scene_id, im_id = (int(value) for value in key.split("_"))
        gt_keys.add((scene_id, im_id))
        if image.ndim != 3 or K.shape != (3, 3):
            raise ValueError(f"Invalid image/intrinsics in {key}")
        if len(gt) != 1 or len(extra["infos"]) != 1 or len(extra["masks"]) != 1:
            raise ValueError(f"Expected one complete GT instance in {key}")
        pose_depth_m = float(gt[0]["cam_t_m2c"][2]) * 0.001
        if pose_depth_m <= 0:
            raise ValueError(f"Non-positive GT depth in {key}")
        gt_instance_count += len(gt)
    if gt_keys != expected_keys:
        raise ValueError("GT iterator did not load every expected frame")

    report = {
        "format": "rgb_self_recovery_assetto_validation_check_v1",
        "dataset_dir": str(dataset_dir),
        "split": split,
        "frame_map_count": len(rows),
        "target_count": len(targets),
        "cnos_detection_count": len(detections),
        "sequence_frame_count": sequence_count,
        "sequence_detection_count": detection_count,
        "sequence_mask_pixels_total": mask_pixels,
        "ground_truth_frame_count": gt_count,
        "ground_truth_instance_count": gt_instance_count,
        "first_timestamp_ms": float(timestamps[0]),
        "last_timestamp_ms": float(timestamps[-1]),
        "template_path": str(template_path),
        "valid": True,
        "interpretation": "development validation set; not an unbiased benchmark",
    }
    return report


def main() -> None:
    args = parser().parse_args()
    report = validate(args.dataset_dir, args.split)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
