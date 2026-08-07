"""Validate a multi-run GRU Assetto dataset and its temporal boundaries."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from tracking.generate_recovery_dataset import iter_gt_frames
from tracking.io import WebDatasetSequence


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--dataset-dir", type=Path, required=True)
    value.add_argument("--split", default="test")
    value.add_argument("--minimum-mask-pixels", type=int, default=64)
    value.add_argument(
        "--maximum-sequence-loader-frames",
        type=int,
        default=512,
        help=(
            "Evenly sample this many frames for the random-access sequential-loader "
            "check; all keys and all GT records are still validated. Use 0 for all."
        ),
    )
    value.add_argument("--output", type=Path)
    return value


def validate(
    dataset_dir: Path,
    split: str,
    minimum_mask_pixels: int,
    maximum_sequence_loader_frames: int = 512,
) -> dict[str, object]:
    frame_map_path = dataset_dir / "frame_map.json"
    targets_path = dataset_dir / "test_targets_bop19.json"
    metadata_path = dataset_dir / "evaluation_metadata.json"
    detection_path = (
        dataset_dir.parent / "cnos-fastsam" /
        f"cnos-fastsam_{dataset_dir.name}-{split}.json"
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
    expected = {(int(row["scene_id"]), int(row["im_id"])) for row in rows}
    target_keys = {(int(row["scene_id"]), int(row["im_id"])) for row in targets}
    detection_keys = {
        (int(row["scene_id"]), int(row["image_id"])) for row in detections
    }
    if len(expected) != len(rows):
        raise ValueError("frame_map contains duplicate scene/image keys")
    if expected != target_keys or expected != detection_keys:
        raise ValueError("frame_map, targets, and detections have different keys")

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source_run"]), str(row["camera_id"]))].append(row)
    sequence_report = []
    for (run, camera), sequence_rows in sorted(grouped.items()):
        sequence_rows.sort(key=lambda row: int(row["source_frame"]))
        frames = np.asarray([int(row["source_frame"]) for row in sequence_rows])
        times = np.asarray([float(row["timestamp_ms"]) for row in sequence_rows])
        if len(frames) > 1 and np.any(np.diff(frames) <= 0):
            raise ValueError(f"Nonmonotonic source frames in {run}/{camera}")
        if len(times) > 1 and np.any(np.diff(times) <= 0):
            raise ValueError(f"Nonmonotonic timestamps in {run}/{camera}")
        segments = 1 + int(np.sum(np.diff(frames) != 1))
        scene_ids = {int(row["scene_id"]) for row in sequence_rows}
        if len(scene_ids) != 1:
            raise ValueError(f"Sequence {run}/{camera} spans multiple BOP scenes")
        sequence_report.append({
            "source_run": run,
            "camera_id": camera,
            "scene_id": next(iter(scene_ids)),
            "frame_count": len(sequence_rows),
            "segment_count": segments,
        })

    sequence = WebDatasetSequence(dataset_dir, split, load_depth=False)
    if maximum_sequence_loader_frames < 0:
        raise ValueError("--maximum-sequence-loader-frames cannot be negative")
    if maximum_sequence_loader_frames and len(sequence.rows) > maximum_sequence_loader_frames:
        indices = np.linspace(
            0, len(sequence.rows) - 1, maximum_sequence_loader_frames, dtype=int
        )
        sequence.rows = [sequence.rows[index] for index in indices]
    loader_expected = {
        (int(row["scene_id"]), int(row["im_id"])) for row in sequence.rows
    }
    loaded_keys = set()
    decoded_mask_pixels = 0
    for frame in sequence:
        key = (int(frame.scene_id), int(frame.im_id))
        loaded_keys.add(key)
        if len(frame.detections) != 1:
            raise ValueError(f"Expected one detection for {key}; got {len(frame.detections)}")
        pixels = int(np.asarray(frame.detections[0].mask, dtype=bool).sum())
        if pixels < minimum_mask_pixels:
            raise ValueError(f"Detection mask for {key} has only {pixels} pixels")
        decoded_mask_pixels += pixels
    if loaded_keys != loader_expected:
        raise ValueError("WebDatasetSequence did not load every selected frame")

    gt_keys = set()
    for key, image, K, _depth, gt, extra in iter_gt_frames(
        dataset_dir, split, load_depth=False
    ):
        scene_id, im_id = (int(value) for value in key.split("_"))
        gt_keys.add((scene_id, im_id))
        if image.ndim != 3 or K.shape != (3, 3):
            raise ValueError(f"Invalid RGB/intrinsics for {key}")
        if len(gt) != 1 or len(extra["infos"]) != 1 or len(extra["masks"]) != 1:
            raise ValueError(f"Incomplete GT for {key}")
        if float(gt[0]["cam_t_m2c"][2]) <= 0:
            raise ValueError(f"Non-positive GT depth for {key}")
    if gt_keys != expected:
        raise ValueError("GT iterator did not load every frame")

    metadata = json.loads(metadata_path.read_text())
    report = {
        "format": "rgb_self_recovery_gru_assetto_validation_v1",
        "dataset_dir": str(dataset_dir),
        "role": metadata.get("role"),
        "frame_count": len(rows),
        "run_count": len({row["source_run"] for row in rows}),
        "sequence_count": len(sequence_report),
        "segment_count": sum(row["segment_count"] for row in sequence_report),
        "target_count": len(targets),
        "detection_count": len(detections),
        "decoded_mask_pixels_total": decoded_mask_pixels,
        "sequence_loader_frames_checked": len(loaded_keys),
        "all_gt_frames_checked": len(gt_keys),
        "sequences": sequence_report,
        "valid": True,
    }
    return report


def main() -> None:
    args = parser().parse_args()
    report = validate(
        args.dataset_dir,
        args.split,
        args.minimum_mask_pixels,
        args.maximum_sequence_loader_frames,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
