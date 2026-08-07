"""Validate a prepared real-world Grounded-SAM inference dataset without GT."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from tracking.io import WebDatasetSequence


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--dataset-dir", type=Path, required=True)
    value.add_argument("--split", default="test")
    value.add_argument("--minimum-mask-pixels", type=int, default=25)
    value.add_argument(
        "--maximum-decoded-frames",
        type=int,
        default=512,
        help="Evenly sample this many frames for image/mask/shard decoding; use 0 for all.",
    )
    value.add_argument(
        "--require-templates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    value.add_argument("--output", type=Path)
    return value


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _unique_keys(rows: list[dict[str, Any]], image_field: str) -> set[tuple[int, int]]:
    keys = [(int(row["scene_id"]), int(row[image_field])) for row in rows]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate frame keys in {image_field} rows: {duplicates[:10]}")
    return set(keys)


def _resolve_detection_path(dataset_dir: Path, metadata: dict[str, Any]) -> Path:
    configured = Path(str(metadata.get("detection_file", "")))
    candidates = []
    if str(configured):
        candidates.extend((configured, Path.cwd() / configured))
    candidates.append(
        dataset_dir.parent
        / "cnos-fastsam"
        / f"cnos-fastsam_{dataset_dir.name}-test.json"
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "Detection JSON was not found; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _sample_rows(rows: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    if maximum < 0:
        raise ValueError("--maximum-decoded-frames cannot be negative")
    if maximum == 0 or len(rows) <= maximum:
        return rows
    indices = np.linspace(0, len(rows) - 1, maximum, dtype=int)
    return [rows[int(index)] for index in np.unique(indices)]


def validate(args: argparse.Namespace) -> dict[str, Any]:
    if args.minimum_mask_pixels < 0:
        raise ValueError("--minimum-mask-pixels cannot be negative")

    dataset_dir = args.dataset_dir.resolve()
    split_dir = dataset_dir / args.split
    frame_map_path = dataset_dir / "frame_map.json"
    targets_path = dataset_dir / "test_targets_bop19.json"
    inference_metadata_path = dataset_dir / "inference_metadata.json"
    key_to_shard_path = split_dir / "key_to_shard.json"

    frame_rows = _load_json(frame_map_path)
    targets = _load_json(targets_path)
    metadata = _load_json(inference_metadata_path)
    key_to_shard = _load_json(key_to_shard_path)
    if not isinstance(frame_rows, list) or not frame_rows:
        raise ValueError("frame_map.json must contain a non-empty list")
    if not isinstance(targets, list) or not targets:
        raise ValueError("test_targets_bop19.json must contain a non-empty list")
    if not isinstance(metadata, dict):
        raise ValueError("inference_metadata.json must contain an object")
    if not isinstance(key_to_shard, dict):
        raise ValueError("key_to_shard.json must contain an object")

    detection_path = _resolve_detection_path(dataset_dir, metadata)
    detections = _load_json(detection_path)
    if not isinstance(detections, list) or not detections:
        raise ValueError("Detection JSON must contain a non-empty list")

    frame_keys = _unique_keys(frame_rows, "im_id")
    target_keys = _unique_keys(targets, "im_id")
    detection_keys = _unique_keys(detections, "image_id")
    shard_keys = {
        tuple(int(value) for value in str(key).split("_"))
        for key in key_to_shard
    }
    if frame_keys != target_keys:
        raise ValueError("frame_map and test targets contain different frame keys")
    if frame_keys != detection_keys:
        raise ValueError(
            "This one-car runtime pipeline requires exactly one detection row per frame; "
            "frame_map and detection keys differ or detections are duplicated"
        )
    if frame_keys != shard_keys:
        raise ValueError("frame_map and key_to_shard contain different frame keys")

    missing_shards = []
    for value in set(key_to_shard.values()):
        path = (
            split_dir / value
            if isinstance(value, str) and value.endswith(".tar")
            else split_dir / f"shard-{int(value):06d}.tar"
        )
        if not path.is_file():
            missing_shards.append(str(path))
    if missing_shards:
        raise FileNotFoundError(f"Missing WebDataset shards: {missing_shards[:10]}")

    missing_sources: dict[str, list[str]] = {
        "image_path": [], "mask_path": [], "sample_metadata_path": []
    }
    invalid_instance_rows = []
    previous_by_sequence: dict[tuple[str, str], int] = {}
    for row in frame_rows:
        for field in missing_sources:
            value = row.get(field)
            if not value or not Path(str(value)).is_file():
                missing_sources[field].append(str(value))
        if len(row.get("instances", [])) != 1:
            invalid_instance_rows.append((int(row["scene_id"]), int(row["im_id"])))
        sequence = (str(row.get("source_root", "")), str(row.get("camera_id", "")))
        source_frame = int(row.get("frame_index", row["im_id"]))
        previous = previous_by_sequence.get(sequence)
        if previous is not None and source_frame <= previous:
            raise ValueError(f"Nonmonotonic source frames in sequence {sequence}")
        previous_by_sequence[sequence] = source_frame
    for field, missing in missing_sources.items():
        if missing:
            raise FileNotFoundError(f"Missing {field} entries: {missing[:10]}")
    if invalid_instance_rows:
        raise ValueError(
            "Translation-cascade runtime export requires exactly one car instance per "
            f"frame; invalid keys: {invalid_instance_rows[:10]}"
        )

    template_dir = dataset_dir.parent / "templates" / dataset_dir.name
    template_png_count = 0
    if args.require_templates:
        object_dir = template_dir / "000001"
        pose_path = template_dir / "object_poses" / "000001.npy"
        if not object_dir.is_dir() or not pose_path.is_file():
            raise FileNotFoundError(
                f"Incomplete template bank: expected {object_dir} and {pose_path}"
            )
        template_png_count = len(list(object_dir.glob("*.png")))
        if template_png_count != 324:
            raise ValueError(
                f"Expected 324 template PNGs for object 1; found {template_png_count}"
            )

    sequence = WebDatasetSequence(dataset_dir, args.split, load_depth=False)
    sequence.rows = _sample_rows(sequence.rows, args.maximum_decoded_frames)
    decoded_keys = set()
    decoded_mask_pixels = 0
    for frame in sequence:
        key = (int(frame.scene_id), int(frame.im_id))
        decoded_keys.add(key)
        if frame.image.ndim != 3 or frame.image.shape[2] != 3:
            raise ValueError(f"Invalid RGB image for {key}: {frame.image.shape}")
        if frame.K.shape != (3, 3) or not np.isfinite(frame.K).all():
            raise ValueError(f"Invalid camera intrinsics for {key}")
        if len(frame.detections) != 1:
            raise ValueError(f"Expected one decoded detection for {key}; got {len(frame.detections)}")
        pixels = int(np.asarray(frame.detections[0].mask, dtype=bool).sum())
        if pixels < args.minimum_mask_pixels:
            raise ValueError(
                f"Decoded detection mask for {key} has {pixels} pixels; "
                f"minimum is {args.minimum_mask_pixels}"
            )
        decoded_mask_pixels += pixels

    expected_decoded = {
        (int(row["scene_id"]), int(row["im_id"])) for row in sequence.rows
    }
    if decoded_keys != expected_decoded:
        raise ValueError("WebDatasetSequence did not decode every selected frame")

    prepared_frames = int(metadata.get("prepared_frames", len(frame_rows)))
    if prepared_frames != len(frame_rows):
        raise ValueError(
            f"inference_metadata prepared_frames={prepared_frames}, frame_map={len(frame_rows)}"
        )

    return {
        "format": "grounded_sam_real_inference_validation_v1",
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "ground_truth_required": False,
        "frame_count": len(frame_rows),
        "target_count": len(targets),
        "detection_count": len(detections),
        "shard_count": len(set(key_to_shard.values())),
        "sequence_count": len(previous_by_sequence),
        "decoded_frame_count": len(decoded_keys),
        "decoded_mask_pixels_total": decoded_mask_pixels,
        "minimum_mask_pixels": args.minimum_mask_pixels,
        "template_dir": str(template_dir),
        "template_png_count": template_png_count if args.require_templates else None,
        "detection_file": str(detection_path),
        "valid": True,
    }


def main() -> None:
    args = parser().parse_args()
    report = validate(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
