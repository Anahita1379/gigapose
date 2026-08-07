#!/usr/bin/env python3
"""Validate a pose-regenerated Assetto distance-bin dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--dataset-root", type=Path, required=True)
    value.add_argument("--source-root", type=Path)
    value.add_argument("--output", type=Path)
    return value


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def key(row: dict[str, str]) -> tuple[str, int, str]:
    frame = row.get("frame", row.get("source_frame"))
    if frame is None:
        raise KeyError("Row has neither frame nor source_frame")
    return row["source_run"], int(frame), row["camera_id"]


def image_path(row: dict[str, str]) -> Path:
    if row.get("image"):
        return Path(row["image"])
    return Path("images") / row["camera_id"] / row["output_filename"]


def mask_path(row: dict[str, str]) -> Path:
    if row.get("masks"):
        return Path(row["masks"])
    return (
        Path("masks") / row["camera_id"]
        / Path(row["output_filename"]).with_suffix(".png").name
    )


def decode(path: Path) -> np.ndarray:
    value = np.asarray(Image.open(path))
    if value.ndim != 3 or value.shape[2] < 3:
        raise ValueError(f"Expected packed RGB/RGBA instance mask: {path}")
    return (
        value[..., 0].astype(np.uint32)
        + (value[..., 1].astype(np.uint32) << 8)
        + (value[..., 2].astype(np.uint32) << 16)
    )


def bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def validate(dataset_root: Path, source_root: Path | None) -> dict[str, object]:
    failures = {
        "empty_masks": 0,
        "wrong_instance_ids": 0,
        "bbox_mismatches": 0,
        "shape_mismatches": 0,
        "missing_images": 0,
        "manifest_mismatches": 0,
        "csv_key_mismatches": 0,
        "unexpected_transform_field_changes": 0,
    }
    allowed_transform_changes = {"xmin", "ymin", "xmax", "ymax", "truncated"}
    total = 0
    per_bin: dict[str, int] = {}
    bin_dirs = sorted(
        path for path in dataset_root.iterdir()
        if path.is_dir() and (path / "source_manifest.csv").is_file()
    )
    for bin_dir in bin_dirs:
        manifests_list = rows(bin_dir / "source_manifest.csv")
        required_csv = (
            bin_dir / "csv" / "transforms.csv",
            bin_dir / "csv" / "bboxes_3d.csv",
        )
        if not manifests_list:
            # Regeneration preserves an empty source manifest for requested
            # bins with no usable frames.  Such a bin is valid only when no
            # partially populated geometry CSV accompanies it.
            if any(path.exists() for path in required_csv):
                failures["csv_key_mismatches"] += 1
            per_bin[bin_dir.name] = 0
            print(f"validated {bin_dir.name}: 0 frames", flush=True)
            continue
        missing_csv = [path for path in required_csv if not path.is_file()]
        if missing_csv:
            failures["csv_key_mismatches"] += 1
            per_bin[bin_dir.name] = 0
            print(
                f"validated {bin_dir.name}: missing geometry CSVs {missing_csv}",
                flush=True,
            )
            continue
        manifests = {key(row): row for row in manifests_list}
        transforms = {
            key(row): row for row in rows(bin_dir / "csv" / "transforms.csv")
        }
        boxes = {key(row): row for row in rows(bin_dir / "csv" / "bboxes_3d.csv")}
        if not (manifests.keys() == transforms.keys() == boxes.keys()):
            failures["csv_key_mismatches"] += 1

        old_transforms = None
        if source_root is not None:
            source_bin = source_root / bin_dir.name
            source_manifests = rows(source_bin / "source_manifest.csv")
            if source_manifests != manifests_list:
                failures["manifest_mismatches"] += 1
            old_transforms = {
                key(row): row
                for row in rows(source_bin / "csv" / "transforms.csv")
            }

        checked = 0
        for frame_key in manifests.keys() & transforms.keys() & boxes.keys():
            manifest = manifests[frame_key]
            transform = transforms[frame_key]
            box = boxes[frame_key]
            rgb = bin_dir / image_path(manifest)
            packed = bin_dir / mask_path(manifest)
            total += 1
            checked += 1
            if not rgb.is_file():
                failures["missing_images"] += 1
            ids = decode(packed)
            expected_shape = (int(box["image_height"]), int(box["image_width"]))
            if ids.shape != expected_shape:
                failures["shape_mismatches"] += 1
            instance_id = int(box["instance_id"])
            foreground = ids == instance_id
            observed_bounds = bounds(foreground)
            if observed_bounds is None:
                failures["empty_masks"] += 1
            if np.any((ids != 0) & (ids != instance_id)):
                failures["wrong_instance_ids"] += 1
            csv_bounds = tuple(
                int(transform[field]) for field in ("xmin", "ymin", "xmax", "ymax")
            )
            box_bounds = tuple(
                int(box[field]) for field in ("xmin", "ymin", "xmax", "ymax")
            )
            if observed_bounds != csv_bounds or observed_bounds != box_bounds:
                failures["bbox_mismatches"] += 1

            if old_transforms is not None and frame_key in old_transforms:
                old = old_transforms[frame_key]
                for field, old_value in old.items():
                    if (
                        field not in allowed_transform_changes
                        and transform[field] != old_value
                    ):
                        failures["unexpected_transform_field_changes"] += 1
        per_bin[bin_dir.name] = checked
        print(f"validated {bin_dir.name}: {checked} frames", flush=True)

    report: dict[str, object] = {
        "format": "assetto_pose_regenerated_validation_v1",
        "dataset_root": str(dataset_root.resolve()),
        "source_root": None if source_root is None else str(source_root.resolve()),
        "frame_count": total,
        "per_bin": per_bin,
        **failures,
        "valid": not any(failures.values()),
    }
    return report


def main() -> None:
    args = parser().parse_args()
    report = validate(args.dataset_root, args.source_root)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
