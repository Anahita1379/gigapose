#!/usr/bin/env python3
"""Validate Assetto camera geometry per camera against CSV bboxes and masks.

Unlike ``validate_generated_masks.py``, this script does not only compare saved
masks to a fresh render using the same settings.  It also reports whether the
unclipped CAD render lands inside the CSV bbox.  This is useful for catching a
bad camera extrinsic/pose conversion that is self-consistent enough to reproduce
the saved generated masks, but still wrong with respect to the recorded image
annotations.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from Assetto_data_prep.common import (
    bbox_from_mask,
    build_aligned_mesh,
    cad_to_camera_pose,
    collect_sessions,
    csv_bbox_xywh,
    decode_instance_ids,
    image_size,
    intrinsics,
    parse_cameras,
    parse_ids,
    validate_model_row,
)
from fine_tuning.ac_geometry import InstanceRenderer, bbox_iou


DEFAULT_CAD = Path("gigaPose_datasets/datasets/racecar/models/obj_000001.ply")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, action="append", required=True)
    parser.add_argument("--cad-path", type=Path, default=DEFAULT_CAD)
    parser.add_argument("--cameras", default="all")
    parser.add_argument("--opponent-ids", default=None)
    parser.add_argument("--mask-dir-name", default="generated_masks")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-frames-per-session", type=int, default=None)
    parser.add_argument("--exclude-truncated", action="store_true")
    parser.add_argument("--cad-scale", type=float, default=1.0)
    parser.add_argument(
        "--cad-axis-convention",
        choices=("x-forward-z-up", "csp"),
        default="x-forward-z-up",
    )
    parser.add_argument(
        "--fit-aabb",
        choices=("none", "uniform", "nonuniform"),
        default="nonuniform",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("fine_tuning/camera_geometry_validation_report.json"),
    )
    return parser.parse_args()


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return float(intersection / union) if union else float("nan")


def median(values: list[float]) -> float | None:
    values = [value for value in values if np.isfinite(value)]
    return float(np.median(values)) if values else None


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["camera_id"])].append(row)
    summary = []
    for camera_id, items in sorted(grouped.items()):
        summary.append(
            {
                "camera_id": camera_id,
                "instances": len(items),
                "rendered_vs_csv_bbox_iou_median": median(
                    [float(row["rendered_vs_csv_bbox_iou"]) for row in items]
                ),
                "rendered_vs_saved_mask_iou_median": median(
                    [float(row["rendered_vs_saved_mask_iou"]) for row in items]
                ),
                "rendered_pixels_median": median(
                    [float(row["rendered_pixels"]) for row in items]
                ),
                "saved_pixels_median": median(
                    [float(row["saved_pixels"]) for row in items]
                ),
                "gt_z_mm_median": median([float(row["gt_z_mm"]) for row in items]),
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    sessions = collect_sessions(
        args.source_root,
        parse_cameras(args.cameras),
        parse_ids(args.opponent_ids),
        args.exclude_truncated,
        args.frame_stride,
        args.max_frames_per_session,
    )

    rows: list[dict[str, object]] = []
    for frames in sessions:
        alignment = build_aligned_mesh(
            args.cad_path,
            frames[0].rows[0],
            args.cad_scale,
            args.cad_axis_convention,
            args.fit_aabb,
        )
        renderer = InstanceRenderer(alignment.mesh)
        try:
            for frame_index, frame in enumerate(frames, start=1):
                size = image_size(frame)
                with Image.open(frame.image_path) as image:
                    if image.size != size:
                        raise ValueError(f"Image size mismatch for {frame.image_path}")
                instance_ids = decode_instance_ids(frame.mask_path(args.mask_dir_name), size)
                for row in frame.rows:
                    validate_model_row(alignment, row)
                poses = [
                    cad_to_camera_pose(row, alignment.local_center)
                    for row in frame.rows
                ]
                segmentation, depth_m = renderer.render(
                    poses, intrinsics(frame), size[0], size[1]
                )
                for render_id, (csv_row, pose) in enumerate(
                    zip(frame.rows, poses), start=1
                ):
                    opponent_id = int(csv_row["opp_id"])
                    rendered = np.logical_and(segmentation == render_id, depth_m > 0)
                    saved = instance_ids == opponent_id
                    csv_bbox = csv_bbox_xywh(csv_row)
                    rendered_bbox = bbox_from_mask(rendered)
                    saved_bbox = bbox_from_mask(saved)
                    rows.append(
                        {
                            "source_root": str(frame.source_root),
                            "camera_id": frame.camera_id,
                            "frame": frame.frame_id,
                            "stem": frame.stem,
                            "opp_id": opponent_id,
                            "gt_z_mm": float(pose[2, 3] * 1000.0),
                            "csv_bbox": csv_bbox,
                            "rendered_bbox_unclipped": rendered_bbox,
                            "saved_bbox": saved_bbox,
                            "rendered_pixels": int(rendered.sum()),
                            "saved_pixels": int(saved.sum()),
                            "rendered_vs_csv_bbox_iou": bbox_iou(
                                rendered_bbox, csv_bbox
                            ),
                            "rendered_vs_saved_mask_iou": mask_iou(rendered, saved),
                        }
                    )
                if frame_index % 250 == 0 or frame_index == len(frames):
                    print(
                        f"{frame.source_root.name}: checked {frame_index}/{len(frames)} frames"
                    )
        finally:
            renderer.close()

    output = {
        "summary_by_camera": summarize(rows),
        "details": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2))
    print(json.dumps(output["summary_by_camera"], indent=2))
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
