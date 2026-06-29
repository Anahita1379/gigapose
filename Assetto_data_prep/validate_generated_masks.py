#!/usr/bin/env python3
"""Validate generated Assetto instance-ID masks against freshly rendered CAD masks.

This is a source-data diagnostic, not a GigaPose evaluation script.  It checks
whether the PNGs in ``generated_masks/<camera>/<stem>.png`` are consistent with
the current CAD alignment, camera intrinsics, and Assetto pose CSVs.

For each selected frame/opponent it reports IoU between:

  - saved mask pixels for the opponent ID
  - freshly rendered CAD silhouette for that opponent

Low IoU usually means the generated masks are stale or were produced with
different CAD/axis/AABB options than the current preparation command.
"""

from __future__ import annotations

import argparse
import json
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
from fine_tuning.ac_geometry import InstanceRenderer


DEFAULT_CAD = Path("gigaPose_datasets/datasets/racecar/models/obj_000001.ply")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check generated Assetto masks against current CAD rendering."
    )
    parser.add_argument("--source-root", type=Path, action="append", required=True)
    parser.add_argument("--cad-path", type=Path, default=DEFAULT_CAD)
    parser.add_argument("--cameras", default="all")
    parser.add_argument("--opponent-ids", default=None)
    parser.add_argument("--mask-dir-name", default="generated_masks")
    parser.add_argument("--frame-stride", type=int, default=1)
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
        "--clip-to-csv-bbox",
        action="store_true",
        default=True,
        help="Match generate_masks.py default behavior.",
    )
    parser.add_argument(
        "--no-clip-to-csv-bbox",
        dest="clip_to_csv_bbox",
        action="store_false",
    )
    parser.add_argument("--min-iou-warning", type=float, default=0.9)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("fine_tuning/generated_mask_validation_report.json"),
    )
    return parser.parse_args()


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return float(intersection / union) if union else float("nan")


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

    report = []
    all_ious = []
    low_iou = 0
    missing_or_empty = 0
    for frames in sessions:
        reference_row = frames[0].rows[0]
        alignment = build_aligned_mesh(
            args.cad_path,
            reference_row,
            args.cad_scale,
            args.cad_axis_convention,
            args.fit_aabb,
        )
        renderer = InstanceRenderer(alignment.mesh)
        try:
            for progress, frame in enumerate(frames, start=1):
                size = image_size(frame)
                if not frame.image_path.is_file():
                    raise FileNotFoundError(f"Missing image: {frame.image_path}")
                with Image.open(frame.image_path) as image:
                    if image.size != size:
                        raise ValueError(f"Image size mismatch for {frame.image_path}")
                instance_ids = decode_instance_ids(frame.mask_path(args.mask_dir_name), size)
                for row in frame.rows:
                    validate_model_row(alignment, row)
                poses = [
                    cad_to_camera_pose(row, alignment.local_center) for row in frame.rows
                ]
                segmentation, depth_m = renderer.render(
                    poses, intrinsics(frame), size[0], size[1]
                )
                for render_id, row in enumerate(frame.rows, start=1):
                    opponent_id = int(row["opp_id"])
                    saved = instance_ids == opponent_id
                    rendered = np.logical_and(segmentation == render_id, depth_m > 0)
                    if args.clip_to_csv_bbox:
                        x, y, width, height = csv_bbox_xywh(row)
                        clipped = np.zeros_like(rendered)
                        clipped[y : y + height, x : x + width] = rendered[
                            y : y + height, x : x + width
                        ]
                        rendered = clipped
                    iou = mask_iou(saved, rendered)
                    saved_pixels = int(saved.sum())
                    rendered_pixels = int(rendered.sum())
                    if saved_pixels == 0 or rendered_pixels == 0:
                        missing_or_empty += 1
                    if np.isfinite(iou):
                        all_ious.append(iou)
                    if not np.isfinite(iou) or iou < args.min_iou_warning:
                        low_iou += 1
                    report.append(
                        {
                            "source_root": str(frame.source_root),
                            "camera_id": frame.camera_id,
                            "frame": frame.frame_id,
                            "stem": frame.stem,
                            "opp_id": opponent_id,
                            "iou": iou,
                            "saved_pixels": saved_pixels,
                            "rendered_pixels": rendered_pixels,
                            "saved_bbox": bbox_from_mask(saved),
                            "rendered_bbox": bbox_from_mask(rendered),
                        }
                    )
                if progress % 250 == 0 or progress == len(frames):
                    print(
                        f"{frame.source_root.name}: checked {progress}/{len(frames)} frames"
                    )
        finally:
            renderer.close()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "checked_instances": len(report),
        "mean_iou": float(np.mean(all_ious)) if all_ious else None,
        "median_iou": float(np.median(all_ious)) if all_ious else None,
        "p05_iou": float(np.percentile(all_ious, 5)) if all_ious else None,
        "low_iou_count": low_iou,
        "missing_or_empty_count": missing_or_empty,
        "min_iou_warning": args.min_iou_warning,
        "details": report,
    }
    args.output_json.write_text(json.dumps(summary, indent=2))
    print(
        f"Checked {len(report)} instances; "
        f"median IoU={summary['median_iou']}; low IoU={low_iou}; "
        f"empty={missing_or_empty}"
    )
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
