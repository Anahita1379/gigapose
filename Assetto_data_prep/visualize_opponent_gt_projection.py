#!/usr/bin/env python3
"""Visualize GT opponent poses from the new Assetto transforms.csv schema.

This script does not use ``csv/bboxes_3d.csv`` and does not require original
instance masks.  It reads:

- ``csv/camera_frames.csv`` for per-camera intrinsics and image size;
- ``csv/transforms.csv`` for visible opponent rows, 2D bbox, AABB, and
  ``T_camera_opponent_visual``;
- ``images/<camera>/<sim_time>_<frame>.jpg`` for the RGB image.

It renders the aligned CAD model at the GT camera pose and draws:

- translucent rendered CAD mask;
- rendered CAD bbox;
- recorder/CSV bbox.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from Assetto_data_prep.common import (
    bbox_from_mask,
    bbox_iou,
    build_aligned_mesh,
    cad_to_camera_pose,
    collect_sessions,
    csv_bbox_xywh,
    image_size,
    intrinsics,
    parse_cameras,
    parse_ids,
    validate_model_row,
)
from fine_tuning.ac_geometry import InstanceRenderer


DEFAULT_CAD = Path("gigaPose_datasets/datasets/racecar/models/obj_000001.ply")
MASK_COLORS = [
    (0, 255, 80, 85),
    (60, 150, 255, 85),
    (255, 210, 0, 85),
    (255, 80, 220, 85),
    (80, 255, 240, 85),
]
RENDERED_BBOX_COLOR = (0, 255, 255)
CSV_BBOX_COLOR = (255, 220, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, action="append", required=True)
    parser.add_argument("--cad-path", type=Path, default=DEFAULT_CAD)
    parser.add_argument("--cameras", default="rear")
    parser.add_argument("--opponent-ids", default=None)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-frames-per-session", type=int, default=50)
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
        "--output-dir",
        type=Path,
        default=Path("fine_tuning/opponent_gt_projection_overlays"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional report path. Defaults to <output-dir>/projection_report.json.",
    )
    return parser.parse_args()


def draw_box(
    draw: ImageDraw.ImageDraw,
    bbox: list[int],
    color: tuple[int, int, int],
    label: str,
    label_y_offset: int = 0,
) -> None:
    x, y, w, h = [int(v) for v in bbox]
    if w <= 0 or h <= 0:
        return
    draw.rectangle((x, y, x + w - 1, y + h - 1), outline=color, width=3)
    draw.rectangle((x, max(0, y - 16 - label_y_offset), x + 260, max(16, y - label_y_offset)), fill=(0, 0, 0))
    draw.text((x + 3, max(0, y - 14 - label_y_offset)), label, fill=color)


def overlay_masks(
    image: Image.Image,
    masks: list[np.ndarray],
    rendered_boxes: list[list[int]],
    csv_boxes: list[list[int]],
    opp_ids: list[int],
    ious: list[float],
) -> Image.Image:
    result = image.convert("RGBA")
    for index, mask in enumerate(masks):
        color = MASK_COLORS[index % len(MASK_COLORS)]
        rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        rgba[mask] = color
        result = Image.alpha_composite(result, Image.fromarray(rgba))

    out = result.convert("RGB")
    draw = ImageDraw.Draw(out)
    draw.rectangle((6, 6, 610, 50), fill=(0, 0, 0))
    draw.text((12, 10), "cyan: rendered CAD bbox    yellow: CSV/recorder bbox", fill=(255, 255, 255))
    draw.text((12, 28), "translucent color: rendered CAD mask from GT pose", fill=(255, 255, 255))

    for index, (rendered, csv_box, opp_id, iou) in enumerate(
        zip(rendered_boxes, csv_boxes, opp_ids, ious)
    ):
        draw_box(
            draw,
            csv_box,
            CSV_BBOX_COLOR,
            f"CSV opp={opp_id}",
            label_y_offset=18,
        )
        draw_box(
            draw,
            rendered,
            RENDERED_BBOX_COLOR,
            f"rendered opp={opp_id} IoU={iou:.2f}",
            label_y_offset=0,
        )
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_json or (args.output_dir / "projection_report.json")

    sessions = collect_sessions(
        args.source_root,
        parse_cameras(args.cameras),
        parse_ids(args.opponent_ids),
        args.exclude_truncated,
        args.frame_stride,
        args.max_frames_per_session,
    )

    rows_out: list[dict[str, object]] = []
    summary: dict[str, list[float]] = {}

    for session_frames in sessions:
        alignment = build_aligned_mesh(
            args.cad_path,
            session_frames[0].rows[0],
            args.cad_scale,
            args.cad_axis_convention,
            args.fit_aabb,
        )
        renderer = InstanceRenderer(alignment.mesh)
        try:
            for frame in session_frames:
                width, height = image_size(frame)
                for row in frame.rows:
                    validate_model_row(alignment, row)
                poses = [cad_to_camera_pose(row, alignment.local_center) for row in frame.rows]
                segmentation, depth_m = renderer.render(
                    poses, intrinsics(frame), width=width, height=height
                )

                masks = []
                rendered_boxes = []
                csv_boxes = []
                opp_ids = []
                ious = []
                for render_id, row in enumerate(frame.rows, start=1):
                    rendered = np.logical_and(segmentation == render_id, depth_m > 0)
                    rendered_box = bbox_from_mask(rendered)
                    csv_box = csv_bbox_xywh(row)
                    iou = bbox_iou(rendered_box, csv_box)
                    masks.append(rendered)
                    rendered_boxes.append(rendered_box)
                    csv_boxes.append(csv_box)
                    opp_ids.append(int(row["opp_id"]))
                    ious.append(iou)
                    summary.setdefault(frame.camera_id, []).append(iou)
                    rows_out.append(
                        {
                            "source_root": str(frame.source_root),
                            "camera_id": frame.camera_id,
                            "frame": frame.frame_id,
                            "stem": frame.stem,
                            "opp_id": int(row["opp_id"]),
                            "visible": row.get("visible", ""),
                            "truncated": row.get("truncated", ""),
                            "rendered_bbox": rendered_box,
                            "csv_bbox": csv_box,
                            "rendered_vs_csv_bbox_iou": iou,
                            "center_camera_m": [
                                float(row["center_camera_x"]),
                                float(row["center_camera_y"]),
                                float(row["center_camera_z"]),
                            ],
                        }
                    )

                image = Image.open(frame.image_path).convert("RGB")
                output_image = overlay_masks(
                    image, masks, rendered_boxes, csv_boxes, opp_ids, ious
                )
                output_name = (
                    f"{frame.source_root.name}_{frame.camera_id}_"
                    f"{frame.frame_id:06d}_{frame.stem}.jpg"
                )
                output_image.save(args.output_dir / output_name, quality=94)
        finally:
            renderer.close()

    summary_out = {
        camera: {
            "instances": len(values),
            "rendered_vs_csv_bbox_iou_mean": float(np.mean(values)) if values else None,
            "rendered_vs_csv_bbox_iou_median": float(np.median(values)) if values else None,
        }
        for camera, values in sorted(summary.items())
    }
    report = {
        "source_roots": [str(path) for path in args.source_root],
        "cad_path": str(args.cad_path),
        "cameras": args.cameras,
        "frame_stride": args.frame_stride,
        "max_frames_per_session": args.max_frames_per_session,
        "cad_axis_convention": args.cad_axis_convention,
        "fit_aabb": args.fit_aabb,
        "summary_by_camera": summary_out,
        "details": rows_out,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(summary_out, indent=2))
    print(f"Wrote overlays to {args.output_dir}")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
