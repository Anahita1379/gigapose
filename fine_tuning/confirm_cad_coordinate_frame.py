"""Generate numerical and visual checks for the CAD/Assetto Corsa frame mapping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from fine_tuning.ac_geometry import (
    AC_CAMERA_TO_OPENCV,
    DEFAULT_CAD_TO_AC_BODY,
    InstanceRenderer,
    bbox_from_mask,
    bbox_iou,
    cad_to_camera_pose,
    csv_bbox_xywh,
    image_stem,
    intrinsics,
    load_centered_mesh,
    read_grouped_rows,
)


DEFAULT_SOURCE = Path(
    "/mnt/ssd2tb/.local_share_backup/Steam/steamapps/common/assettocorsa/"
    "apps/lua/multi_cam_obs/frames/20260619_clear_2opponent_withMask"
)
DEFAULT_CAD = Path("gigaPose_datasets/datasets/racecar/models/obj_000001.ply")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--cad-path", type=Path, default=DEFAULT_CAD)
    parser.add_argument("--camera", default="front")
    parser.add_argument("--num-frames", type=int, default=6)
    parser.add_argument("--cad-scale", type=float, default=1.0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("fine_tuning/coordinate_check")
    )
    return parser.parse_args()


def overlay(image: Image.Image, masks: list[np.ndarray], boxes: list[list[int]]) -> Image.Image:
    result = image.convert("RGBA")
    colors = [(0, 255, 70, 100), (20, 120, 255, 100), (255, 220, 0, 100)]
    for index, mask in enumerate(masks):
        color = colors[index % len(colors)]
        rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        rgba[mask] = color
        result = Image.alpha_composite(result, Image.fromarray(rgba))
    draw = ImageDraw.Draw(result)
    for index, (x, y, width, height) in enumerate(boxes):
        draw.rectangle(
            (x, y, x + width - 1, y + height - 1),
            outline=colors[index % len(colors)][:3] + (255,),
            width=3,
        )
    return result.convert("RGB")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mesh = load_centered_mesh(args.cad_path, args.cad_scale)
    grouped = read_grouped_rows(args.source_root, [args.camera])
    groups = [item for item in sorted(grouped.items()) if item[0][0] == args.camera]
    groups = groups[: args.num_frames]
    if not groups:
        raise RuntimeError("No annotated frames matched the requested camera.")

    # A 180-degree yaw alternative. Rectangular annotations usually cannot
    # distinguish it, so both overlays are emitted for a human front/rear check.
    reverse_yaw = DEFAULT_CAD_TO_AC_BODY @ np.diag([-1.0, -1.0, 1.0])
    candidates = {
        "recommended": DEFAULT_CAD_TO_AC_BODY,
        "yaw_180_alternative": reverse_yaw,
    }
    renderer = InstanceRenderer(mesh)
    report = {
        "cad_path": str(args.cad_path.resolve()),
        "cad_centered_extents_m": mesh.extents.tolist(),
        "ac_camera_to_opencv_determinant": float(np.linalg.det(AC_CAMERA_TO_OPENCV)),
        "candidates": {},
        "interpretation": (
            "Choose the candidate whose rendered green/blue silhouettes face the same "
            "direction as the cars. CSV rectangles alone cannot prove front versus rear."
        ),
    }
    try:
        for candidate_name, mapping in candidates.items():
            all_ious = []
            ac_sizes = []
            for (_, frame), rows in groups:
                first = rows[0]
                width, height = int(first["image_width"]), int(first["image_height"])
                poses = [cad_to_camera_pose(row, mapping) for row in rows]
                segmentation, _ = renderer.render(
                    poses, intrinsics(first), width=width, height=height
                )
                masks = [segmentation == index for index in range(1, len(rows) + 1)]
                csv_boxes = [csv_bbox_xywh(row) for row in rows]
                rendered_boxes = [bbox_from_mask(mask) for mask in masks]
                all_ious.extend(
                    bbox_iou(rendered, expected)
                    for rendered, expected in zip(rendered_boxes, csv_boxes)
                )
                ac_sizes.extend(
                    [float(row[f"size_{axis}"]) for axis in "xyz"] for row in rows
                )
                image_path = (
                    args.source_root / "images" / args.camera / f"{image_stem(first)}.jpg"
                )
                output_name = f"{candidate_name}_{args.camera}_{frame:06d}.jpg"
                overlay(Image.open(image_path), masks, csv_boxes).save(
                    args.output_dir / output_name, quality=92
                )
            median_ac_size = np.median(np.asarray(ac_sizes), axis=0)
            mapped_cad_size = np.abs(mapping) @ mesh.extents
            report["candidates"][candidate_name] = {
                "cad_to_ac_body": mapping.tolist(),
                "determinant": float(np.linalg.det(mapping)),
                "mean_rendered_bbox_iou": float(np.mean(all_ious)),
                "median_ac_aabb_size_m": median_ac_size.tolist(),
                "mapped_cad_extent_m": mapped_cad_size.tolist(),
                "extent_ratio_cad_over_ac": (mapped_cad_size / median_ac_size).tolist(),
            }
    finally:
        renderer.close()

    report_path = args.output_dir / "coordinate_frame_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {report_path}")
    print(
        "Inspect recommended_*.jpg versus yaw_180_alternative_*.jpg before "
        "preparing training data."
    )


if __name__ == "__main__":
    main()
