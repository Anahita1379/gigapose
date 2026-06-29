"""Visualize multiple model predictions as one side-by-side panel per car.

For each selected image, this writes a single output image.  If the source frame
contains N GT cars, the output has N panels next to each other.  Each panel is a
copy of the full image with:

  - GT/reference box for that car
  - one predicted box per model, paired to the same GT car

This is useful for multi-car images where separate files make it hard to see
which prediction belongs to which car.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from fine_tuning.compare_gigapose_predictions import load_gt, read_ply_vertices
from fine_tuning.visualize_prediction_gt_comparison import (
    bbox_corners_from_vertices,
    center_error_px,
    draw_projected_box,
    load_image_and_camera,
    predictions_by_gt_key,
    rotation_error_deg,
    translation_error_mm,
)


COLORS = [
    (255, 60, 60),
    (60, 140, 255),
    (255, 210, 20),
    (210, 40, 255),
    (20, 220, 220),
    (255, 140, 20),
]
GT_COLOR = (0, 255, 80)


def parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--model must be name=/path/to/predictions.csv")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Model name cannot be empty")
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model,
        required=True,
        help="Repeat as name=/path/to/predictions.csv",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("gigaPose_datasets/datasets/assettocorsa_benchmark"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument(
        "--sort-by",
        choices=("image", "worst_center", "best_center"),
        default="image",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fine_tuning/metrics/multi_model_per_car_visuals"),
    )
    return parser.parse_args()


def draw_panel_header(
    draw: ImageDraw.ImageDraw,
    gt_index: int,
    model_count: int,
) -> None:
    height = 28 + model_count * 24
    draw.rectangle((6, 6, 520, height), fill=(0, 0, 0))
    draw.text((12, 10), f"GT car index {gt_index}", fill=GT_COLOR)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gt_by_image, K_by_image = load_gt(args.dataset_dir, args.split)
    mesh_path = args.mesh or args.dataset_dir / "models" / "obj_000001.ply"
    vertices_mm = read_ply_vertices(mesh_path, max_points=0)
    if vertices_mm is None:
        raise ValueError(f"Could not read mesh vertices from {mesh_path}")
    corners_mm = bbox_corners_from_vertices(vertices_mm)

    model_predictions = {
        name: predictions_by_gt_key(path, gt_by_image, K_by_image)
        for name, path in args.model
    }

    candidates: list[dict[str, Any]] = []
    for (scene_id, im_id), gt_items in gt_by_image.items():
        if not gt_items:
            continue
        image, K = load_image_and_camera(args.dataset_dir, args.split, scene_id, im_id)
        worst_center = 0.0
        complete_gt_items = []
        for gt in gt_items:
            key = (scene_id, im_id, gt["gt_index"])
            preds = {
                name: pred_map[key]
                for name, pred_map in model_predictions.items()
                if key in pred_map
            }
            if not preds:
                continue
            center_errors = [center_error_px(pred, gt, K) for pred in preds.values()]
            worst_center = max(worst_center, max(center_errors))
            complete_gt_items.append((gt, preds))
        if not complete_gt_items:
            continue
        candidates.append(
            {
                "scene_id": scene_id,
                "im_id": im_id,
                "image": image,
                "K": K,
                "gt_items": complete_gt_items,
                "worst_center": worst_center,
            }
        )

    if args.sort_by == "worst_center":
        candidates.sort(key=lambda item: -item["worst_center"])
    elif args.sort_by == "best_center":
        candidates.sort(key=lambda item: item["worst_center"])
    else:
        candidates.sort(key=lambda item: (item["scene_id"], item["im_id"]))

    candidates = candidates[:: max(args.every, 1)]
    if args.max_images is not None:
        candidates = candidates[: args.max_images]

    rows = []
    for item in candidates:
        image: Image.Image = item["image"]
        K = item["K"]
        panels = []
        for gt, preds in item["gt_items"]:
            panel = image.copy()
            draw = ImageDraw.Draw(panel)
            draw_panel_header(draw, gt["gt_index"], len(preds))
            draw_projected_box(
                draw,
                corners_mm,
                gt["R"],
                gt["t_mm"],
                K,
                GT_COLOR,
                "GT",
                24,
            )
            for model_idx, (name, pred) in enumerate(sorted(preds.items())):
                color = COLORS[model_idx % len(COLORS)]
                label = (
                    f"{name}: t={translation_error_mm(pred, gt):.0f}mm "
                    f"R={rotation_error_deg(pred, gt):.1f}deg "
                    f"c={center_error_px(pred, gt, K):.0f}px"
                )
                draw_projected_box(
                    draw,
                    corners_mm,
                    pred["R"],
                    pred["t_mm"],
                    K,
                    color,
                    label,
                    48 + model_idx * 24,
                )
            panels.append(panel)

        output = Image.new("RGB", (image.width * len(panels), image.height))
        for idx, panel in enumerate(panels):
            output.paste(panel, (idx * image.width, 0))

        scene_id = item["scene_id"]
        im_id = item["im_id"]
        output_path = args.output_dir / f"{scene_id:06d}_{im_id:06d}_cars_side_by_side.jpg"
        output.save(output_path, quality=94)
        rows.append(
            {
                "scene_id": scene_id,
                "im_id": im_id,
                "car_panels": len(panels),
                "worst_center_error_px": item["worst_center"],
                "output": str(output_path),
            }
        )

    with (args.output_dir / "multi_model_per_car_index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()) if rows else ["output"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} side-by-side frame visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
