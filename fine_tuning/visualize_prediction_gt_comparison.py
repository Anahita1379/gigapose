"""Visualize GT, original, and fine-tuned GigaPose predictions together.

This draws projected CAD 3D bounding boxes on the prepared Assetto images:

  - GT/reference pose: green
  - baseline/original prediction: red
  - fine-tuned prediction: blue

The GT is read from the prepared WebDataset ``gt.json`` files, i.e. the
Assetto-derived camera-frame poses written by ``Assetto_data_prep``.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from fine_tuning.compare_gigapose_predictions import (
    assign_predictions_without_instance_ids,
    global_instance_map,
    infer_prediction_translation_scale,
    load_gt,
    load_prediction_rows,
    read_ply_vertices,
)


GT_COLOR = (0, 255, 80)
BASELINE_COLOR = (255, 60, 60)
FINETUNED_COLOR = (60, 140, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--finetuned-predictions", type=Path, required=True)
    parser.add_argument("--baseline-name", default="original")
    parser.add_argument("--finetuned-name", default="finetuned")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("gigaPose_datasets/datasets/assettocorsa_inference"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument(
        "--sort-by",
        choices=("image", "translation_improvement", "rotation_improvement", "center_improvement"),
        default="image",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fine_tuning/prediction_gt_comparison/visual_overlays"),
    )
    return parser.parse_args()


def bbox_corners_from_vertices(vertices_mm: np.ndarray) -> np.ndarray:
    mins = vertices_mm.min(axis=0)
    maxs = vertices_mm.max(axis=0)
    x0, y0, z0 = mins
    x1, y1, z1 = maxs
    return np.asarray(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=float,
    )


def project(points_obj_mm: np.ndarray, R: np.ndarray, t_mm: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points_cam = (R @ points_obj_mm.T).T + t_mm.reshape(1, 3)
    valid = points_cam[:, 2] > 1e-6
    uvw = (K @ points_cam.T).T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-9)
    return uv, valid


def draw_projected_box(
    draw: ImageDraw.ImageDraw,
    corners_mm: np.ndarray,
    R: np.ndarray,
    t_mm: np.ndarray,
    K: np.ndarray,
    color: tuple[int, int, int],
    label: str,
    label_offset: int,
    width: int = 3,
) -> None:
    uv, valid = project(corners_mm, R, t_mm, K)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        if valid[i] and valid[j]:
            draw.line(
                [tuple(uv[i].round()), tuple(uv[j].round())],
                fill=color,
                width=width,
            )

    center_uv, center_valid = project(np.zeros((1, 3), dtype=float), R, t_mm, K)
    if bool(center_valid[0]):
        x, y = center_uv[0]
        r = 5
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=width)

    draw.rectangle((8, 8 + label_offset, 360, 28 + label_offset), fill=(0, 0, 0))
    draw.text((12, 11 + label_offset), label, fill=color)


def shard_path_for_key(split_dir: Path, key: str) -> Path:
    mapping = json.loads((split_dir / "key_to_shard.json").read_text())
    return split_dir / f"shard-{int(mapping[key]):06d}.tar"


def load_image_and_camera(dataset_dir: Path, split: str, scene_id: int, im_id: int) -> tuple[Image.Image, np.ndarray]:
    split_dir = dataset_dir / split
    key = f"{scene_id:06d}_{im_id:06d}"
    shard_path = shard_path_for_key(split_dir, key)
    with tarfile.open(shard_path) as tar:
        names = set(tar.getnames())
        rgb_name = next(
            (f"{key}.{suffix}" for suffix in ("rgb.jpg", "rgb.png") if f"{key}.{suffix}" in names),
            None,
        )
        if rgb_name is None:
            raise FileNotFoundError(f"No RGB image for {key} in {shard_path}")
        image = Image.open(io.BytesIO(tar.extractfile(rgb_name).read())).convert("RGB")
        camera = json.loads(tar.extractfile(f"{key}.camera.json").read())
    return image, np.asarray(camera["cam_K"], dtype=float).reshape(3, 3)


def predictions_by_gt_key(
    prediction_path: Path,
    gt_by_image: dict[tuple[int, int], list[dict[str, Any]]],
    instance_map: dict[int, tuple[int, int, int]],
) -> dict[tuple[int, int, int], dict[str, Any]]:
    rows = load_prediction_rows(prediction_path)
    t_scale = infer_prediction_translation_scale(rows)
    for row in rows:
        row["t_mm"] = row["t"] * t_scale

    output: dict[tuple[int, int, int], dict[str, Any]] = {}
    if rows and "instance_id" in rows[0] and instance_map:
        top_by_instance: dict[int, dict[str, Any]] = {}
        for row in rows:
            instance_id = row["instance_id"]
            if instance_id not in top_by_instance or row["score"] > top_by_instance[instance_id]["score"]:
                top_by_instance[instance_id] = row
        for instance_id, row in top_by_instance.items():
            if instance_id in instance_map:
                output[instance_map[instance_id]] = row
        return output

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["scene_id"], row["im_id"])].append(row)
    for (scene_id, im_id), gt_items in gt_by_image.items():
        pairs = assign_predictions_without_instance_ids(grouped.get((scene_id, im_id), []), gt_items)
        for pred, gt in pairs:
            output[(scene_id, im_id, gt["gt_index"])] = pred
    return output


def center_error_px(pose: dict[str, Any], gt: dict[str, Any], K: np.ndarray) -> float:
    pred_uv, _ = project(np.zeros((1, 3), dtype=float), pose["R"], pose["t_mm"], K)
    gt_uv, _ = project(np.zeros((1, 3), dtype=float), gt["R"], gt["t_mm"], K)
    return float(np.linalg.norm(pred_uv[0] - gt_uv[0]))


def translation_error_mm(pose: dict[str, Any], gt: dict[str, Any]) -> float:
    return float(np.linalg.norm(pose["t_mm"] - gt["t_mm"]))


def rotation_error_deg(pose: dict[str, Any], gt: dict[str, Any]) -> float:
    delta = pose["R"] @ gt["R"].T
    cos_theta = (np.trace(delta) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gt_by_image, _ = load_gt(args.dataset_dir, args.split)
    instance_map = global_instance_map(args.dataset_dir)
    mesh_path = args.mesh or args.dataset_dir / "models" / "obj_000001.ply"
    vertices_mm = read_ply_vertices(mesh_path, max_points=0)
    if vertices_mm is None:
        raise ValueError(f"Could not read mesh vertices from {mesh_path}")
    corners_mm = bbox_corners_from_vertices(vertices_mm)

    baseline = predictions_by_gt_key(args.baseline_predictions, gt_by_image, instance_map)
    finetuned = predictions_by_gt_key(args.finetuned_predictions, gt_by_image, instance_map)

    candidates = []
    for (scene_id, im_id), gt_items in gt_by_image.items():
        for gt in gt_items:
            key = (scene_id, im_id, gt["gt_index"])
            if key not in baseline or key not in finetuned:
                continue
            image, K = load_image_and_camera(args.dataset_dir, args.split, scene_id, im_id)
            base = baseline[key]
            tuned = finetuned[key]
            item = {
                "key": key,
                "scene_id": scene_id,
                "im_id": im_id,
                "gt": gt,
                "baseline": base,
                "finetuned": tuned,
                "K": K,
                "image": image,
                "translation_improvement": translation_error_mm(base, gt) - translation_error_mm(tuned, gt),
                "rotation_improvement": rotation_error_deg(base, gt) - rotation_error_deg(tuned, gt),
                "center_improvement": center_error_px(base, gt, K) - center_error_px(tuned, gt, K),
            }
            candidates.append(item)

    if args.sort_by != "image":
        candidates.sort(key=lambda item: item[args.sort_by])
    else:
        candidates.sort(key=lambda item: item["key"])
    candidates = candidates[:: max(args.every, 1)]
    if args.max_images is not None:
        candidates = candidates[: args.max_images]

    rows = []
    for item in candidates:
        scene_id, im_id, gt_index = item["key"]
        image = item["image"].copy()
        draw = ImageDraw.Draw(image)
        K = item["K"]
        gt = item["gt"]
        base = item["baseline"]
        tuned = item["finetuned"]

        draw_projected_box(draw, corners_mm, gt["R"], gt["t_mm"], K, GT_COLOR, "GT", 0)
        draw_projected_box(
            draw,
            corners_mm,
            base["R"],
            base["t_mm"],
            K,
            BASELINE_COLOR,
            f"{args.baseline_name}: t={translation_error_mm(base, gt):.0f}mm R={rotation_error_deg(base, gt):.1f}deg",
            24,
        )
        draw_projected_box(
            draw,
            corners_mm,
            tuned["R"],
            tuned["t_mm"],
            K,
            FINETUNED_COLOR,
            f"{args.finetuned_name}: t={translation_error_mm(tuned, gt):.0f}mm R={rotation_error_deg(tuned, gt):.1f}deg",
            48,
        )
        output_path = args.output_dir / f"{scene_id:06d}_{im_id:06d}_gt{gt_index:02d}.jpg"
        image.save(output_path, quality=94)
        rows.append(
            {
                "scene_id": scene_id,
                "im_id": im_id,
                "gt_index": gt_index,
                "translation_improvement_mm": item["translation_improvement"],
                "rotation_improvement_deg": item["rotation_improvement"],
                "center_improvement_px": item["center_improvement"],
                "output": str(output_path),
            }
        )

    with (args.output_dir / "visual_overlay_index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["output"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} GT/original/fine-tuned overlays to {args.output_dir}")


if __name__ == "__main__":
    main()
