"""Overlay all GigaPose CAD pose predictions on their WebDataset RGB images."""

from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from fine_tuning.ac_geometry import InstanceRenderer, bbox_from_mask


COLORS = [
    (0, 255, 80),
    (20, 140, 255),
    (255, 90, 30),
    (210, 40, 255),
    (255, 210, 20),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("gigaPose_datasets/datasets/assettocorsa"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=1.0,
        help=(
            "Multiply CSV translations by this before rendering. Use 0.001 when "
            "the CSV stores millimeters and the mesh uses meters."
        ),
    )
    parser.add_argument(
        "--center-mesh",
        action="store_true",
        help="Center the mesh bounds; enable only if inference used centered templates.",
    )
    parser.add_argument("--alpha", type=float, default=0.42)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("fine_tuning/prediction_overlays")
    )
    return parser.parse_args()


def load_prediction_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(newline="") as prediction_file:
        for row_index, row in enumerate(csv.DictReader(prediction_file)):
            prediction = {
                "scene_id": int(row["scene_id"]),
                "im_id": int(row["im_id"]),
                "obj_id": int(row["obj_id"]),
                "score": float(row["score"]),
                "R": np.fromstring(row["R"], sep=" ").reshape(3, 3),
                "t": np.fromstring(row["t"], sep=" ").reshape(3),
                "row_index": row_index,
            }
            if "instance_id" in row and row["instance_id"] != "":
                prediction["instance_id"] = int(row["instance_id"])
            rows.append(prediction)
    return rows


def select_multi_hypothesis_top1(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Keep one hypothesis per explicit instance in MultiHypothesis CSV files."""
    if not rows or "instance_id" not in rows[0]:
        return rows
    grouped = defaultdict(list)
    for row in rows:
        key = (row["scene_id"], row["im_id"], row["obj_id"], row["instance_id"])
        grouped[key].append(row)
    return [max(hypotheses, key=lambda item: item["score"]) for hypotheses in grouped.values()]


def shard_path_for_key(split_dir: Path, key: str) -> Path:
    mapping_path = split_dir / "key_to_shard.json"
    if not mapping_path.is_file() and split_dir.name == "train_pbr_web":
        mapping_path = split_dir.parent / "key_to_shard.json"
    mapping = json.loads(mapping_path.read_text())
    if key not in mapping:
        raise KeyError(f"Image key {key} is absent from {mapping_path}")
    shard_id = int(mapping[key])
    return split_dir / f"shard-{shard_id:06d}.tar"


def load_image_and_camera(
    split_dir: Path, scene_id: int, im_id: int
) -> tuple[Image.Image, np.ndarray]:
    key = f"{scene_id:06d}_{im_id:06d}"
    shard_path = shard_path_for_key(split_dir, key)
    with tarfile.open(shard_path) as tar:
        names = set(tar.getnames())
        rgb_name = next(
            (f"{key}.{suffix}" for suffix in ("rgb.jpg", "rgb.png") if f"{key}.{suffix}" in names),
            None,
        )
        if rgb_name is None:
            raise FileNotFoundError(f"No RGB member for {key} in {shard_path}")
        rgb_file = tar.extractfile(rgb_name)
        camera_file = tar.extractfile(f"{key}.camera.json")
        if rgb_file is None or camera_file is None:
            raise FileNotFoundError(f"Incomplete sample {key} in {shard_path}")
        image = Image.open(io.BytesIO(rgb_file.read())).convert("RGB")
        camera = json.loads(camera_file.read())
    return image, np.asarray(camera["cam_K"], dtype=float).reshape(3, 3)


def blend_predictions(
    image: Image.Image,
    segmentation: np.ndarray,
    predictions: list[dict[str, object]],
    alpha: float,
) -> Image.Image:
    base = np.asarray(image, dtype=np.float32).copy()
    for render_id, _ in enumerate(predictions, start=1):
        mask = segmentation == render_id
        color = np.asarray(COLORS[(render_id - 1) % len(COLORS)], dtype=np.float32)
        base[mask] = (1.0 - alpha) * base[mask] + alpha * color
    result = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(result)
    for render_id, prediction in enumerate(predictions, start=1):
        bbox = bbox_from_mask(segmentation == render_id)
        if bbox[2] == 0 or bbox[3] == 0:
            continue
        x, y, width, height = bbox
        color = COLORS[(render_id - 1) % len(COLORS)]
        draw.rectangle((x, y, x + width - 1, y + height - 1), outline=color, width=3)
        label = f"pred {render_id}: score={prediction['score']:.4f}"
        draw.rectangle((x, max(0, y - 18), x + 170, y), fill=(0, 0, 0))
        draw.text((x + 3, max(0, y - 16)), label, fill=color)
    return result


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between zero and one.")
    split_dir = args.dataset_dir / args.split
    mesh_path = args.mesh or args.dataset_dir / "models" / "obj_000001.ply"
    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError(f"Could not load mesh: {mesh_path}")
    mesh = mesh.copy()
    if args.center_mesh:
        mesh.apply_translation(-mesh.bounds.mean(axis=0))

    predictions = select_multi_hypothesis_top1(load_prediction_rows(args.predictions))
    if args.min_score is not None:
        predictions = [row for row in predictions if row["score"] >= args.min_score]
    grouped = defaultdict(list)
    for prediction in predictions:
        grouped[(prediction["scene_id"], prediction["im_id"])].append(prediction)
    image_keys = sorted(grouped)
    if args.max_images is not None:
        image_keys = image_keys[: args.max_images]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    renderer = InstanceRenderer(mesh)
    report = []
    try:
        for scene_id, im_id in image_keys:
            rows = sorted(grouped[(scene_id, im_id)], key=lambda item: -item["score"])
            image, K = load_image_and_camera(split_dir, scene_id, im_id)
            poses = []
            for row in rows:
                pose = np.eye(4, dtype=float)
                pose[:3, :3] = row["R"]
                pose[:3, 3] = row["t"] * args.translation_scale
                poses.append(pose)
            segmentation, _ = renderer.render(poses, K, image.width, image.height)
            overlay = blend_predictions(image, segmentation, rows, args.alpha)
            side_by_side = Image.new("RGB", (image.width * 2, image.height))
            side_by_side.paste(image, (0, 0))
            side_by_side.paste(overlay, (image.width, 0))
            output_path = args.output_dir / f"{scene_id:06d}_{im_id:06d}_overlay.jpg"
            side_by_side.save(output_path, quality=94)
            report.append(
                {
                    "scene_id": scene_id,
                    "im_id": im_id,
                    "prediction_count": len(rows),
                    "scores": [row["score"] for row in rows],
                    "output": str(output_path),
                }
            )
    finally:
        renderer.close()
    report_path = args.output_dir / "prediction_overlay_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Rendered {len(report)} images / {sum(r['prediction_count'] for r in report)} poses")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()

