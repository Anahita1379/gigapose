"""Visualize GigaPose predictions against EPnPv2 labels on real images.

Input is a CSV written by ``fine_tuning.select_real_label_candidates``.  The
script draws projected CAD 3D bounding boxes:

  - EPnPv2 pose/reference: green
  - GigaPose pose aligned into the EPnPv2 frame: blue

Typical use:

    python -m fine_tuning.visualize_epnp_gigapose_comparison \
      --candidate-csv gigaPose_datasets/results/real_20260505_front_gsam_v4_label_candidates/best_candidate_per_epnp_label.csv \
      --dataset-dir gigaPose_datasets/datasets/real_20260505_front_gsam_v4 \
      --output-dir gigaPose_datasets/results/real_20260505_front_gsam_v4_label_candidates/visual_overlays
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


EPNP_COLOR = (0, 255, 80)
GIGAPOSE_COLOR = (60, 140, 255)
MASK_COLOR = (255, 210, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        required=True,
        help=(
            "CSV from select_real_label_candidates.py. Use best_candidate_per_epnp_label.csv "
            "or selected_samples.csv for the clearest visuals."
        ),
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument(
        "--sort-by",
        choices=("image", "translation_error", "rotation_error", "score"),
        default="image",
    )
    parser.add_argument(
        "--draw-mask-bbox",
        action="store_true",
        help="Also draw the Grounded-SAM/Detection bbox from frame_map.json, if available.",
    )
    parser.add_argument(
        "--bbox-match-mode",
        choices=("instance_id", "nearest_projected_center"),
        default="nearest_projected_center",
        help=(
            "How to choose which detection bbox to draw when an image has multiple cars. "
            "nearest_projected_center is safer for multi-car real data; instance_id keeps "
            "the old behavior."
        ),
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Optional score filter before visualization.",
    )
    parser.add_argument(
        "--max-translation-error-mm",
        type=float,
        default=None,
        help=(
            "Optional filter for candidate CSVs. Useful when visualizing "
            "best_candidate_per_epnp_label.csv, which can still contain bad matches."
        ),
    )
    parser.add_argument(
        "--max-rotation-error-deg",
        type=float,
        default=None,
        help=(
            "Optional filter for candidate CSVs. Useful when visualizing "
            "best_candidate_per_epnp_label.csv, which can still contain bad matches."
        ),
    )
    return parser.parse_args()


def text_to_matrix(value: str) -> np.ndarray:
    vals = np.fromstring(str(value).strip().strip("[]").replace(";", " "), sep=" ")
    if vals.size != 16:
        raise ValueError(f"Expected 16 values for matrix text, got {vals.size}")
    return vals.reshape(4, 4)


def read_ply_vertices(path: Path) -> np.ndarray | None:
    if path is None or not path.is_file():
        return None
    data = path.read_bytes()
    header_end = data.find(b"end_header\n")
    if header_end < 0:
        return None
    header_end += len(b"end_header\n")
    header = data[:header_end].decode("latin1")
    vertex_count = None
    for line in header.splitlines():
        if line.startswith("element vertex "):
            vertex_count = int(line.split()[-1])
            break
    if vertex_count is None:
        return None

    if "format binary_little_endian" in header:
        vertices = np.frombuffer(
            data[header_end : header_end + vertex_count * 12],
            dtype="<f4",
        ).reshape(vertex_count, 3)
    elif "format ascii" in header:
        rows = data[header_end:].decode("latin1").splitlines()[:vertex_count]
        vertices = np.asarray([[float(v) for v in row.split()[:3]] for row in rows])
    else:
        return None

    vertices = np.asarray(vertices, dtype=float)
    if np.linalg.norm(vertices.ptp(axis=0)) < 100.0:
        vertices = vertices * 1000.0
    return vertices


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


def project(points_obj_mm: np.ndarray, T_cam_obj: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R = T_cam_obj[:3, :3]
    t = T_cam_obj[:3, 3]
    points_cam = (R @ points_obj_mm.T).T + t.reshape(1, 3)
    valid = points_cam[:, 2] > 1e-6
    uvw = (K @ points_cam.T).T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-9)
    return uv, valid


def project_pose_center(T_cam_obj: np.ndarray, K: np.ndarray) -> tuple[np.ndarray | None, bool]:
    uv, valid = project(np.zeros((1, 3), dtype=float), T_cam_obj, K)
    if not bool(valid[0]) or not np.all(np.isfinite(uv[0])):
        return None, False
    return uv[0], True


def draw_projected_box(
    draw: ImageDraw.ImageDraw,
    corners_mm: np.ndarray,
    T_cam_obj: np.ndarray,
    K: np.ndarray,
    color: tuple[int, int, int],
    label: str,
    label_offset: int,
    width: int = 3,
) -> None:
    uv, valid = project(corners_mm, T_cam_obj, K)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        if valid[i] and valid[j]:
            draw.line([tuple(uv[i].round()), tuple(uv[j].round())], fill=color, width=width)

    center_uv, center_valid = project(np.zeros((1, 3), dtype=float), T_cam_obj, K)
    if bool(center_valid[0]):
        x, y = center_uv[0]
        r = 5
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=width)

    draw.rectangle((8, 8 + label_offset, 620, 31 + label_offset), fill=(0, 0, 0))
    draw.text((12, 12 + label_offset), label, fill=color)


def draw_bbox(draw: ImageDraw.ImageDraw, bbox: list[float], label: str) -> None:
    x, y, w, h = bbox
    draw.rectangle((x, y, x + w, y + h), outline=MASK_COLOR, width=2)
    draw.text((x, max(0, y - 14)), label, fill=MASK_COLOR)


def bbox_center(bbox: list[float]) -> np.ndarray:
    x, y, w, h = map(float, bbox)
    return np.asarray([x + 0.5 * w, y + 0.5 * h], dtype=float)


def choose_detection_bbox(
    instances: list[dict[str, Any]],
    row: dict[str, str],
    T_gigapose: np.ndarray,
    T_epnp: np.ndarray,
    K: np.ndarray,
    mode: str,
) -> tuple[list[float] | None, int | None, float | None]:
    if not instances:
        return None, None, None

    if mode == "instance_id":
        idx = int(float(row.get("instance_id") or 0))
        idx = min(max(idx, 0), len(instances) - 1)
        bbox = instances[idx].get("bbox")
        return (list(map(float, bbox)), idx, None) if bbox else (None, idx, None)

    center_uv, ok = project_pose_center(T_gigapose, K)
    if not ok:
        center_uv, ok = project_pose_center(T_epnp, K)
    if not ok or center_uv is None:
        return None, None, None

    best_idx = None
    best_bbox = None
    best_dist = None
    for idx, instance in enumerate(instances):
        bbox = instance.get("bbox")
        if not bbox:
            continue
        dist = float(np.linalg.norm(bbox_center(list(map(float, bbox))) - center_uv))
        if best_dist is None or dist < best_dist:
            best_idx = idx
            best_bbox = list(map(float, bbox))
            best_dist = dist
    return best_bbox, best_idx, best_dist


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def maybe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def keep_row(row: dict[str, str], args: argparse.Namespace) -> bool:
    if not row.get("T_epnp_obj") or not row.get("T_gigapose_aligned_epnp_obj"):
        return False
    score = maybe_float(row.get("score"))
    trans = maybe_float(row.get("translation_error_mm"))
    rot = maybe_float(row.get("rotation_error_deg"))
    if args.min_score is not None and (score is None or score < args.min_score):
        return False
    if args.max_translation_error_mm is not None and (
        trans is None or trans > args.max_translation_error_mm
    ):
        return False
    if args.max_rotation_error_deg is not None and (
        rot is None or rot > args.max_rotation_error_deg
    ):
        return False
    return True


def load_frame_map(dataset_dir: Path) -> dict[tuple[int, int], dict[str, Any]]:
    path = dataset_dir / "frame_map.json"
    if not path.is_file():
        return {}
    rows = json.loads(path.read_text())
    return {(int(row["scene_id"]), int(row["im_id"])): row for row in rows}


def shard_path_for_key(split_dir: Path, key: str) -> Path:
    mapping = json.loads((split_dir / "key_to_shard.json").read_text())
    return split_dir / f"shard-{int(mapping[key]):06d}.tar"


def load_camera_K(dataset_dir: Path, split: str, scene_id: int, im_id: int) -> np.ndarray:
    split_dir = dataset_dir / split
    key = f"{scene_id:06d}_{im_id:06d}"
    shard_path = shard_path_for_key(split_dir, key)
    with tarfile.open(shard_path) as tar:
        camera = json.loads(tar.extractfile(f"{key}.camera.json").read())
    return np.asarray(camera["cam_K"], dtype=float).reshape(3, 3)


def load_image(dataset_dir: Path, split: str, scene_id: int, im_id: int, frame_info: dict[str, Any]) -> Image.Image:
    image_path = frame_info.get("image_path")
    if image_path and Path(image_path).is_file():
        return Image.open(image_path).convert("RGB")

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
        return Image.open(io.BytesIO(tar.extractfile(rgb_name).read())).convert("RGB")


def row_sort_key(row: dict[str, str], mode: str) -> Any:
    if mode == "translation_error":
        return -float(row.get("translation_error_mm", "nan"))
    if mode == "rotation_error":
        return -float(row.get("rotation_error_deg", "nan"))
    if mode == "score":
        return -float(row.get("score", "nan"))
    return (int(row["scene_id"]), int(row["im_id"]), int(row.get("epnp_record_index") or 0))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = args.mesh or args.dataset_dir / "models" / "obj_000001.ply"
    vertices_mm = read_ply_vertices(mesh_path)
    if vertices_mm is None:
        raise ValueError(f"Could not read mesh vertices from {mesh_path}")
    corners_mm = bbox_corners_from_vertices(vertices_mm)

    frame_map = load_frame_map(args.dataset_dir)
    rows = read_rows(args.candidate_csv)
    input_row_count = len(rows)
    rows = [row for row in rows if keep_row(row, args)]
    rows.sort(key=lambda row: row_sort_key(row, args.sort_by))
    rows = rows[:: max(args.every, 1)]
    if args.max_images is not None:
        rows = rows[: args.max_images]

    index_rows = []
    for row in rows:
        scene_id = int(row["scene_id"])
        im_id = int(row["im_id"])
        frame_info = frame_map.get((scene_id, im_id), {})
        image = load_image(args.dataset_dir, args.split, scene_id, im_id, frame_info)
        K = load_camera_K(args.dataset_dir, args.split, scene_id, im_id)

        T_epnp = text_to_matrix(row["T_epnp_obj"])
        T_giga = text_to_matrix(row["T_gigapose_aligned_epnp_obj"])

        draw = ImageDraw.Draw(image)
        draw_projected_box(
            draw,
            corners_mm,
            T_epnp,
            K,
            EPNP_COLOR,
            "EPnPv2",
            0,
        )
        draw_projected_box(
            draw,
            corners_mm,
            T_giga,
            K,
            GIGAPOSE_COLOR,
            (
                f"GigaPose aligned: t={float(row['translation_error_mm']):.0f}mm "
                f"R={float(row['rotation_error_deg']):.1f}deg score={float(row['score']):.3f}"
            ),
            26,
        )

        if args.draw_mask_bbox:
            instances = frame_info.get("instances") or []
            if instances:
                bbox, bbox_idx, bbox_dist = choose_detection_bbox(
                    instances,
                    row,
                    T_giga,
                    T_epnp,
                    K,
                    args.bbox_match_mode,
                )
                if bbox:
                    label = "GSAM/detection bbox"
                    if bbox_idx is not None:
                        label += f" #{bbox_idx}"
                    if bbox_dist is not None:
                        label += f" d={bbox_dist:.0f}px"
                    draw_bbox(draw, bbox, label)
            else:
                bbox_idx = None
                bbox_dist = None
        else:
            bbox_idx = None
            bbox_dist = None

        output_path = args.output_dir / f"{scene_id:06d}_{im_id:06d}_epnp{int(row.get('epnp_record_index') or 0):02d}.jpg"
        image.save(output_path, quality=94)
        index_rows.append(
            {
                "scene_id": scene_id,
                "im_id": im_id,
                "match_key": row.get("match_key", ""),
                "score": row.get("score", ""),
                "translation_error_mm": row.get("translation_error_mm", ""),
                "rotation_error_deg": row.get("rotation_error_deg", ""),
                "epnp_label_path": row.get("epnp_label_path", ""),
                "bbox_match_mode": args.bbox_match_mode if args.draw_mask_bbox else "",
                "bbox_match_index": "" if bbox_idx is None else bbox_idx,
                "bbox_match_distance_px": "" if bbox_dist is None else f"{bbox_dist:.6g}",
                "output": str(output_path),
            }
        )

    with (args.output_dir / "visual_overlay_index.csv").open("w", newline="") as handle:
        fieldnames = list(index_rows[0].keys()) if index_rows else [
            "scene_id",
            "im_id",
            "match_key",
            "score",
            "translation_error_mm",
            "rotation_error_deg",
            "epnp_label_path",
            "bbox_match_mode",
            "bbox_match_index",
            "bbox_match_distance_px",
            "output",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    print(
        f"Wrote {len(index_rows)} EPnPv2/GigaPose overlays to {args.output_dir} "
        f"({len(rows)} visualized after filters from {input_row_count} input rows)"
    )


if __name__ == "__main__":
    main()
