"""Visualize extrinsic optimization before/after on real images.

For each selected sample, this draws projected CAD boxes from the map-frame
object pose using:

  - original metadata calibration:
      T_map_cam = t_map_lidar @ t_lidar_camera_prior
  - optimized calibration:
      T_map_cam = t_map_lidar @ T_correction @ t_lidar_camera_prior
  - optional GigaPose camera-object pose from selected_samples.csv

Typical use:

    python -m fine_tuning.visualize_extrinsic_optimization_on_images \
      --selected-samples gigaPose_datasets/results/real_world_data/combined_front_selected_samples.csv \
      --optimization-dir gigaPose_datasets/results/real_world_data/extrinsic_optimization_front_metadata_xy \
      --mesh gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
      --output-dir gigaPose_datasets/results/real_world_data/extrinsic_optimization_front_metadata_xy/image_overlays
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


ORIGINAL_COLOR = (255, 80, 70)
OPTIMIZED_COLOR = (70, 150, 255)
GIGAPOSE_COLOR = (0, 255, 100)
BBOX_COLOR = (255, 215, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-samples", type=Path, required=True)
    parser.add_argument("--optimization-dir", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument(
        "--sort-by",
        choices=("image", "translation_error", "rotation_error", "score"),
        default="image",
    )
    parser.add_argument(
        "--draw-gigapose",
        action="store_true",
        help="Also draw the selected GigaPose camera-object pose.",
    )
    parser.add_argument(
        "--draw-detection-bbox",
        action="store_true",
        help="Draw detector/ROI bbox from metadata when available.",
    )
    parser.add_argument(
        "--map-z-mode",
        choices=(
            "raw",
            "metadata_lidar",
            "ground_truth_pose",
            "ego_relative",
            "session_lidar_offset",
            "session_lidar_affine",
        ),
        default="raw",
        help=(
            "How to handle T_map_object_raw z before projecting. 'raw' uses the label z as-is. "
            "'metadata_lidar' shifts label z into metadata t_map_lidar's z convention. "
            "'ground_truth_pose' shifts label z into metadata ground_truth_pose z convention. "
            "'ego_relative' preserves the label object's height relative to metadata "
            "ground_truth_pose, but expresses it in metadata t_map_lidar's z convention. "
            "'session_lidar_offset' estimates one median z offset per session and "
            "preserves EPnP z variation while shifting it into metadata lidar z. "
            "'session_lidar_affine' uses median_lidar_z + session_z_scale * "
            "(label_z - median_label_z). "
            "This is for visualization only when altitude conventions differ."
        ),
    )
    parser.add_argument(
        "--session-z-scale",
        type=float,
        default=1.0,
        help=(
            "Scale for EPnP hill/downhill z variation with --map-z-mode "
            "session_lidar_affine. 0 flattens to session median lidar z; 1 "
            "preserves full EPnP variation; negative values invert it."
        ),
    )
    parser.add_argument(
        "--write-debug-projections",
        action="store_true",
        help="Write projection center/depth diagnostics into overlay_index.csv.",
    )
    parser.add_argument(
        "--projection-model",
        choices=("pinhole", "metadata"),
        default="pinhole",
        help=(
            "Projection model for drawing. 'pinhole' preserves old behavior. "
            "'metadata' uses metadata distortion_model and camera_intrinsics.d "
            "when available, including plumb_bob and equidistant."
        ),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def text_to_matrix(value: str) -> np.ndarray:
    vals = np.fromstring(str(value).strip().strip("[]").replace(";", " "), sep=" ")
    if vals.size != 16:
        raise ValueError(f"Expected 16 values, got {vals.size}")
    return vals.reshape(4, 4)


def matrix_3x4_or_4x4(value: Any, unit: str = "m") -> np.ndarray:
    flat = np.asarray(value, dtype=float).reshape(-1)
    if flat.size == 16:
        T = flat.reshape(4, 4).copy()
    elif flat.size == 12:
        T = np.eye(4, dtype=float)
        T[:3, :] = flat.reshape(3, 4)
    else:
        raise ValueError(f"Expected 12 or 16 values, got {flat.size}")
    if unit == "m" or (unit == "auto" and np.nanmedian(np.abs(T[:3, 3])) < 1000.0):
        T[:3, 3] *= 1000.0
    return T


def read_ply_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    header_end = data.find(b"end_header\n")
    if header_end < 0:
        raise ValueError(f"Could not find PLY header end in {path}")
    header_end += len(b"end_header\n")
    header = data[:header_end].decode("latin1")
    vertex_count = None
    for line in header.splitlines():
        if line.startswith("element vertex "):
            vertex_count = int(line.split()[-1])
            break
    if vertex_count is None:
        raise ValueError(f"Could not find vertex count in {path}")
    if "format binary_little_endian" in header:
        vertices = np.frombuffer(data[header_end : header_end + vertex_count * 12], dtype="<f4").reshape(
            vertex_count, 3
        )
    elif "format ascii" in header:
        rows = data[header_end:].decode("latin1").splitlines()[:vertex_count]
        vertices = np.asarray([[float(v) for v in row.split()[:3]] for row in rows])
    else:
        raise ValueError(f"Unsupported PLY format in {path}")
    vertices = np.asarray(vertices, dtype=float)
    if np.linalg.norm(vertices.ptp(axis=0)) < 100.0:
        vertices *= 1000.0
    return vertices


def bbox_corners(vertices_mm: np.ndarray) -> np.ndarray:
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


def metadata_path_from_row(row: dict[str, str]) -> Path:
    for key in ("sample_metadata_path", "metadata_path"):
        value = row.get(key)
        if value and Path(value).is_file():
            return Path(value)
    label = row.get("epnp_label_path", "")
    if not label:
        raise ValueError("No epnp_label_path to infer metadata path")
    label_path = Path(label)
    match = re.match(r"(.+)_([0-9]+)$", label_path.stem)
    if not match:
        raise ValueError(f"Cannot infer metadata from label name {label_path.name}")
    ts, obj = match.groups()
    return label_path.parent.parent / "metadata" / f"sample_{ts}_{obj}.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("This script needs PyYAML: pip install pyyaml or conda install pyyaml") from exc
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Metadata YAML is not a mapping: {path}")
    return data


def load_image(metadata: dict[str, Any], metadata_path: Path) -> Image.Image:
    image_path = Path(str(metadata.get("image_path", "")))
    if not image_path.is_absolute():
        image_path = metadata_path.parent.parent / image_path
    if image_path.is_file():
        return Image.open(image_path).convert("RGB")
    source_image_path = metadata.get("source_image_path")
    if source_image_path and Path(str(source_image_path)).is_file():
        return Image.open(str(source_image_path)).convert("RGB")
    raise FileNotFoundError(f"Could not find image for metadata {metadata_path}")


def load_camera(metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    intrinsics = metadata.get("camera_intrinsics", {})
    if "k" in intrinsics:
        K = np.asarray(intrinsics["k"], dtype=float).reshape(3, 3)
        D = np.asarray(intrinsics.get("d", []), dtype=float).reshape(-1)
        model = str(intrinsics.get("distortion_model", metadata.get("distortion_model", "pinhole")))
        return K, D, model
    if "K" in metadata:
        K = np.asarray(metadata["K"], dtype=float).reshape(3, 3)
        D = np.asarray(metadata.get("D", []), dtype=float).reshape(-1)
        model = str(metadata.get("distortion_model", "pinhole"))
        return K, D, model
    raise KeyError("No camera intrinsics k/K found in metadata")


def load_target_map_pose(row: dict[str, str]) -> tuple[np.ndarray, dict[str, Any]]:
    label_path = Path(row["epnp_label_path"])
    data = json.loads(label_path.read_text())
    if "T_map_object_raw" not in data:
        raise KeyError(f"T_map_object_raw missing from {label_path}")
    return matrix_3x4_or_4x4(data["T_map_object_raw"], unit="auto"), data


def metadata_session_key(metadata_path: str | Path) -> str:
    path = Path(metadata_path)
    if path.parent.name == "metadata":
        return str(path.parent.parent)
    return str(path.parent)


def compute_session_lidar_z_stats(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    offsets_by_session: dict[str, list[float]] = {}
    label_z_by_session: dict[str, list[float]] = {}
    lidar_z_by_session: dict[str, list[float]] = {}
    for row in rows:
        try:
            metadata_path = metadata_path_from_row(row)
            metadata = load_yaml(metadata_path)
            T_map_obj_raw, _ = load_target_map_pose(row)
            lidar_z_m = float(np.asarray(metadata["t_map_lidar"], dtype=float).reshape(4, 4)[2, 3])
            label_z_m = float(T_map_obj_raw[2, 3] * 0.001)
        except Exception:
            continue
        session = metadata_session_key(metadata_path)
        offsets_by_session.setdefault(session, []).append(lidar_z_m - label_z_m)
        label_z_by_session.setdefault(session, []).append(label_z_m)
        lidar_z_by_session.setdefault(session, []).append(lidar_z_m)
    out: dict[str, dict[str, float]] = {}
    for key in set(offsets_by_session) | set(label_z_by_session) | set(lidar_z_by_session):
        out[key] = {
            "offset_m": float(np.median(np.asarray(offsets_by_session.get(key, [0.0]), dtype=float))),
            "label_z_median": float(np.median(np.asarray(label_z_by_session.get(key, [0.0]), dtype=float))),
            "lidar_z_median": float(np.median(np.asarray(lidar_z_by_session.get(key, [0.0]), dtype=float))),
        }
    return out


def apply_map_z_mode(
    T_map_obj: np.ndarray,
    label_data: dict[str, Any],
    metadata: dict[str, Any],
    mode: str,
    session_z_offset_m: float | None = None,
    session_label_z_median: float | None = None,
    session_lidar_z_median: float | None = None,
    session_z_scale: float = 1.0,
) -> np.ndarray:
    if mode == "raw":
        return T_map_obj
    out = T_map_obj.copy()
    if isinstance(label_data.get("map_pose"), dict):
        label_z_m = float(label_data["map_pose"].get("position", {}).get("z", out[2, 3] * 0.001))
    else:
        label_z_m = float(out[2, 3] * 0.001)

    if mode == "metadata_lidar":
        target_z_m = float(np.asarray(metadata["t_map_lidar"], dtype=float).reshape(4, 4)[2, 3])
    elif mode == "ground_truth_pose":
        target_z_m = float(metadata["ground_truth_pose"]["position"]["z"])
    elif mode == "ego_relative":
        lidar_z_m = float(np.asarray(metadata["t_map_lidar"], dtype=float).reshape(4, 4)[2, 3])
        ego_z_m = float(metadata["ground_truth_pose"]["position"]["z"])
        target_z_m = lidar_z_m + (label_z_m - ego_z_m)
    elif mode == "session_lidar_offset":
        if session_z_offset_m is None:
            raise ValueError("session_lidar_offset mode requires a precomputed session z offset.")
        target_z_m = label_z_m + session_z_offset_m
    elif mode == "session_lidar_affine":
        if session_label_z_median is None or session_lidar_z_median is None:
            raise ValueError("session_lidar_affine mode requires precomputed session z medians.")
        target_z_m = session_lidar_z_median + session_z_scale * (label_z_m - session_label_z_median)
    else:
        raise ValueError(f"Unknown map z mode: {mode}")
    out[2, 3] += (target_z_m - label_z_m) * 1000.0
    return out


def map_z_debug(
    T_map_obj_raw: np.ndarray,
    T_map_obj_adjusted: np.ndarray,
    label_data: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, float]:
    raw_z_m = float(T_map_obj_raw[2, 3] * 0.001)
    adjusted_z_m = float(T_map_obj_adjusted[2, 3] * 0.001)
    if isinstance(label_data.get("map_pose"), dict):
        label_z_m = float(label_data["map_pose"].get("position", {}).get("z", raw_z_m))
    else:
        label_z_m = raw_z_m
    lidar_z_m = float(np.asarray(metadata["t_map_lidar"], dtype=float).reshape(4, 4)[2, 3])
    ego_z_m = float(metadata.get("ground_truth_pose", {}).get("position", {}).get("z", float("nan")))
    return {
        "raw_map_object_z_m": raw_z_m,
        "label_map_pose_z_m": label_z_m,
        "adjusted_map_object_z_m": adjusted_z_m,
        "metadata_lidar_z_m": lidar_z_m,
        "metadata_ground_truth_pose_z_m": ego_z_m,
        "object_minus_ego_z_m": label_z_m - ego_z_m if np.isfinite(ego_z_m) else float("nan"),
        "adjusted_minus_lidar_z_m": adjusted_z_m - lidar_z_m,
    }


def project(
    points_obj_mm: np.ndarray,
    T_cam_obj: np.ndarray,
    K: np.ndarray,
    D: np.ndarray | None = None,
    distortion_model: str = "pinhole",
    projection_model: str = "pinhole",
) -> tuple[np.ndarray, np.ndarray]:
    points_cam = (T_cam_obj[:3, :3] @ points_obj_mm.T).T + T_cam_obj[:3, 3].reshape(1, 3)
    z = points_cam[:, 2]
    valid = z > 1e-6
    x = points_cam[:, 0] / np.maximum(z, 1e-9)
    y = points_cam[:, 1] / np.maximum(z, 1e-9)
    D = np.asarray([] if D is None else D, dtype=float).reshape(-1)

    if projection_model == "metadata" and distortion_model == "plumb_bob" and D.size >= 4:
        k1, k2, p1, p2 = D[:4]
        k3 = D[4] if D.size >= 5 else 0.0
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    elif projection_model == "metadata" and distortion_model == "equidistant" and D.size >= 4:
        k1, k2, k3, k4 = D[:4]
        r = np.sqrt(x * x + y * y)
        theta = np.arctan(r)
        theta2 = theta * theta
        theta_d = theta * (
            1.0
            + k1 * theta2
            + k2 * theta2 * theta2
            + k3 * theta2 * theta2 * theta2
            + k4 * theta2 * theta2 * theta2 * theta2
        )
        scale = np.divide(theta_d, r, out=np.ones_like(r), where=r > 1e-12)
        xd = x * scale
        yd = y * scale
    else:
        xd, yd = x, y

    uv = np.empty((points_obj_mm.shape[0], 2), dtype=float)
    uv[:, 0] = K[0, 0] * xd + K[0, 2]
    uv[:, 1] = K[1, 1] * yd + K[1, 2]
    valid &= np.isfinite(uv).all(axis=1)
    return uv, valid


def projection_debug(
    T_cam_obj: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    distortion_model: str,
    projection_model: str,
) -> dict[str, Any]:
    uv, valid = project(np.zeros((1, 3), dtype=float), T_cam_obj, K, D, distortion_model, projection_model)
    return {
        "center_u": float(uv[0, 0]),
        "center_v": float(uv[0, 1]),
        "center_depth_mm": float(T_cam_obj[2, 3]),
        "center_valid_depth": bool(valid[0]),
    }


def draw_projected_box(
    draw: ImageDraw.ImageDraw,
    corners_mm: np.ndarray,
    T_cam_obj: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    distortion_model: str,
    projection_model: str,
    color: tuple[int, int, int],
    label: str,
    label_offset: int,
    width: int = 3,
) -> None:
    uv, valid = project(corners_mm, T_cam_obj, K, D, distortion_model, projection_model)
    center_uv, center_valid = project(
        np.zeros((1, 3), dtype=float), T_cam_obj, K, D, distortion_model, projection_model
    )
    visible_edges = 0
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        if valid[i] and valid[j]:
            draw.line([tuple(uv[i].round()), tuple(uv[j].round())], fill=color, width=width)
            visible_edges += 1
    if bool(center_valid[0]):
        x, y = center_uv[0]
        r = 5
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=max(width, 3))
        draw.line((x - 8, y, x + 8, y), fill=color, width=max(width, 2))
        draw.line((x, y - 8, x, y + 8), fill=color, width=max(width, 2))
    else:
        # If the center is behind the camera, put a small diagnostic marker in
        # the legend area instead of silently drawing nothing.
        x = 680
        y = 19 + label_offset
        draw.line((x - 7, y - 7, x + 7, y + 7), fill=color, width=max(width, 2))
        draw.line((x - 7, y + 7, x + 7, y - 7), fill=color, width=max(width, 2))
    draw.rectangle((8, 8 + label_offset, 700, 31 + label_offset), fill=(0, 0, 0))
    depth_m = float(T_cam_obj[2, 3] * 0.001)
    status = f"{label} | center=({center_uv[0,0]:.0f},{center_uv[0,1]:.0f}) z={depth_m:.1f}m edges={visible_edges}"
    if not bool(center_valid[0]):
        status += " BEHIND"
    draw.text((12, 12 + label_offset), status, fill=color)


def draw_bbox(draw: ImageDraw.ImageDraw, metadata: dict[str, Any]) -> None:
    bbox = metadata.get("yolo_bbox") or metadata.get("detector_bbox_xywh")
    if isinstance(bbox, dict):
        x, y, w, h = [float(bbox[k]) for k in ("x", "y", "width", "height")]
    elif isinstance(bbox, list) and len(bbox) >= 4:
        x, y, w, h = [float(v) for v in bbox[:4]]
    else:
        return
    draw.rectangle((x, y, x + w, y + h), outline=BBOX_COLOR, width=2)
    draw.text((x, max(0, y - 14)), "detection bbox", fill=BBOX_COLOR)


def row_sort_value(row: dict[str, str], mode: str) -> Any:
    if mode == "translation_error":
        return -float(row.get("translation_error_mm", "nan"))
    if mode == "rotation_error":
        return -float(row.get("rotation_error_deg", "nan"))
    if mode == "score":
        return -float(row.get("score", "nan"))
    return row.get("match_key", "")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    extrinsics = json.loads((args.optimization_dir / "optimized_extrinsics.json").read_text())
    correction = np.asarray(extrinsics["T_lidar_camera_correction_left_multiply"], dtype=float).reshape(4, 4)
    corners = bbox_corners(read_ply_vertices(args.mesh))

    rows = read_csv(args.selected_samples)
    session_z_stats = (
        compute_session_lidar_z_stats(rows)
        if args.map_z_mode in ("session_lidar_offset", "session_lidar_affine")
        else {}
    )
    rows.sort(key=lambda row: row_sort_value(row, args.sort_by))
    rows = rows[:: max(args.every, 1)]
    if args.max_images is not None:
        rows = rows[: args.max_images]

    index_rows = []
    for row in rows:
        try:
            metadata_path = metadata_path_from_row(row)
            metadata = load_yaml(metadata_path)
            image = load_image(metadata, metadata_path)
            K, D, distortion_model = load_camera(metadata)
            T_map_lidar = matrix_3x4_or_4x4(metadata["t_map_lidar"], unit="m")
            T_lidar_cam_prior = matrix_3x4_or_4x4(metadata["t_lidar_camera_prior"], unit="m")
            T_map_obj_raw, label_data = load_target_map_pose(row)
            z_stats = session_z_stats.get(metadata_session_key(metadata_path), {})
            T_map_obj = apply_map_z_mode(
                T_map_obj_raw,
                label_data,
                metadata,
                args.map_z_mode,
                z_stats.get("offset_m"),
                z_stats.get("label_z_median"),
                z_stats.get("lidar_z_median"),
                args.session_z_scale,
            )
            z_debug = map_z_debug(T_map_obj_raw, T_map_obj, label_data, metadata)
            T_map_cam_prior = T_map_lidar @ T_lidar_cam_prior
            T_map_cam_opt = T_map_lidar @ correction @ T_lidar_cam_prior
            T_cam_obj_prior = np.linalg.inv(T_map_cam_prior) @ T_map_obj
            T_cam_obj_opt = np.linalg.inv(T_map_cam_opt) @ T_map_obj
            T_gigapose = text_to_matrix(row["T_gigapose_cam_obj"])
        except Exception as exc:
            print(f"Skipping row {row.get('match_key', '')}: {exc}")
            continue

        draw = ImageDraw.Draw(image)
        draw_projected_box(
            draw,
            corners,
            T_cam_obj_prior,
            K,
            D,
            distortion_model,
            args.projection_model,
            ORIGINAL_COLOR,
            "map pose via original extrinsic",
            0,
        )
        draw_projected_box(
            draw,
            corners,
            T_cam_obj_opt,
            K,
            D,
            distortion_model,
            args.projection_model,
            OPTIMIZED_COLOR,
            "map pose via optimized extrinsic",
            26,
        )
        if args.draw_gigapose:
            draw_projected_box(
                draw,
                corners,
                T_gigapose,
                K,
                D,
                distortion_model,
                args.projection_model,
                GIGAPOSE_COLOR,
                "selected GigaPose pose",
                52,
            )
        if args.draw_detection_bbox:
            draw_bbox(draw, metadata)

        stem = Path(row.get("epnp_label_path", row.get("match_key", "sample"))).stem
        output_path = args.output_dir / f"{stem}_extrinsic_before_after.jpg"
        image.save(output_path, quality=94)
        index_row = {
            "match_key": row.get("match_key", ""),
            "metadata_path": str(metadata_path),
            "epnp_label_path": row.get("epnp_label_path", ""),
            "map_z_mode": args.map_z_mode,
            "translation_error_mm": row.get("translation_error_mm", ""),
            "rotation_error_deg": row.get("rotation_error_deg", ""),
            "output": str(output_path),
            **z_debug,
            "projection_model": args.projection_model,
            "metadata_distortion_model": distortion_model,
        }
        if args.write_debug_projections:
            for prefix, T in (
                ("original", T_cam_obj_prior),
                ("optimized", T_cam_obj_opt),
                ("gigapose", T_gigapose),
            ):
                for key, value in projection_debug(T, K, D, distortion_model, args.projection_model).items():
                    index_row[f"{prefix}_{key}"] = value
        index_rows.append(index_row)

    with (args.output_dir / "overlay_index.csv").open("w", newline="") as handle:
        fieldnames = list(index_rows[0].keys()) if index_rows else [
            "match_key",
            "metadata_path",
            "epnp_label_path",
            "translation_error_mm",
            "rotation_error_deg",
            "output",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    print(f"Wrote {len(index_rows)} extrinsic before/after image overlays to {args.output_dir}")


if __name__ == "__main__":
    main()
