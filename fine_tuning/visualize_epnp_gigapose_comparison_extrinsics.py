"""Visualize extrinsics-corrected EPnP and aligned GigaPose poses.

This is the extrinsics-aware counterpart of
``fine_tuning.visualize_epnp_gigapose_comparison``. The original visualizer is
left unchanged.

The candidate selector compares poses after an empirical frame alignment:

    right alignment: T_gigapose_aligned = T_gigapose_raw @ X

That fitted ``X`` may absorb prediction bias as well as a true object-frame
difference. In particular, it can contain metres of translation, so it must
not be treated as a physical CAD-coordinate transform.

This visualizer explicitly converts the raw CAD vertices into the centered
object convention used by the EPnP labels and renders both comparison poses:

    GigaPose: T_gigapose_aligned
    EPnP:     T_epnp_corrected

For the race-car labels, the centered-pose origin is not the mesh AABB center.
It is the raw-CAD point

    center_raw = [-0.2411941141, 0.0009010172, 0.3329219520] metres

and therefore the vertices rendered with a centered pose are

    p_centered = p_raw - center_raw.

The raw/aligned poses are still used to reconstruct and validate ``X`` for
diagnostics, but ``X`` is never applied to mesh vertices or the EPnP pose.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import ImageDraw

from fine_tuning import visualize_epnp_gigapose_comparison as base


DEFAULT_RAW_OBJECT_CENTER_M = (-0.2411941141, 0.0009010172, 0.3329219520)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        required=True,
        help="selected_samples.csv from the extrinsics-aware candidate selector.",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-images",
        type=int,
        default=100,
        help=(
            "Maximum number of overlay JPEGs to write. Center-offset diagnostics "
            "are still calculated for every row remaining after filters and --every."
        ),
    )
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument(
        "--sort-by",
        choices=("image", "translation_error", "rotation_error", "score"),
        default="image",
    )
    parser.add_argument("--draw-mask-bbox", action="store_true")
    parser.add_argument(
        "--bbox-match-mode",
        choices=("instance_id", "nearest_projected_center"),
        default="nearest_projected_center",
    )
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--max-translation-error-mm", type=float, default=None)
    parser.add_argument("--max-rotation-error-deg", type=float, default=None)
    parser.add_argument(
        "--frame-transform-side",
        choices=("right", "left"),
        default="right",
        help=(
            "Alignment convention used by the selector. Your current candidate "
            "runs use 'right'. The transform is recovered from each CSV row."
        ),
    )
    parser.add_argument(
        "--projection-model",
        choices=("metadata", "pinhole"),
        default="metadata",
        help=(
            "Projection used for the CAD boxes, pose centers, bbox association, "
            "and center-offset diagnostics. 'metadata' reads K, D, and the "
            "distortion model from each sample metadata YAML and supports the "
            "equidistant camera model."
        ),
    )
    parser.add_argument(
        "--object-center-mode",
        choices=("epnp_raw", "mesh_aabb"),
        default="epnp_raw",
        help=(
            "Coordinate origin represented by the input centered poses. "
            "'epnp_raw' subtracts --raw-object-center-m from raw CAD vertices "
            "(the correct convention for the EPnPv2 race-car labels). "
            "'mesh_aabb' preserves the previous AABB-centering behavior."
        ),
    )
    parser.add_argument(
        "--raw-object-center-m",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=DEFAULT_RAW_OBJECT_CENTER_M,
        help=(
            "Raw-CAD point, in metres, used as the origin of "
            "T_camera_object_centered. Default: %(default)s."
        ),
    )
    return parser.parse_args()


def metadata_path_from_row(row: dict[str, str]) -> Path:
    """Resolve the source metadata YAML paired with a selected sample."""

    for key in ("sample_metadata_path", "metadata_path"):
        value = row.get(key)
        if value and Path(value).is_file():
            return Path(value)

    label_text = row.get("epnp_label_path", "")
    if not label_text:
        raise FileNotFoundError(
            "Candidate row has no usable sample_metadata_path or epnp_label_path"
        )
    label_path = Path(label_text)
    if label_path.is_file():
        try:
            label_data = json.loads(label_path.read_text())
        except (OSError, ValueError, TypeError):
            label_data = {}
        value = (
            label_data.get("metadata_path") if isinstance(label_data, dict) else None
        )
        if value and Path(str(value)).is_file():
            return Path(str(value))

    match = re.match(r"(.+)_([0-9]+)$", label_path.stem)
    if match:
        timestamp, object_index = match.groups()
        inferred = (
            label_path.parent.parent
            / "metadata"
            / f"sample_{timestamp}_{object_index}.yaml"
        )
        if inferred.is_file():
            return inferred

    requested = row.get("sample_metadata_path") or row.get("metadata_path") or ""
    raise FileNotFoundError(
        f"Could not resolve metadata YAML for {label_path}; requested={requested!r}"
    )


def load_metadata_camera(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "Metadata projection requires PyYAML: install pyyaml in this environment."
        ) from exc

    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Metadata YAML is not a mapping: {path}")
    intrinsics = data.get("camera_intrinsics", {})
    if isinstance(intrinsics, dict) and "k" in intrinsics:
        K = np.asarray(intrinsics["k"], dtype=float).reshape(3, 3)
        D = np.asarray(intrinsics.get("d", []), dtype=float).reshape(-1)
        model = str(
            intrinsics.get("distortion_model", data.get("distortion_model", "pinhole"))
        ).lower()
        if model == "equidistant" and D.size < 4:
            raise ValueError(
                f"Equidistant metadata camera requires at least four D values: {path}"
            )
        return K, D, model
    if "K" in data:
        K = np.asarray(data["K"], dtype=float).reshape(3, 3)
        D = np.asarray(data.get("D", []), dtype=float).reshape(-1)
        model = str(data.get("distortion_model", "pinhole")).lower()
        if model == "equidistant" and D.size < 4:
            raise ValueError(
                f"Equidistant metadata camera requires at least four D values: {path}"
            )
        return K, D, model
    raise KeyError(f"No camera_intrinsics.k or K in metadata YAML: {path}")


def project_points(
    points_obj_mm: np.ndarray,
    T_cam_obj: np.ndarray,
    K: np.ndarray,
    D: np.ndarray | None,
    distortion_model: str,
    projection_model: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Project with the same distortion convention as extrinsic optimization."""

    points_cam = (T_cam_obj[:3, :3] @ points_obj_mm.T).T + T_cam_obj[:3, 3].reshape(
        1, 3
    )
    z = points_cam[:, 2]
    valid = z > 1e-6
    x = points_cam[:, 0] / np.maximum(z, 1e-9)
    y = points_cam[:, 1] / np.maximum(z, 1e-9)
    D = np.asarray([] if D is None else D, dtype=float).reshape(-1)
    model = distortion_model.lower()

    if projection_model == "metadata" and model == "plumb_bob" and D.size >= 4:
        k1, k2, p1, p2 = D[:4]
        k3 = D[4] if D.size >= 5 else 0.0
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    elif projection_model == "metadata" and model == "equidistant" and D.size >= 4:
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


def project_pose_center(
    T_cam_obj: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    distortion_model: str,
    projection_model: str,
) -> tuple[np.ndarray | None, bool]:
    uv, valid = project_points(
        np.zeros((1, 3), dtype=float),
        T_cam_obj,
        K,
        D,
        distortion_model,
        projection_model,
    )
    if not bool(valid[0]) or not np.all(np.isfinite(uv[0])):
        return None, False
    return uv[0], True


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
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for i, j in edges:
        # Under an equidistant camera model, a straight 3D box edge need not
        # remain a straight image-space segment. Sample along each edge before
        # projection so the overlay follows the actual distortion model.
        alpha = np.linspace(0.0, 1.0, 17).reshape(-1, 1)
        edge_points = (1.0 - alpha) * corners_mm[i] + alpha * corners_mm[j]
        edge_uv, edge_valid = project_points(
            edge_points,
            T_cam_obj,
            K,
            D,
            distortion_model,
            projection_model,
        )
        visible_uv = edge_uv[edge_valid]
        if visible_uv.shape[0] >= 2:
            draw.line(
                [tuple(point.round()) for point in visible_uv],
                fill=color,
                width=width,
            )

    center_uv, center_valid = project_pose_center(
        T_cam_obj, K, D, distortion_model, projection_model
    )
    if center_valid and center_uv is not None:
        x, y = center_uv
        radius = 5
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=color,
            width=width,
        )
        draw.line((x - 8, y, x + 8, y), fill=color, width=max(width, 2))
        draw.line((x, y - 8, x, y + 8), fill=color, width=max(width, 2))

    draw.rectangle((8, 8 + label_offset, 720, 31 + label_offset), fill=(0, 0, 0))
    draw.text((12, 12 + label_offset), label, fill=color)


def choose_detection_bbox(
    instances: list[dict[str, Any]],
    row: dict[str, str],
    T_gigapose: np.ndarray,
    T_epnp: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    distortion_model: str,
    projection_model: str,
    mode: str,
) -> tuple[list[float] | None, int | None, float | None]:
    if not instances:
        return None, None, None
    if mode == "instance_id":
        index = int(float(row.get("instance_id") or 0))
        index = min(max(index, 0), len(instances) - 1)
        bbox = instances[index].get("bbox")
        return (list(map(float, bbox)), index, None) if bbox else (None, index, None)

    center_uv, ok = project_pose_center(
        T_gigapose, K, D, distortion_model, projection_model
    )
    if not ok:
        center_uv, ok = project_pose_center(
            T_epnp, K, D, distortion_model, projection_model
        )
    if not ok or center_uv is None:
        return None, None, None

    best_bbox = None
    best_index = None
    best_distance = None
    for index, instance in enumerate(instances):
        bbox = instance.get("bbox")
        if not bbox:
            continue
        bbox = list(map(float, bbox))
        distance = float(np.linalg.norm(base.bbox_center(bbox) - center_uv))
        if best_distance is None or distance < best_distance:
            best_bbox = bbox
            best_index = index
            best_distance = distance
    return best_bbox, best_index, best_distance


def finite_summary(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "p10": None,
            "p90": None,
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
    }


def aligned_pose_for_row(row: dict[str, str]) -> np.ndarray:
    value = row.get("T_gigapose_aligned_epnp_obj")
    if not value:
        raise ValueError("Candidate row is missing T_gigapose_aligned_epnp_obj.")
    return base.text_to_matrix(value)


def centered_comparison_poses(
    row: dict[str, str],
    frame_transform_side: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return aligned GigaPose, corrected EPnP, X, and consistency error."""

    raw_value = row.get("T_gigapose_cam_obj")
    epnp_value = row.get("T_epnp_obj")
    if not raw_value or not epnp_value:
        raise ValueError(
            "Candidate row must contain T_gigapose_cam_obj and T_epnp_obj."
        )

    T_raw = base.text_to_matrix(raw_value)
    T_aligned = aligned_pose_for_row(row)
    T_epnp = base.text_to_matrix(epnp_value)

    if frame_transform_side == "right":
        X = np.linalg.inv(T_raw) @ T_aligned
        reconstructed = T_raw @ X
    elif frame_transform_side == "left":
        X = T_aligned @ np.linalg.inv(T_raw)
        reconstructed = X @ T_raw
    else:
        raise ValueError(f"Unknown frame-transform side: {frame_transform_side!r}")

    consistency = float(np.max(np.abs(reconstructed - T_aligned)))
    return T_aligned, T_epnp, X, consistency


def keep_row(row: dict[str, str], args: argparse.Namespace) -> bool:
    if not base.keep_row(row, args):
        return False
    return bool(row.get("T_gigapose_cam_obj"))


def main() -> None:
    args = parse_args()
    if args.every <= 0:
        raise ValueError("--every must be positive")
    if args.max_images is not None and args.max_images < 0:
        raise ValueError("--max-images must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = args.mesh or args.dataset_dir / "models" / "obj_000001.ply"
    vertices_mm = base.read_ply_vertices(mesh_path)
    if vertices_mm is None:
        raise ValueError(f"Could not read mesh vertices from {mesh_path}")
    mesh_min_mm = vertices_mm.min(axis=0)
    mesh_max_mm = vertices_mm.max(axis=0)
    mesh_aabb_center_mm = 0.5 * (mesh_min_mm + mesh_max_mm)
    raw_object_center_m = np.asarray(args.raw_object_center_m, dtype=float)
    if raw_object_center_m.shape != (3,) or not np.isfinite(
        raw_object_center_m
    ).all():
        raise ValueError("--raw-object-center-m must contain three finite values")
    if args.object_center_mode == "epnp_raw":
        pose_center_in_raw_cad_mm = raw_object_center_m * 1000.0
        center_source = "raw_object_center_m"
    else:
        pose_center_in_raw_cad_mm = mesh_aabb_center_mm
        center_source = "mesh_aabb"
    centered_vertices_mm = (
        vertices_mm - pose_center_in_raw_cad_mm.reshape(1, 3)
    )
    centered_corners_mm = base.bbox_corners_from_vertices(centered_vertices_mm)
    print(
        "Converting raw CAD to centered-pose coordinates using",
        center_source,
        pose_center_in_raw_cad_mm.tolist(),
        "mm; mesh AABB center is",
        mesh_aabb_center_mm.tolist(),
        "mm",
    )

    frame_map = base.load_frame_map(args.dataset_dir)
    rows = base.read_rows(args.candidate_csv)
    input_row_count = len(rows)
    rows = [row for row in rows if keep_row(row, args)]
    rows.sort(key=lambda row: base.row_sort_key(row, args.sort_by))
    rows = rows[:: args.every]

    index_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    model_counts: dict[str, int] = {}
    overlay_limit = len(rows) if args.max_images is None else args.max_images
    overlays_written = 0
    for row_index, row in enumerate(rows):
        scene_id = int(row["scene_id"])
        im_id = int(row["im_id"])
        frame_info = frame_map.get((scene_id, im_id), {})
        T_gigapose, T_epnp, X, consistency = centered_comparison_poses(
            row, args.frame_transform_side
        )
        if consistency > 1e-5:
            raise ValueError(
                f"Frame-transform reconstruction failed for "
                f"{scene_id:06d}_{im_id:06d}: max error={consistency:.6g}"
            )

        diagnostic: dict[str, Any] = {
            "row_index": row_index,
            "scene_id": scene_id,
            "im_id": im_id,
            "match_key": row.get("match_key", ""),
            "score": row.get("score", ""),
            "projection_model": args.projection_model,
            "distortion_model": "",
            "distortion_coefficients": "",
            "metadata_path": row.get("sample_metadata_path", ""),
            "bbox_match_mode": args.bbox_match_mode,
            "bbox_match_index": "",
            "bbox_match_distance_px": "",
            "detection_center_u_px": "",
            "detection_center_v_px": "",
            "corrected_center_u_px": "",
            "corrected_center_v_px": "",
            "corrected_center_minus_detection_du_px": "",
            "corrected_center_minus_detection_dv_px": "",
            "corrected_center_u_metadata_px": "",
            "corrected_center_v_metadata_px": "",
            "metadata_minus_detection_du_px": "",
            "metadata_minus_detection_dv_px": "",
            "corrected_center_u_pinhole_px": "",
            "corrected_center_v_pinhole_px": "",
            "pinhole_minus_detection_du_px": "",
            "pinhole_minus_detection_dv_px": "",
            "metadata_minus_pinhole_du_px": "",
            "metadata_minus_pinhole_dv_px": "",
            "aligned_gigapose_center_u_px": "",
            "aligned_gigapose_center_v_px": "",
            "corrected_center_depth_mm": float(T_epnp[2, 3]),
            "diagnostic_status": "",
            "diagnostic_error": "",
        }
        try:
            metadata_path = metadata_path_from_row(row)
            K, D, distortion_model = load_metadata_camera(metadata_path)
            diagnostic["metadata_path"] = str(metadata_path)
            diagnostic["distortion_model"] = distortion_model
            diagnostic["distortion_coefficients"] = " ".join(
                f"{value:.12g}" for value in D
            )
            model_counts[distortion_model] = model_counts.get(distortion_model, 0) + 1
        except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
            diagnostic["diagnostic_status"] = "camera_metadata_error"
            diagnostic["diagnostic_error"] = str(exc)
            diagnostic_rows.append(diagnostic)
            continue

        corrected_metadata_uv, corrected_metadata_ok = project_pose_center(
            T_epnp, K, D, distortion_model, "metadata"
        )
        corrected_pinhole_uv, corrected_pinhole_ok = project_pose_center(
            T_epnp, K, D, distortion_model, "pinhole"
        )
        active_corrected_uv = (
            corrected_metadata_uv
            if args.projection_model == "metadata"
            else corrected_pinhole_uv
        )
        active_corrected_ok = (
            corrected_metadata_ok
            if args.projection_model == "metadata"
            else corrected_pinhole_ok
        )
        aligned_uv, aligned_ok = project_pose_center(
            T_gigapose, K, D, distortion_model, args.projection_model
        )
        if corrected_metadata_ok and corrected_metadata_uv is not None:
            diagnostic["corrected_center_u_metadata_px"] = float(
                corrected_metadata_uv[0]
            )
            diagnostic["corrected_center_v_metadata_px"] = float(
                corrected_metadata_uv[1]
            )
        if corrected_pinhole_ok and corrected_pinhole_uv is not None:
            diagnostic["corrected_center_u_pinhole_px"] = float(corrected_pinhole_uv[0])
            diagnostic["corrected_center_v_pinhole_px"] = float(corrected_pinhole_uv[1])
        if (
            corrected_metadata_ok
            and corrected_metadata_uv is not None
            and corrected_pinhole_ok
            and corrected_pinhole_uv is not None
        ):
            distortion_shift = corrected_metadata_uv - corrected_pinhole_uv
            diagnostic["metadata_minus_pinhole_du_px"] = float(distortion_shift[0])
            diagnostic["metadata_minus_pinhole_dv_px"] = float(distortion_shift[1])
        if active_corrected_ok and active_corrected_uv is not None:
            diagnostic["corrected_center_u_px"] = float(active_corrected_uv[0])
            diagnostic["corrected_center_v_px"] = float(active_corrected_uv[1])
        if aligned_ok and aligned_uv is not None:
            diagnostic["aligned_gigapose_center_u_px"] = float(aligned_uv[0])
            diagnostic["aligned_gigapose_center_v_px"] = float(aligned_uv[1])

        instances = frame_info.get("instances") or []
        bbox, bbox_idx, bbox_dist = choose_detection_bbox(
            instances,
            row,
            T_gigapose,
            T_epnp,
            K,
            D,
            distortion_model,
            args.projection_model,
            args.bbox_match_mode,
        )
        diagnostic["bbox_match_index"] = "" if bbox_idx is None else bbox_idx
        diagnostic["bbox_match_distance_px"] = (
            "" if bbox_dist is None else float(bbox_dist)
        )
        if bbox is None:
            diagnostic["diagnostic_status"] = "no_detection_bbox"
        elif not active_corrected_ok or active_corrected_uv is None:
            diagnostic["diagnostic_status"] = "invalid_corrected_pose_center"
        else:
            detection_uv = base.bbox_center(bbox)
            active_delta = active_corrected_uv - detection_uv
            diagnostic["detection_center_u_px"] = float(detection_uv[0])
            diagnostic["detection_center_v_px"] = float(detection_uv[1])
            diagnostic["corrected_center_minus_detection_du_px"] = float(
                active_delta[0]
            )
            diagnostic["corrected_center_minus_detection_dv_px"] = float(
                active_delta[1]
            )
            if corrected_metadata_ok and corrected_metadata_uv is not None:
                metadata_delta = corrected_metadata_uv - detection_uv
                diagnostic["metadata_minus_detection_du_px"] = float(metadata_delta[0])
                diagnostic["metadata_minus_detection_dv_px"] = float(metadata_delta[1])
            if corrected_pinhole_ok and corrected_pinhole_uv is not None:
                pinhole_delta = corrected_pinhole_uv - detection_uv
                diagnostic["pinhole_minus_detection_du_px"] = float(pinhole_delta[0])
                diagnostic["pinhole_minus_detection_dv_px"] = float(pinhole_delta[1])
            diagnostic["diagnostic_status"] = "ok"
        diagnostic_rows.append(diagnostic)

        if overlays_written >= overlay_limit:
            continue
        image = base.load_image(
            args.dataset_dir, args.split, scene_id, im_id, frame_info
        )
        draw = ImageDraw.Draw(image)
        delta_label = ""
        if diagnostic["diagnostic_status"] == "ok":
            delta_label = (
                " | corrected-detection "
                f"du={float(diagnostic['corrected_center_minus_detection_du_px']):+.1f}px "
                f"dv={float(diagnostic['corrected_center_minus_detection_dv_px']):+.1f}px"
            )
        draw_projected_box(
            draw,
            centered_corners_mm,
            T_epnp,
            K,
            D,
            distortion_model,
            args.projection_model,
            base.EPNP_COLOR,
            f"EPnPv2 corrected ({distortion_model}/{args.projection_model}){delta_label}",
            0,
        )
        draw_projected_box(
            draw,
            centered_corners_mm,
            T_gigapose,
            K,
            D,
            distortion_model,
            args.projection_model,
            base.GIGAPOSE_COLOR,
            (
                f"GigaPose aligned: t={float(row['translation_error_mm']):.0f}mm "
                f"R={float(row['rotation_error_deg']):.1f}deg "
                f"score={float(row['score']):.3f}"
            ),
            26,
        )
        if args.draw_mask_bbox and bbox:
            label = "GSAM/detection bbox"
            if bbox_idx is not None:
                label += f" #{bbox_idx}"
            if bbox_dist is not None:
                label += f" d={bbox_dist:.0f}px"
            base.draw_bbox(draw, bbox, label)

        output_path = args.output_dir / (
            f"{scene_id:06d}_{im_id:06d}_epnp"
            f"{int(row.get('epnp_record_index') or 0):02d}.jpg"
        )
        image.save(output_path, quality=94)
        overlays_written += 1
        index_rows.append(
            {
                "scene_id": scene_id,
                "im_id": im_id,
                "match_key": row.get("match_key", ""),
                "score": row.get("score", ""),
                "aligned_translation_error_mm": row.get("translation_error_mm", ""),
                "aligned_rotation_error_deg": row.get("rotation_error_deg", ""),
                "frame_transform_side": args.frame_transform_side,
                "frame_transform_translation_norm_mm": float(np.linalg.norm(X[:3, 3])),
                "frame_transform_reconstruction_max_abs": consistency,
                "object_center_mode": args.object_center_mode,
                "pose_center_in_raw_cad_x_mm": float(pose_center_in_raw_cad_mm[0]),
                "pose_center_in_raw_cad_y_mm": float(pose_center_in_raw_cad_mm[1]),
                "pose_center_in_raw_cad_z_mm": float(pose_center_in_raw_cad_mm[2]),
                "mesh_aabb_center_x_mm": float(mesh_aabb_center_mm[0]),
                "mesh_aabb_center_y_mm": float(mesh_aabb_center_mm[1]),
                "mesh_aabb_center_z_mm": float(mesh_aabb_center_mm[2]),
                "epnp_label_path": row.get("epnp_label_path", ""),
                "metadata_path": diagnostic["metadata_path"],
                "projection_model": args.projection_model,
                "distortion_model": distortion_model,
                "bbox_match_mode": args.bbox_match_mode,
                "bbox_match_index": diagnostic["bbox_match_index"],
                "bbox_match_distance_px": (diagnostic["bbox_match_distance_px"]),
                "corrected_center_minus_detection_du_px": diagnostic[
                    "corrected_center_minus_detection_du_px"
                ],
                "corrected_center_minus_detection_dv_px": diagnostic[
                    "corrected_center_minus_detection_dv_px"
                ],
                "output": str(output_path),
            }
        )

    index_path = args.output_dir / "visual_overlay_index.csv"
    fieldnames = [
        "scene_id",
        "im_id",
        "match_key",
        "score",
        "aligned_translation_error_mm",
        "aligned_rotation_error_deg",
        "frame_transform_side",
        "frame_transform_translation_norm_mm",
        "frame_transform_reconstruction_max_abs",
        "object_center_mode",
        "pose_center_in_raw_cad_x_mm",
        "pose_center_in_raw_cad_y_mm",
        "pose_center_in_raw_cad_z_mm",
        "mesh_aabb_center_x_mm",
        "mesh_aabb_center_y_mm",
        "mesh_aabb_center_z_mm",
        "epnp_label_path",
        "metadata_path",
        "projection_model",
        "distortion_model",
        "bbox_match_mode",
        "bbox_match_index",
        "bbox_match_distance_px",
        "corrected_center_minus_detection_du_px",
        "corrected_center_minus_detection_dv_px",
        "output",
    ]
    with index_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    diagnostic_path = args.output_dir / "corrected_center_offsets.csv"
    diagnostic_fieldnames = [
        "row_index",
        "scene_id",
        "im_id",
        "match_key",
        "score",
        "projection_model",
        "distortion_model",
        "distortion_coefficients",
        "metadata_path",
        "bbox_match_mode",
        "bbox_match_index",
        "bbox_match_distance_px",
        "detection_center_u_px",
        "detection_center_v_px",
        "corrected_center_u_px",
        "corrected_center_v_px",
        "corrected_center_minus_detection_du_px",
        "corrected_center_minus_detection_dv_px",
        "corrected_center_u_metadata_px",
        "corrected_center_v_metadata_px",
        "metadata_minus_detection_du_px",
        "metadata_minus_detection_dv_px",
        "corrected_center_u_pinhole_px",
        "corrected_center_v_pinhole_px",
        "pinhole_minus_detection_du_px",
        "pinhole_minus_detection_dv_px",
        "metadata_minus_pinhole_du_px",
        "metadata_minus_pinhole_dv_px",
        "aligned_gigapose_center_u_px",
        "aligned_gigapose_center_v_px",
        "corrected_center_depth_mm",
        "diagnostic_status",
        "diagnostic_error",
    ]
    with diagnostic_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=diagnostic_fieldnames)
        writer.writeheader()
        writer.writerows(diagnostic_rows)

    valid_diagnostics = [
        item for item in diagnostic_rows if item["diagnostic_status"] == "ok"
    ]

    def values(field: str) -> list[float]:
        return [float(item[field]) for item in valid_diagnostics if item[field] != ""]

    active_du = values("corrected_center_minus_detection_du_px")
    active_dv = values("corrected_center_minus_detection_dv_px")
    metadata_du = values("metadata_minus_detection_du_px")
    metadata_dv = values("metadata_minus_detection_dv_px")
    pinhole_du = values("pinhole_minus_detection_du_px")
    pinhole_dv = values("pinhole_minus_detection_dv_px")
    distortion_du = values("metadata_minus_pinhole_du_px")
    distortion_dv = values("metadata_minus_pinhole_dv_px")
    active_dv_array = np.asarray(active_dv, dtype=float)
    active_dv_median = (
        float(np.median(active_dv_array)) if active_dv_array.size else None
    )
    status_counts: dict[str, int] = {}
    for item in diagnostic_rows:
        status = str(item["diagnostic_status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "candidate_csv": str(args.candidate_csv),
        "projection_model": args.projection_model,
        "object_center_mode": args.object_center_mode,
        "pose_center_in_raw_cad_mm": pose_center_in_raw_cad_mm.tolist(),
        "mesh_aabb_center_mm": mesh_aabb_center_mm.tolist(),
        "sign_convention": (
            "delta = corrected_pose_center - detection_bbox_center; positive "
            "delta_v means the corrected pose is lower in the image"
        ),
        "input_rows": input_row_count,
        "rows_after_filters_and_sampling": len(rows),
        "overlays_written": len(index_rows),
        "valid_center_offsets": len(valid_diagnostics),
        "distortion_model_counts": model_counts,
        "diagnostic_status_counts": status_counts,
        "active_delta_u_px": finite_summary(active_du),
        "active_delta_v_px": finite_summary(active_dv),
        "metadata_distorted_delta_u_px": finite_summary(metadata_du),
        "metadata_distorted_delta_v_px": finite_summary(metadata_dv),
        "pinhole_delta_u_px": finite_summary(pinhole_du),
        "pinhole_delta_v_px": finite_summary(pinhole_dv),
        "metadata_minus_pinhole_center_u_px": finite_summary(distortion_du),
        "metadata_minus_pinhole_center_v_px": finite_summary(distortion_dv),
        "fraction_corrected_center_below_detection": (
            float(np.mean(active_dv_array > 0.0)) if active_dv_array.size else None
        ),
        "median_downward_shift_remains": (
            bool(active_dv_median > 0.0) if active_dv_median is not None else None
        ),
        "median_downward_shift_px": active_dv_median,
    }
    summary_path = args.output_dir / "corrected_center_offset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(
        f"Wrote {len(index_rows)} centered-convention EPnPv2/GigaPose overlays to "
        f"{args.output_dir}; wrote {len(diagnostic_rows)} center diagnostics "
        f"({len(valid_diagnostics)} valid) from {input_row_count} input rows"
    )
    if active_dv_median is None:
        print(
            "No valid corrected-pose/detection center pairs were available; see "
            f"{diagnostic_path} for per-row status."
        )
    else:
        direction = "below" if active_dv_median > 0.0 else "above or aligned with"
        print(
            f"Median corrected-center minus detection-center: "
            f"du={np.median(np.asarray(active_du)):+.3f}px, "
            f"dv={active_dv_median:+.3f}px; corrected center remains {direction} "
            "the detection center."
        )
    print(f"Wrote {diagnostic_path} and {summary_path}")


if __name__ == "__main__":
    main()
