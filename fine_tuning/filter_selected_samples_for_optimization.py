"""Filter selected real-label samples before extrinsic optimization.

This script is intentionally separate from the optimizer.  It takes a
``selected_samples.csv`` file produced by ``select_real_label_candidates.py`` or
``combine_selected_samples.py`` and writes a cleaner CSV for calibration.

The main checks are:

- metadata quality, e.g. reject ``ground_truth_pose_missing: true``;
- label source, e.g. keep only ``export_pose_source=ground_truth_pose``;
- GigaPose-vs-map projected center agreement;
- GigaPose-vs-map projected depth agreement;
- optional detector bbox center agreement.

For stereo cameras this also supports metadata distortion models.  Currently:

- ``plumb_bob``: radial/tangential distortion with k1, k2, p1, p2, optional k3;
- ``equidistant``: OpenCV-style fisheye/equidistant with k1..k4;
- anything else falls back to pinhole.

Example:

    python -m fine_tuning.filter_selected_samples_for_optimization \
      --input gigaPose_datasets/results/real_world_data/combined_stereo_left_selected_samples.csv \
      --output-dir gigaPose_datasets/results/real_world_data/stereo_left_filtered \
      --map-z-mode session_lidar_offset \
      --projection-model metadata \
      --reject-ground-truth-missing \
      --allowed-export-pose-source ground_truth_pose \
      --max-depth-diff-m 10 \
      --max-relative-depth-diff 0.20 \
      --max-center-diff-px 80 \
      --max-bbox-center-diff-px 120
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="selected_samples.csv to filter.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clean-name", default="selected_samples_clean.csv")
    parser.add_argument("--rejected-name", default="rejected_samples.csv")
    parser.add_argument(
        "--metadata-path-field",
        default="sample_metadata_path",
        help=(
            "Optional CSV field containing metadata YAML path. If missing, infer "
            "metadata/sample_<timestamp>_<instance>.yaml from epnp_label_path."
        ),
    )
    parser.add_argument(
        "--map-z-mode",
        choices=("raw", "metadata_lidar", "ground_truth_pose", "session_lidar_offset", "session_lidar_affine"),
        default="session_lidar_offset",
        help=(
            "How to adjust T_map_object_raw z before projecting through metadata extrinsics. "
            "session_lidar_offset preserves EPnP z variation but shifts each session to "
            "the metadata lidar z convention. session_lidar_affine uses "
            "median_lidar_z + session_z_scale * (label_z - median_label_z)."
        ),
    )
    parser.add_argument(
        "--session-z-scale",
        type=float,
        default=1.0,
        help=(
            "Scale for EPnP hill/downhill z variation when using "
            "--map-z-mode session_lidar_affine."
        ),
    )
    parser.add_argument(
        "--projection-model",
        choices=("pinhole", "metadata"),
        default="metadata",
        help="Use metadata distortion_model when projecting, or force ideal pinhole.",
    )
    parser.add_argument(
        "--reject-ground-truth-missing",
        action="store_true",
        help="Reject rows whose metadata has ground_truth_pose_missing: true.",
    )
    parser.add_argument(
        "--allowed-export-pose-source",
        action="append",
        default=None,
        help=(
            "Keep only metadata export_pose_source values in this list. Repeat for "
            "multiple values. If omitted, all sources are allowed."
        ),
    )
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--max-selected-translation-error-mm", type=float, default=None)
    parser.add_argument("--max-selected-rotation-error-deg", type=float, default=None)
    parser.add_argument(
        "--max-depth-diff-m",
        type=float,
        default=None,
        help="Reject when abs(map_projected_depth - gigapose_depth) exceeds this many meters.",
    )
    parser.add_argument(
        "--max-relative-depth-diff",
        type=float,
        default=None,
        help="Reject when abs(depth diff) / GigaPose depth exceeds this ratio.",
    )
    parser.add_argument(
        "--max-center-diff-px",
        type=float,
        default=None,
        help="Reject when map-projected center and GigaPose center differ by more than this many pixels.",
    )
    parser.add_argument(
        "--max-bbox-center-diff-px",
        type=float,
        default=None,
        help="Reject when map-projected center and metadata yolo_bbox center differ by more than this many pixels.",
    )
    parser.add_argument(
        "--dedupe-by",
        choices=("none", "epnp_label_path", "match_key_epnp"),
        default="match_key_epnp",
        help="Optional deduplication after filtering.",
    )
    parser.add_argument(
        "--keep",
        choices=("first", "lowest-diagnostics-error", "highest-score"),
        default="lowest-diagnostics-error",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["filter_status"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("This script needs PyYAML: pip install pyyaml or conda install pyyaml") from exc
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Metadata YAML is not a mapping: {path}")
    return data


def text_to_matrix(value: str) -> np.ndarray:
    vals = np.fromstring(str(value).strip().strip("[]").replace(";", " "), sep=" ")
    if vals.size != 16:
        raise ValueError(f"Expected 16 matrix values, got {vals.size}")
    return vals.reshape(4, 4)


def normalize_translation(t: np.ndarray, unit: str) -> np.ndarray:
    t = np.asarray(t, dtype=float).reshape(3)
    if unit == "m":
        return t * 1000.0
    if unit == "mm":
        return t
    return t * 1000.0 if np.nanmedian(np.abs(t)) < 1000.0 else t


def matrix_3x4_or_4x4(value: Any, unit: str = "auto") -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    flat = arr.reshape(-1)
    if flat.size == 16:
        T = flat.reshape(4, 4).copy()
    elif flat.size == 12:
        T = np.eye(4, dtype=float)
        T[:3, :] = flat.reshape(3, 4)
    else:
        raise ValueError(f"Expected 12 or 16 values, got {flat.size}")
    T[:3, 3] = normalize_translation(T[:3, 3], unit)
    return T


def metadata_path_from_row(row: dict[str, str], field: str) -> Path:
    value = row.get(field) or row.get("metadata_path")
    if value and Path(value).is_file():
        return Path(value)
    label = row.get("epnp_label_path", "")
    if not label:
        raise ValueError("No epnp_label_path to infer metadata path.")
    label_path = Path(label)
    match = re.match(r"(.+)_([0-9]+)$", label_path.stem)
    if not match:
        raise ValueError(f"Cannot infer metadata sample name from {label_path.name}")
    timestamp, obj_index = match.groups()
    return label_path.parent.parent / "metadata" / f"sample_{timestamp}_{obj_index}.yaml"


def metadata_session_key(metadata_path: str | Path) -> str:
    path = Path(metadata_path)
    if path.parent.name == "metadata":
        return str(path.parent.parent)
    return str(path.parent)


def load_map_pose_from_label(row: dict[str, str]) -> tuple[np.ndarray, dict[str, Any]]:
    label_path = Path(row["epnp_label_path"])
    data = json.loads(label_path.read_text())
    if "T_map_object_raw" not in data:
        raise KeyError(f"T_map_object_raw missing from {label_path}")
    return matrix_3x4_or_4x4(data["T_map_object_raw"], unit="auto"), data


def label_map_z_m(label_data: dict[str, Any], T_map_obj: np.ndarray) -> float:
    if isinstance(label_data.get("map_pose"), dict):
        return float(label_data["map_pose"].get("position", {}).get("z", T_map_obj[2, 3] * 0.001))
    return float(T_map_obj[2, 3] * 0.001)


def compute_session_lidar_z_stats(rows: list[dict[str, str]], metadata_path_field: str) -> dict[str, dict[str, float]]:
    offsets: dict[str, list[float]] = defaultdict(list)
    label_zs: dict[str, list[float]] = defaultdict(list)
    lidar_zs: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        try:
            metadata_path = metadata_path_from_row(row, metadata_path_field)
            metadata = load_yaml(metadata_path)
            T_map_obj, label_data = load_map_pose_from_label(row)
            lidar_z_m = float(np.asarray(metadata["t_map_lidar"], dtype=float).reshape(4, 4)[2, 3])
            z_m = label_map_z_m(label_data, T_map_obj)
            session = metadata_session_key(metadata_path)
            offsets[session].append(lidar_z_m - z_m)
            label_zs[session].append(z_m)
            lidar_zs[session].append(lidar_z_m)
        except Exception:
            continue
    out: dict[str, dict[str, float]] = {}
    for key in set(offsets) | set(label_zs) | set(lidar_zs):
        out[key] = {
            "offset_m": float(np.median(np.asarray(offsets.get(key, [0.0]), dtype=float))),
            "label_z_median": float(np.median(np.asarray(label_zs.get(key, [0.0]), dtype=float))),
            "lidar_z_median": float(np.median(np.asarray(lidar_zs.get(key, [0.0]), dtype=float))),
        }
    return out


def apply_map_z_mode(
    T_map_obj: np.ndarray,
    label_data: dict[str, Any],
    metadata: dict[str, Any],
    mode: str,
    session_z_offset_m: float | None,
    session_label_z_median: float | None,
    session_lidar_z_median: float | None,
    session_z_scale: float,
) -> np.ndarray:
    if mode == "raw":
        return T_map_obj
    out = T_map_obj.copy()
    current_z_m = label_map_z_m(label_data, T_map_obj)
    if mode == "metadata_lidar":
        target_z_m = float(np.asarray(metadata["t_map_lidar"], dtype=float).reshape(4, 4)[2, 3])
    elif mode == "ground_truth_pose":
        target_z_m = float(metadata["ground_truth_pose"]["position"]["z"])
    elif mode == "session_lidar_offset":
        if session_z_offset_m is None:
            raise ValueError("session_lidar_offset requires a computed session offset.")
        target_z_m = current_z_m + session_z_offset_m
    elif mode == "session_lidar_affine":
        if session_label_z_median is None or session_lidar_z_median is None:
            raise ValueError("session_lidar_affine requires computed session medians.")
        target_z_m = session_lidar_z_median + session_z_scale * (current_z_m - session_label_z_median)
    else:
        raise ValueError(f"Unknown map z mode: {mode}")
    out[2, 3] += (target_z_m - current_z_m) * 1000.0
    return out


def load_K_D_model(metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    intrinsics = metadata.get("camera_intrinsics", {})
    if isinstance(intrinsics, dict) and "k" in intrinsics:
        K = np.asarray(intrinsics["k"], dtype=float).reshape(3, 3)
        D = np.asarray(intrinsics.get("d", []), dtype=float).reshape(-1)
        model = str(intrinsics.get("distortion_model", metadata.get("distortion_model", "pinhole")))
        return K, D, model
    if "K" in metadata:
        K = np.asarray(metadata["K"], dtype=float).reshape(3, 3)
        D = np.asarray(metadata.get("D", []), dtype=float).reshape(-1)
        model = str(metadata.get("distortion_model", "pinhole"))
        return K, D, model
    raise KeyError("No camera intrinsics found in metadata.")


def project_points(
    points_obj_mm: np.ndarray,
    T_cam_obj: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    distortion_model: str,
    projection_model: str,
) -> tuple[np.ndarray, np.ndarray]:
    points_cam = (T_cam_obj[:3, :3] @ points_obj_mm.T).T + T_cam_obj[:3, 3].reshape(1, 3)
    z = points_cam[:, 2]
    valid = z > 1e-6
    x = points_cam[:, 0] / np.maximum(z, 1e-9)
    y = points_cam[:, 1] / np.maximum(z, 1e-9)

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


def project_origin(
    T_cam_obj: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    distortion_model: str,
    projection_model: str,
) -> tuple[np.ndarray, bool, float]:
    uv, valid = project_points(
        np.zeros((1, 3), dtype=float),
        T_cam_obj,
        K,
        D,
        distortion_model,
        projection_model,
    )
    return uv[0], bool(valid[0]), float(T_cam_obj[2, 3])


def bbox_center(metadata: dict[str, Any]) -> tuple[np.ndarray, bool]:
    bbox = metadata.get("yolo_bbox") or metadata.get("detector_bbox_xywh")
    try:
        if isinstance(bbox, dict):
            x, y, w, h = [float(bbox[k]) for k in ("x", "y", "width", "height")]
        elif isinstance(bbox, list) and len(bbox) >= 4:
            x, y, w, h = [float(v) for v in bbox[:4]]
        else:
            return np.zeros(2), False
        return np.asarray([x + 0.5 * w, y + 0.5 * h], dtype=float), True
    except Exception:
        return np.zeros(2), False


def float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def fail_if(condition: bool, reasons: list[str], reason: str) -> None:
    if condition:
        reasons.append(reason)


def diagnostic_error(row: dict[str, Any]) -> float:
    center = float_or_nan(row.get("map_vs_gigapose_center_diff_px"))
    depth = float_or_nan(row.get("map_vs_gigapose_depth_diff_m"))
    selected_t = float_or_nan(row.get("translation_error_mm"))
    selected_r = float_or_nan(row.get("rotation_error_deg"))
    score = float_or_nan(row.get("score"))
    total = 0.0
    if math.isfinite(center):
        total += center / 100.0
    if math.isfinite(depth):
        total += abs(depth) / 10.0
    if math.isfinite(selected_t):
        total += selected_t / 2000.0
    if math.isfinite(selected_r):
        total += selected_r / 30.0
    if math.isfinite(score):
        total -= score * 0.1
    return total


def dedupe_key(row: dict[str, Any], mode: str) -> tuple[Any, ...] | None:
    if mode == "none":
        return None
    if mode == "epnp_label_path":
        return (row.get("epnp_label_path", ""), row.get("epnp_record_index", ""))
    if mode == "match_key_epnp":
        return (row.get("match_key", ""), row.get("epnp_label_path", ""), row.get("epnp_record_index", ""))
    raise ValueError(f"Unknown dedupe mode: {mode}")


def prefer(new: dict[str, Any], old: dict[str, Any], mode: str) -> bool:
    if mode == "first":
        return False
    if mode == "highest-score":
        return float_or_nan(new.get("score")) > float_or_nan(old.get("score"))
    if mode == "lowest-diagnostics-error":
        return diagnostic_error(new) < diagnostic_error(old)
    raise ValueError(f"Unknown keep mode: {mode}")


def dedupe(rows: list[dict[str, Any]], mode: str, keep: str) -> list[dict[str, Any]]:
    if mode == "none":
        return rows
    kept: dict[tuple[Any, ...], dict[str, Any]] = {}
    passthrough = []
    for row in rows:
        key = dedupe_key(row, mode)
        if key is None or not any(str(part) for part in key):
            passthrough.append(row)
            continue
        if key not in kept or prefer(row, kept[key], keep):
            kept[key] = row
    return passthrough + list(kept.values())


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.input)
    session_z_stats = (
        compute_session_lidar_z_stats(rows, args.metadata_path_field)
        if args.map_z_mode in ("session_lidar_offset", "session_lidar_affine")
        else {}
    )

    enriched = []
    clean = []
    rejected = []
    reason_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    distortion_counter: Counter[str] = Counter()

    for row in rows:
        out = dict(row)
        reasons: list[str] = []
        try:
            metadata_path = metadata_path_from_row(row, args.metadata_path_field)
            metadata = load_yaml(metadata_path)
            T_map_obj_raw, label_data = load_map_pose_from_label(row)
            session_key = metadata_session_key(metadata_path)
            z_stats = session_z_stats.get(session_key, {})
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
            T_map_lidar = matrix_3x4_or_4x4(metadata["t_map_lidar"], unit="m")
            T_lidar_cam = matrix_3x4_or_4x4(metadata["t_lidar_camera_prior"], unit="m")
            T_map_cam = T_map_lidar @ T_lidar_cam
            T_cam_obj_map = np.linalg.inv(T_map_cam) @ T_map_obj
            T_giga = text_to_matrix(row["T_gigapose_cam_obj"])
            K, D, distortion_model = load_K_D_model(metadata)
            map_uv, map_valid, map_depth_mm = project_origin(
                T_cam_obj_map, K, D, distortion_model, args.projection_model
            )
            giga_uv, giga_valid, giga_depth_mm = project_origin(
                T_giga, K, D, distortion_model, args.projection_model
            )
            bbox_uv, bbox_valid = bbox_center(metadata)

            center_diff = float(np.linalg.norm(map_uv - giga_uv)) if map_valid and giga_valid else float("nan")
            depth_diff_m = (map_depth_mm - giga_depth_mm) * 0.001
            relative_depth_diff = abs(depth_diff_m) / max(abs(giga_depth_mm) * 0.001, 1e-9)
            bbox_center_diff = float(np.linalg.norm(map_uv - bbox_uv)) if map_valid and bbox_valid else float("nan")
            source = str(metadata.get("export_pose_source", ""))
            source_counter[source] += 1
            distortion_counter[str(distortion_model)] += 1

            out.update(
                {
                    "sample_metadata_path": str(metadata_path),
                    "metadata_session_key": session_key,
                    "metadata_export_pose_source": source,
                    "metadata_ground_truth_pose_missing": str(bool(metadata.get("ground_truth_pose_missing", False))),
                    "metadata_distortion_model": str(distortion_model),
                    "map_z_mode": args.map_z_mode,
                    "session_lidar_z_offset_m": z_stats.get("offset_m", ""),
                    "session_label_z_median": z_stats.get("label_z_median", ""),
                    "session_lidar_z_median": z_stats.get("lidar_z_median", ""),
                    "session_z_scale": args.session_z_scale,
                    "raw_map_object_z_m": float(T_map_obj_raw[2, 3] * 0.001),
                    "adjusted_map_object_z_m": float(T_map_obj[2, 3] * 0.001),
                    "metadata_lidar_z_m": float(T_map_lidar[2, 3] * 0.001),
                    "map_center_u": float(map_uv[0]),
                    "map_center_v": float(map_uv[1]),
                    "map_center_valid": str(bool(map_valid)),
                    "map_depth_m": float(map_depth_mm * 0.001),
                    "gigapose_center_u": float(giga_uv[0]),
                    "gigapose_center_v": float(giga_uv[1]),
                    "gigapose_center_valid": str(bool(giga_valid)),
                    "gigapose_depth_m": float(giga_depth_mm * 0.001),
                    "map_vs_gigapose_center_diff_px": center_diff,
                    "map_vs_gigapose_depth_diff_m": depth_diff_m,
                    "map_vs_gigapose_relative_depth_diff": relative_depth_diff,
                    "bbox_center_u": float(bbox_uv[0]) if bbox_valid else "",
                    "bbox_center_v": float(bbox_uv[1]) if bbox_valid else "",
                    "map_vs_bbox_center_diff_px": bbox_center_diff,
                }
            )

            fail_if(not map_valid, reasons, "map_projection_invalid")
            fail_if(not giga_valid, reasons, "gigapose_projection_invalid")
            fail_if(args.reject_ground_truth_missing and bool(metadata.get("ground_truth_pose_missing", False)), reasons, "ground_truth_pose_missing")
            fail_if(
                args.allowed_export_pose_source is not None and source not in set(args.allowed_export_pose_source),
                reasons,
                f"export_pose_source_not_allowed:{source}",
            )
            if args.min_score is not None:
                fail_if(float_or_nan(row.get("score")) < args.min_score, reasons, "score_below_threshold")
            if args.max_selected_translation_error_mm is not None:
                fail_if(
                    float_or_nan(row.get("translation_error_mm")) > args.max_selected_translation_error_mm,
                    reasons,
                    "selected_translation_error_too_high",
                )
            if args.max_selected_rotation_error_deg is not None:
                fail_if(
                    float_or_nan(row.get("rotation_error_deg")) > args.max_selected_rotation_error_deg,
                    reasons,
                    "selected_rotation_error_too_high",
                )
            if args.max_depth_diff_m is not None:
                fail_if(abs(depth_diff_m) > args.max_depth_diff_m, reasons, "depth_diff_too_high")
            if args.max_relative_depth_diff is not None:
                fail_if(relative_depth_diff > args.max_relative_depth_diff, reasons, "relative_depth_diff_too_high")
            if args.max_center_diff_px is not None:
                fail_if(center_diff > args.max_center_diff_px, reasons, "center_diff_too_high")
            if args.max_bbox_center_diff_px is not None and math.isfinite(bbox_center_diff):
                fail_if(bbox_center_diff > args.max_bbox_center_diff_px, reasons, "bbox_center_diff_too_high")

        except Exception as exc:
            reasons.append(f"exception:{type(exc).__name__}")
            out["filter_exception"] = str(exc)

        out["filter_reasons"] = ";".join(reasons)
        out["filter_status"] = "rejected" if reasons else "clean"
        enriched.append(out)
        if reasons:
            rejected.append(out)
            reason_counter.update(reasons)
        else:
            clean.append(out)

    clean_before_dedupe = len(clean)
    clean = dedupe(clean, args.dedupe_by, args.keep)

    clean.sort(key=lambda r: diagnostic_error(r))
    rejected.sort(key=lambda r: (str(r.get("filter_reasons", "")), diagnostic_error(r)))

    clean_path = args.output_dir / args.clean_name
    rejected_path = args.output_dir / args.rejected_name
    diagnostics_path = args.output_dir / "all_sample_diagnostics.csv"
    report_path = args.output_dir / "filter_report.json"

    write_csv(clean_path, clean)
    write_csv(rejected_path, rejected)
    write_csv(diagnostics_path, enriched)

    def finite_values(key: str, source_rows: list[dict[str, Any]]) -> list[float]:
        vals = []
        for row in source_rows:
            v = float_or_nan(row.get(key))
            if math.isfinite(v):
                vals.append(v)
        return vals

    def stats(key: str, source_rows: list[dict[str, Any]]) -> dict[str, float]:
        vals = finite_values(key, source_rows)
        if not vals:
            return {}
        arr = np.asarray(vals, dtype=float)
        return {
            f"{key}_mean": float(np.mean(arr)),
            f"{key}_median": float(np.median(arr)),
            f"{key}_p90": float(np.percentile(arr, 90)),
        }

    report = {
        "input": str(args.input),
        "input_rows": len(rows),
        "clean_rows_before_dedupe": clean_before_dedupe,
        "clean_rows": len(clean),
        "rejected_rows": len(rejected),
        "clean_csv": str(clean_path),
        "rejected_csv": str(rejected_path),
        "all_diagnostics_csv": str(diagnostics_path),
        "map_z_mode": args.map_z_mode,
        "projection_model": args.projection_model,
        "filters": {
            "reject_ground_truth_missing": args.reject_ground_truth_missing,
            "allowed_export_pose_source": args.allowed_export_pose_source,
            "min_score": args.min_score,
            "max_selected_translation_error_mm": args.max_selected_translation_error_mm,
            "max_selected_rotation_error_deg": args.max_selected_rotation_error_deg,
            "max_depth_diff_m": args.max_depth_diff_m,
            "max_relative_depth_diff": args.max_relative_depth_diff,
            "max_center_diff_px": args.max_center_diff_px,
            "max_bbox_center_diff_px": args.max_bbox_center_diff_px,
        },
        "session_lidar_z_stats": session_z_stats,
        "rejection_reasons": dict(reason_counter),
        "metadata_export_pose_source_counts": dict(source_counter),
        "metadata_distortion_model_counts": dict(distortion_counter),
        "clean_stats": {
            **stats("map_vs_gigapose_center_diff_px", clean),
            **stats("map_vs_gigapose_depth_diff_m", clean),
            **stats("map_vs_gigapose_relative_depth_diff", clean),
            **stats("map_vs_bbox_center_diff_px", clean),
        },
        "all_stats": {
            **stats("map_vs_gigapose_center_diff_px", enriched),
            **stats("map_vs_gigapose_depth_diff_m", enriched),
            **stats("map_vs_gigapose_relative_depth_diff", enriched),
            **stats("map_vs_bbox_center_diff_px", enriched),
        },
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote clean samples to {clean_path}")


if __name__ == "__main__":
    main()
