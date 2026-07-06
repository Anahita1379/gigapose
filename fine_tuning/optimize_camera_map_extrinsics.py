"""Optimize map-to-camera extrinsic correction from selected real labels.

This script uses selected samples produced by:

    python -m fine_tuning.select_real_label_candidates ...

Expected selected CSV columns:

    T_gigapose_cam_obj
    T_epnp_obj
    epnp_label_path

Interpretation:

- ``T_gigapose_cam_obj`` maps object coordinates into the GigaPose camera frame.
- ``T_epnp_obj`` maps object coordinates into the EPnPv2 frame used during
  candidate selection.  For recent real-data runs this is often the EPnPv2
  camera frame, not the map frame.
- The initial extrinsic ``T_map_cam`` maps camera coordinates into the map frame.

For true map-to-camera optimization with a static camera, pass
``--epnp-map-pose-key`` so the script loads the map-frame pose, usually
``T_map_object_raw``, from each selected sample's ``epnp_label_path``.  Then the
optimized model is:

    T_map_object_raw ≈ T_map_cam_optimized @ T_gigapose_cam_obj
    T_map_cam_optimized = exp(delta) @ T_map_cam_initial

For the ARCL/Assetto real folders, the ego vehicle moves, so each frame has its
own ``t_map_lidar`` metadata.  In that case use ``--use-sample-metadata`` and
the optimized model becomes:

    T_map_object_raw ≈ T_map_lidar_i @ (exp(delta) @ T_lidar_camera_prior_i) @ T_gigapose_cam_obj

The correction has a prior. By default translation correction is penalized more
strongly than rotation correction, because the current translation extrinsics
are assumed to be more reliable than the rotational extrinsics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selected-samples",
        type=Path,
        required=True,
        help="selected_samples.csv from select_real_label_candidates.py.",
    )
    parser.add_argument(
        "--initial-extrinsic",
        type=Path,
        default=None,
        help=(
            "JSON containing initial T_map_cam as key T_map_cam, matrix, or T. "
            "Translations are assumed to be in millimeters unless --initial-unit m. "
            "Not required with --use-sample-metadata."
        ),
    )
    parser.add_argument(
        "--initial-unit",
        choices=("mm", "m"),
        default="mm",
        help="Translation unit used in the initial extrinsic JSON.",
    )
    parser.add_argument(
        "--epnp-map-pose-key",
        default=None,
        help=(
            "Optional pose key to load from each selected row's epnp_label_path, "
            "e.g. T_map_object_raw. Use this for map-camera extrinsic optimization. "
            "If omitted, the script uses selected CSV column T_epnp_obj."
        ),
    )
    parser.add_argument(
        "--use-sample-metadata",
        action="store_true",
        help=(
            "Load per-sample metadata YAML and optimize a fixed correction to "
            "t_lidar_camera_prior using each sample's t_map_lidar. This is the "
            "right mode for moving-ego ARCL folders."
        ),
    )
    parser.add_argument(
        "--metadata-path-field",
        default="sample_metadata_path",
        help=(
            "Optional selected CSV field containing metadata YAML path. If absent, "
            "the script derives metadata/sample_<timestamp>_<id>.yaml from epnp_label_path."
        ),
    )
    parser.add_argument(
        "--epnp-map-pose-unit",
        choices=("auto", "m", "mm"),
        default="auto",
        help="Translation unit for --epnp-map-pose-key values.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gigaPose_datasets/results/real_world_extrinsics"),
    )
    parser.add_argument(
        "--translation-sigma-mm",
        type=float,
        default=1000.0,
        help="Pose residual scale for object translation error.",
    )
    parser.add_argument(
        "--rotation-sigma-deg",
        type=float,
        default=10.0,
        help="Pose residual scale for object rotation error.",
    )
    parser.add_argument(
        "--translation-prior-weight",
        type=float,
        default=10.0,
        help=(
            "Prior weight on changing the extrinsic translation. Larger values "
            "keep translation closer to the initial extrinsic."
        ),
    )
    parser.add_argument(
        "--rotation-prior-weight",
        type=float,
        default=1.0,
        help=(
            "Prior weight on changing the extrinsic rotation. Lower than "
            "translation-prior-weight if rotation is expected to be less reliable."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for debugging.",
    )
    parser.add_argument(
        "--robust-loss",
        choices=("linear", "soft_l1", "huber", "cauchy", "arctan"),
        default="soft_l1",
    )
    return parser.parse_args()


def text_to_matrix(value: str) -> np.ndarray:
    values = np.fromstring(str(value).strip().strip("[]").replace(";", " "), sep=" ")
    if values.size != 16:
        raise ValueError(f"Expected 16 values for 4x4 matrix, got {values.size}")
    return values.reshape(4, 4)


def normalize_translation(t: np.ndarray, unit: str) -> np.ndarray:
    t = np.asarray(t, dtype=float).reshape(3)
    if unit == "m":
        return t * 1000.0
    if unit == "mm":
        return t
    return t * 1000.0 if np.nanmedian(np.abs(t)) < 1000.0 else t


def matrix_3x4_or_4x4_to_transform(value: Any, unit: str) -> np.ndarray:
    arr = text_to_matrix(value) if isinstance(value, str) else np.asarray(value, dtype=float)
    flat = arr.reshape(-1)
    if flat.size == 16:
        T = flat.reshape(4, 4).copy()
    elif flat.size == 12:
        T = np.eye(4, dtype=float)
        T[:3, :] = flat.reshape(3, 4)
    else:
        raise ValueError(f"Expected 12 or 16 values for a 3x4/4x4 pose matrix, got {flat.size}")
    T[:3, 3] = normalize_translation(T[:3, 3], unit)
    return T


def matrix_to_text(T: np.ndarray) -> str:
    return " ".join(f"{v:.12g}" for v in np.asarray(T, dtype=float).reshape(-1))


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def so3_exp(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    W = skew(w)
    if theta < 1e-12:
        return np.eye(3) + W
    A = math.sin(theta) / theta
    B = (1.0 - math.cos(theta)) / (theta * theta)
    return np.eye(3) + A * W + B * (W @ W)


def so3_log(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(math.acos(cos_theta))
    if theta < 1e-12:
        return np.zeros(3)
    return (
        theta
        / (2.0 * math.sin(theta))
        * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    )


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Small SE(3) update.

    xi = [rx, ry, rz, tx_mm, ty_mm, tz_mm].

    For an extrinsic correction this small-update form is enough and easier to
    reason about than the full SE(3) V matrix. Rotation is in radians,
    translation is in millimeters.
    """
    xi = np.asarray(xi, dtype=float).reshape(6)
    T = np.eye(4, dtype=float)
    T[:3, :3] = so3_exp(xi[:3])
    T[:3, 3] = xi[3:6]
    return T


def rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    delta = R_pred @ R_gt.T
    cos_theta = (np.trace(delta) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


def load_initial_extrinsic(path: Path, unit: str) -> np.ndarray:
    data = json.loads(path.read_text())
    for key in ("T_map_cam", "matrix", "T", "T_world_camera", "T_map_camera"):
        if key in data:
            T = np.asarray(data[key], dtype=float).reshape(4, 4)
            if unit == "m":
                T[:3, 3] *= 1000.0
            return T
    raise ValueError(
        f"{path} must contain one of T_map_cam, matrix, T, T_world_camera, T_map_camera"
    )


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "Reading metadata YAML needs PyYAML. Install it with `pip install pyyaml` "
            "or `conda install pyyaml`."
        ) from exc
    with path.open() as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a YAML mapping")
    return data


def metadata_path_from_row(row: dict[str, Any], field: str) -> Path:
    value = row.get(field)
    if value:
        path = Path(value)
        if path.is_file():
            return path

    label_path_text = row.get("epnp_label_path")
    if not label_path_text:
        raise ValueError("Cannot infer metadata path without epnp_label_path")
    label_path = Path(label_path_text)
    match = re.match(r"(.+)_([0-9]+)$", label_path.stem)
    if not match:
        raise ValueError(f"Cannot infer metadata sample name from {label_path.name}")
    timestamp, obj_index = match.groups()
    metadata_path = label_path.parent.parent / "metadata" / f"sample_{timestamp}_{obj_index}.yaml"
    if metadata_path.is_file():
        return metadata_path
    raise FileNotFoundError(f"Could not find metadata YAML for {label_path}: {metadata_path}")


def matrix_from_yaml_key(data: dict[str, Any], key: str, unit: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"{key} missing from metadata YAML")
    return matrix_3x4_or_4x4_to_transform(data[key], unit)


def load_sample_metadata_transforms(row: dict[str, Any], metadata_path_field: str) -> tuple[np.ndarray, np.ndarray, str]:
    path = metadata_path_from_row(row, metadata_path_field)
    data = load_yaml(path)
    # The real-data metadata stores these in meters.
    T_map_lidar = matrix_from_yaml_key(data, "t_map_lidar", "m")
    T_lidar_camera_prior = matrix_from_yaml_key(data, "t_lidar_camera_prior", "m")
    return T_map_lidar, T_lidar_camera_prior, str(path)


def load_map_pose_from_epnp_label(path: Path, pose_key: str, unit: str) -> np.ndarray:
    data = json.loads(path.read_text())
    if pose_key not in data:
        raise KeyError(f"{pose_key} not found in {path}")
    return matrix_3x4_or_4x4_to_transform(data[pose_key], unit)


def load_selected_samples(
    path: Path,
    max_samples: int | None,
    epnp_map_pose_key: str | None,
    epnp_map_pose_unit: str,
    use_sample_metadata: bool,
    metadata_path_field: str,
) -> list[dict[str, Any]]:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                T_giga = text_to_matrix(row["T_gigapose_cam_obj"])
                if epnp_map_pose_key:
                    label_path = Path(row["epnp_label_path"])
                    T_target = load_map_pose_from_epnp_label(
                        label_path, epnp_map_pose_key, epnp_map_pose_unit
                    )
                else:
                    T_target = text_to_matrix(row["T_epnp_obj"])
                metadata_values = {}
                if use_sample_metadata:
                    T_map_lidar, T_lidar_camera_prior, metadata_path = load_sample_metadata_transforms(
                        row, metadata_path_field
                    )
                    metadata_values = {
                        "T_map_lidar": T_map_lidar,
                        "T_lidar_camera_prior": T_lidar_camera_prior,
                        "metadata_path": metadata_path,
                    }
            except Exception as exc:
                print(f"Skipping malformed selected sample: {exc}")
                continue
            rows.append(
                {
                    **row,
                    "T_gigapose_cam_obj": T_giga,
                    "T_target_obj": T_target,
                    **metadata_values,
                }
            )
            if max_samples is not None and len(rows) >= max_samples:
                break
    if not rows:
        if epnp_map_pose_key:
            raise ValueError(
                f"No valid selected samples found in {path}. Check that each "
                f"epnp_label_path exists and contains {epnp_map_pose_key}."
            )
        raise ValueError(f"No valid selected samples found in {path}")
    return rows


def residual_vector(
    xi: np.ndarray,
    T_map_cam_initial: np.ndarray,
    samples: list[dict[str, Any]],
    translation_sigma_mm: float,
    rotation_sigma_deg: float,
    translation_prior_weight: float,
    rotation_prior_weight: float,
) -> np.ndarray:
    T_delta = se3_exp(xi)
    T_map_cam = T_delta @ T_map_cam_initial
    residuals = []
    rotation_sigma_rad = math.radians(rotation_sigma_deg)

    for sample in samples:
        T_pred = T_map_cam @ sample["T_gigapose_cam_obj"]
        T_gt = sample["T_target_obj"]
        t_res = (T_pred[:3, 3] - T_gt[:3, 3]) / translation_sigma_mm
        r_res = so3_log(T_pred[:3, :3] @ T_gt[:3, :3].T) / rotation_sigma_rad
        residuals.extend(t_res.tolist())
        residuals.extend(r_res.tolist())

    # Priors on changing the extrinsic itself. Translation prior is intentionally
    # stronger by default.
    residuals.extend((xi[:3] * rotation_prior_weight).tolist())
    residuals.extend(((xi[3:6] / translation_sigma_mm) * translation_prior_weight).tolist())
    return np.asarray(residuals, dtype=float)


def residual_vector_sample_metadata(
    xi: np.ndarray,
    samples: list[dict[str, Any]],
    translation_sigma_mm: float,
    rotation_sigma_deg: float,
    translation_prior_weight: float,
    rotation_prior_weight: float,
) -> np.ndarray:
    T_delta = se3_exp(xi)
    residuals = []
    rotation_sigma_rad = math.radians(rotation_sigma_deg)

    for sample in samples:
        T_lidar_camera = T_delta @ sample["T_lidar_camera_prior"]
        T_map_cam = sample["T_map_lidar"] @ T_lidar_camera
        T_pred = T_map_cam @ sample["T_gigapose_cam_obj"]
        T_gt = sample["T_target_obj"]
        t_res = (T_pred[:3, 3] - T_gt[:3, 3]) / translation_sigma_mm
        r_res = so3_log(T_pred[:3, :3] @ T_gt[:3, :3].T) / rotation_sigma_rad
        residuals.extend(t_res.tolist())
        residuals.extend(r_res.tolist())

    residuals.extend((xi[:3] * rotation_prior_weight).tolist())
    residuals.extend(((xi[3:6] / translation_sigma_mm) * translation_prior_weight).tolist())
    return np.asarray(residuals, dtype=float)


def compute_sample_errors(T_map_cam: np.ndarray, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        T_pred = T_map_cam @ sample["T_gigapose_cam_obj"]
        T_gt = sample["T_target_obj"]
        rows.append(
            {
                "match_key": sample.get("match_key", ""),
                "scene_id": sample.get("scene_id", ""),
                "im_id": sample.get("im_id", ""),
                "instance_id": sample.get("instance_id", ""),
                "score": sample.get("score", ""),
                "translation_error_mm": float(np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3])),
                "rotation_error_deg": rotation_error_deg(T_pred[:3, :3], T_gt[:3, :3]),
            }
        )
    return rows


def compute_sample_metadata_errors(T_delta: np.ndarray, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        T_lidar_camera = T_delta @ sample["T_lidar_camera_prior"]
        T_map_cam = sample["T_map_lidar"] @ T_lidar_camera
        T_pred = T_map_cam @ sample["T_gigapose_cam_obj"]
        T_gt = sample["T_target_obj"]
        rows.append(
            {
                "match_key": sample.get("match_key", ""),
                "scene_id": sample.get("scene_id", ""),
                "im_id": sample.get("im_id", ""),
                "instance_id": sample.get("instance_id", ""),
                "score": sample.get("score", ""),
                "metadata_path": sample.get("metadata_path", ""),
                "translation_error_mm": float(np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3])),
                "rotation_error_deg": rotation_error_deg(T_pred[:3, :3], T_gt[:3, :3]),
            }
        )
    return rows


def summarize_errors(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    out = {}
    for key in ("translation_error_mm", "rotation_error_deg"):
        vals = np.asarray([float(row[key]) for row in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        out[f"{prefix}_{key}_mean"] = float(np.mean(vals))
        out[f"{prefix}_{key}_median"] = float(np.median(vals))
        out[f"{prefix}_{key}_p90"] = float(np.percentile(vals, 90))
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise SystemExit(
            "This optimization template needs scipy. Install it in the gigapose "
            "environment with `pip install scipy` or `conda install scipy`."
        ) from exc

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_selected_samples(
        args.selected_samples,
        args.max_samples,
        args.epnp_map_pose_key,
        args.epnp_map_pose_unit,
        args.use_sample_metadata,
        args.metadata_path_field,
    )
    if args.use_sample_metadata:
        T_initial = np.eye(4, dtype=float)
        before_rows = compute_sample_metadata_errors(T_initial, samples)
        residual_fn = residual_vector_sample_metadata
        residual_args = (
            samples,
            args.translation_sigma_mm,
            args.rotation_sigma_deg,
            args.translation_prior_weight,
            args.rotation_prior_weight,
        )
    else:
        if args.initial_extrinsic is None:
            raise SystemExit("--initial-extrinsic is required unless --use-sample-metadata is set.")
        T_initial = load_initial_extrinsic(args.initial_extrinsic, args.initial_unit)
        before_rows = compute_sample_errors(T_initial, samples)
        residual_fn = residual_vector
        residual_args = (
            T_initial,
            samples,
            args.translation_sigma_mm,
            args.rotation_sigma_deg,
            args.translation_prior_weight,
            args.rotation_prior_weight,
        )

    result = least_squares(
        residual_fn,
        x0=np.zeros(6, dtype=float),
        args=residual_args,
        loss=args.robust_loss,
        max_nfev=500,
    )

    xi = result.x
    T_delta = se3_exp(xi)
    T_optimized = T_delta @ T_initial
    if args.use_sample_metadata:
        after_rows = compute_sample_metadata_errors(T_delta, samples)
    else:
        after_rows = compute_sample_errors(T_optimized, samples)

    write_csv(args.output_dir / "errors_before_optimization.csv", before_rows)
    write_csv(args.output_dir / "errors_after_optimization.csv", after_rows)

    extrinsics = {
        "description": "Optimized map-to-camera extrinsic correction from selected GigaPose/EPnPv2 samples.",
        "translation_unit": "mm",
        "target_pose_source": args.epnp_map_pose_key or "selected_csv:T_epnp_obj",
        "optimization_mode": "sample_metadata_lidar_camera_prior" if args.use_sample_metadata else "static_T_map_cam",
        "T_map_cam_initial": T_initial.tolist() if not args.use_sample_metadata else None,
        "T_correction_left_multiply": T_delta.tolist(),
        "T_map_cam_optimized": T_optimized.tolist() if not args.use_sample_metadata else None,
        "T_lidar_camera_correction_left_multiply": T_delta.tolist() if args.use_sample_metadata else None,
        "correction_rotation_rpy_like_vector_rad": xi[:3].tolist(),
        "correction_translation_mm": xi[3:6].tolist(),
        "note": (
            "Sample-metadata mode: apply T_lidar_camera_optimized = "
            "T_lidar_camera_correction_left_multiply @ t_lidar_camera_prior for each frame, "
            "then T_map_cam = t_map_lidar @ T_lidar_camera_optimized."
            if args.use_sample_metadata
            else "Use T_map_cam_optimized as camera-to-map transform if your pipeline expects T_map_cam."
        ),
    }
    (args.output_dir / "optimized_extrinsics.json").write_text(json.dumps(extrinsics, indent=2))

    report = {
        "selected_samples": len(samples),
        "optimizer_success": bool(result.success),
        "optimizer_message": result.message,
        "optimizer_cost": float(result.cost),
        "robust_loss": args.robust_loss,
        "target_pose_source": args.epnp_map_pose_key or "selected_csv:T_epnp_obj",
        "optimization_mode": "sample_metadata_lidar_camera_prior" if args.use_sample_metadata else "static_T_map_cam",
        "translation_sigma_mm": args.translation_sigma_mm,
        "rotation_sigma_deg": args.rotation_sigma_deg,
        "translation_prior_weight": args.translation_prior_weight,
        "rotation_prior_weight": args.rotation_prior_weight,
        "correction_translation_norm_mm": float(np.linalg.norm(xi[3:6])),
        "correction_rotation_norm_deg": float(np.degrees(np.linalg.norm(xi[:3]))),
        **summarize_errors(before_rows, "before"),
        **summarize_errors(after_rows, "after"),
    }
    (args.output_dir / "optimization_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"Wrote optimized extrinsics to {args.output_dir / 'optimized_extrinsics.json'}")


if __name__ == "__main__":
    main()
