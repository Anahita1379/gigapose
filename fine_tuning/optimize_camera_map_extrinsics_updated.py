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

When the GigaPose CAD frame differs from the EPnP centered-object frame, pass
the right-side transform from the original selector:

    T_gigapose_aligned = T_gigapose @ X_right

The optimizer then uses ``T_gigapose_aligned`` everywhere and embeds
``X_right`` in ``optimized_extrinsics.json`` so downstream selection can reuse
the identical convention.

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
        "--gigapose-frame-transform-json",
        type=Path,
        default=None,
        help=(
            "Optional frame_transform_gigapose_to_epnp.json produced by "
            "select_real_label_candidates. The transform is applied to every raw "
            "T_gigapose_cam_obj before extrinsic optimization. For the current "
            "centered-CAD/EPnP setup this should be the previously estimated "
            "right-side object-frame transform."
        ),
    )
    parser.add_argument(
        "--gigapose-frame-transform-side",
        choices=("left", "right"),
        default=None,
        help=(
            "How to apply --gigapose-frame-transform-json. If omitted, read "
            "frame_transform_side from that JSON. Use 'right' for "
            "T_aligned = T_gigapose @ X."
        ),
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
        "--use-epnp-label-extrinsics",
        action="store_true",
        help=(
            "Do not use metadata YAML. Instead derive each sample's original "
            "camera-to-map extrinsic directly from the EPnP label JSON as "
            "T_map_camera = T_map_object @ inv(T_camera_object). This requires "
            "--epnp-map-pose-key and --epnp-camera-pose-key."
        ),
    )
    parser.add_argument(
        "--epnp-camera-pose-key",
        default="T_camera_object_centered",
        help=(
            "Camera-frame object pose key inside each EPnP label JSON, used with "
            "--use-epnp-label-extrinsics to derive the per-sample original "
            "camera-to-map extrinsic."
        ),
    )
    parser.add_argument(
        "--epnp-camera-pose-unit",
        choices=("auto", "m", "mm"),
        default="auto",
        help="Translation unit for --epnp-camera-pose-key values.",
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
        "--epnp-label-root-override",
        type=Path,
        default=None,
        help=(
            "Optional folder containing the EPnP label JSONs to use for optimization. "
            "Each selected row keeps its label filename, but the parent folder is "
            "replaced by this folder. Use this to force EPnPv2_gt_mesh_z_hybrid_labels "
            "without editing selected_samples.csv."
        ),
    )
    parser.add_argument(
        "--epnp-label-dir-name",
        default=None,
        help=(
            "Optional sibling label folder name to use per row, e.g. "
            "EPnPv2_gt_mesh_z_hybrid_labels. This is safer than "
            "--epnp-label-root-override for combined CSVs spanning multiple "
            "sessions/cameras."
        ),
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
        "--translation-residual-components",
        choices=("xyz", "xy", "xz", "yz", "x", "y", "z"),
        default="xyz",
        help=(
            "Which map-frame translation residual components to include in the "
            "optimization cost. Use 'xy' when map altitude/z conventions are "
            "inconsistent but horizontal map position is meaningful."
        ),
    )
    parser.add_argument(
        "--rotation-sigma-deg",
        type=float,
        default=10.0,
        help="Pose residual scale for object rotation error.",
    )
    parser.add_argument(
        "--image-center-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for an image-plane center residual. In --use-sample-metadata "
            "mode, this projects T_map_object_raw through the current extrinsic "
            "and keeps it close to the selected GigaPose projected center. "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--image-center-sigma-px",
        type=float,
        default=50.0,
        help="Pixel residual scale for --image-center-weight.",
    )
    parser.add_argument(
        "--projection-model",
        choices=("pinhole", "metadata"),
        default="pinhole",
        help=(
            "Projection model for the image-center residual. 'pinhole' preserves "
            "old behavior. 'metadata' uses metadata distortion_model and "
            "camera_intrinsics.d when available, including plumb_bob and "
            "equidistant."
        ),
    )
    parser.add_argument(
        "--image-center-map-z-mode",
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
            "Map-object z convention used only for the image-center residual. "
            "'raw' uses T_map_object_raw z. 'metadata_lidar' shifts map object z "
            "into metadata t_map_lidar's z convention. 'ground_truth_pose' shifts "
            "relative to metadata ground_truth_pose z. 'ego_relative' preserves "
            "the EPnP object's height relative to metadata ground_truth_pose, but "
            "expresses it in the metadata t_map_lidar z convention. "
            "'session_lidar_offset' estimates one median z offset per session "
            "from selected samples, preserving EPnP z variation while shifting "
            "it into the metadata lidar z convention. 'session_lidar_affine' "
            "uses median_lidar_z + session_z_scale * (label_z - median_label_z)."
        ),
    )
    parser.add_argument(
        "--session-z-scale",
        type=float,
        default=1.0,
        help=(
            "Scale for EPnP hill/downhill z variation when using "
            "--image-center-map-z-mode session_lidar_affine. 1.0 is equivalent "
            "to preserving the full selected-label z variation; 0.0 flattens to "
            "the session median lidar z; negative values invert the variation."
        ),
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


def load_gigapose_frame_transform(
    path: Path | None,
    requested_side: str | None,
) -> tuple[np.ndarray | None, str | None, Path | None]:
    if path is None:
        if requested_side is not None:
            raise ValueError(
                "--gigapose-frame-transform-side requires "
                "--gigapose-frame-transform-json"
            )
        return None, None, None
    if not path.is_file():
        raise FileNotFoundError(f"Missing GigaPose frame transform: {path}")

    data = json.loads(path.read_text())
    value = None
    for key in (
        "T_epnp_gigapose",
        "T_epnp_gigapose_right",
        "T_epnp_gigapose_left",
        "T_target_source",
        "matrix",
        "T",
    ):
        if data.get(key) is not None:
            value = data[key]
            break
    if value is None:
        raise ValueError(f"{path} does not contain a supported 4x4 transform")

    stored_side = data.get("frame_transform_side")
    if stored_side is not None:
        stored_side = str(stored_side)
        if stored_side not in ("left", "right"):
            raise ValueError(
                f"{path} has invalid frame_transform_side={stored_side!r}"
            )
    if (
        requested_side is not None
        and stored_side is not None
        and requested_side != stored_side
    ):
        raise ValueError(
            f"Requested frame-transform side {requested_side!r}, but {path} "
            f"records {stored_side!r}"
        )
    side = requested_side or stored_side
    if side is None:
        raise ValueError(
            f"{path} does not record frame_transform_side; pass "
            "--gigapose-frame-transform-side explicitly"
        )
    transform = np.asarray(value, dtype=float).reshape(4, 4)
    if not np.isfinite(transform).all():
        raise ValueError(f"{path} contains non-finite transform values")
    return transform, side, path


def apply_gigapose_frame_transform(
    T_gigapose: np.ndarray,
    transform: np.ndarray | None,
    side: str | None,
) -> np.ndarray:
    if transform is None:
        return T_gigapose
    if side == "left":
        return transform @ T_gigapose
    if side == "right":
        return T_gigapose @ transform
    raise ValueError(f"Unknown GigaPose frame-transform side: {side!r}")


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


def project_points(
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


def project_origin(
    T_cam_obj: np.ndarray,
    K: np.ndarray,
    D: np.ndarray | None = None,
    distortion_model: str = "pinhole",
    projection_model: str = "pinhole",
) -> tuple[np.ndarray, bool]:
    uv, valid = project_points(
        np.zeros((1, 3), dtype=float),
        T_cam_obj,
        K,
        D,
        distortion_model,
        projection_model,
    )
    return uv[0], bool(valid[0])


def translation_component_indices(components: str) -> list[int]:
    mapping = {"x": 0, "y": 1, "z": 2}
    return [mapping[c] for c in components]


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

    # Newer EPnPv2 label JSONs explicitly store the metadata YAML used to
    # generate that label. Prefer that over filename inference, because it makes
    # the GT-label <-> calibration/ego-metadata pairing unambiguous.
    if label_path.is_file():
        try:
            label_data = json.loads(label_path.read_text())
        except Exception:
            label_data = {}
        metadata_path_text = label_data.get("metadata_path") if isinstance(label_data, dict) else None
        if metadata_path_text:
            metadata_path = Path(metadata_path_text)
            if metadata_path.is_file():
                return metadata_path

    match = re.match(r"(.+)_([0-9]+)$", label_path.stem)
    if not match:
        raise ValueError(f"Cannot infer metadata sample name from {label_path.name}")
    timestamp, obj_index = match.groups()
    metadata_path = label_path.parent.parent / "metadata" / f"sample_{timestamp}_{obj_index}.yaml"
    if metadata_path.is_file():
        return metadata_path
    raise FileNotFoundError(f"Could not find metadata YAML for {label_path}: {metadata_path}")


def label_path_from_row(
    row: dict[str, Any],
    label_root_override: Path | None = None,
    label_dir_name: str | None = None,
) -> Path:
    label_path_text = row.get("epnp_label_path")
    if not label_path_text:
        raise ValueError("Selected sample row is missing epnp_label_path")
    label_path = Path(label_path_text)
    if label_dir_name:
        label_path = label_path.parent.parent / label_dir_name / label_path.name
    if label_root_override is not None:
        label_path = label_root_override / label_path.name
    return label_path


def matrix_from_yaml_key(data: dict[str, Any], key: str, unit: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"{key} missing from metadata YAML")
    return matrix_3x4_or_4x4_to_transform(data[key], unit)


def load_metadata_camera(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str] | None:
    intrinsics = data.get("camera_intrinsics", {})
    if isinstance(intrinsics, dict) and "k" in intrinsics:
        K = np.asarray(intrinsics["k"], dtype=float).reshape(3, 3)
        D = np.asarray(intrinsics.get("d", []), dtype=float).reshape(-1)
        model = str(intrinsics.get("distortion_model", data.get("distortion_model", "pinhole")))
        return K, D, model
    if "K" in data:
        K = np.asarray(data["K"], dtype=float).reshape(3, 3)
        D = np.asarray(data.get("D", []), dtype=float).reshape(-1)
        model = str(data.get("distortion_model", "pinhole"))
        return K, D, model
    return None


def load_label_camera(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str] | None:
    if "K" in data:
        K = np.asarray(data["K"], dtype=float).reshape(3, 3)
        D = np.asarray(data.get("D", []), dtype=float).reshape(-1)
        model = str(data.get("distortion_model", "pinhole") or "pinhole")
        return K, D, model
    return load_metadata_camera(data)


def metadata_session_key(metadata_path: str | Path) -> str:
    path = Path(metadata_path)
    # .../<session>/<camera>/metadata/sample_*.yaml -> .../<session>/<camera>
    if path.parent.name == "metadata":
        return str(path.parent.parent)
    return str(path.parent)


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
        raise ValueError(f"Unknown image-center map-z mode: {mode}")
    out[2, 3] += (target_z_m - label_z_m) * 1000.0
    return out


def add_session_lidar_z_stats(samples: list[dict[str, Any]]) -> None:
    offsets_by_session: dict[str, list[float]] = {}
    label_z_by_session: dict[str, list[float]] = {}
    lidar_z_by_session: dict[str, list[float]] = {}
    for sample in samples:
        metadata = sample.get("metadata")
        metadata_path = sample.get("metadata_path")
        if metadata is None or not metadata_path:
            continue
        lidar_z_m = float(np.asarray(metadata["t_map_lidar"], dtype=float).reshape(4, 4)[2, 3])
        label_z_m = float(sample["T_target_obj"][2, 3] * 0.001)
        session = metadata_session_key(metadata_path)
        offsets_by_session.setdefault(session, []).append(lidar_z_m - label_z_m)
        label_z_by_session.setdefault(session, []).append(label_z_m)
        lidar_z_by_session.setdefault(session, []).append(lidar_z_m)

    median_offsets = {
        key: float(np.median(np.asarray(values, dtype=float)))
        for key, values in offsets_by_session.items()
        if values
    }
    median_label_z = {
        key: float(np.median(np.asarray(values, dtype=float)))
        for key, values in label_z_by_session.items()
        if values
    }
    median_lidar_z = {
        key: float(np.median(np.asarray(values, dtype=float)))
        for key, values in lidar_z_by_session.items()
        if values
    }
    for sample in samples:
        metadata_path = sample.get("metadata_path")
        if not metadata_path:
            continue
        session = metadata_session_key(metadata_path)
        sample["session_lidar_z_offset_m"] = median_offsets.get(session, 0.0)
        sample["session_label_z_median"] = median_label_z.get(session, 0.0)
        sample["session_lidar_z_median"] = median_lidar_z.get(session, 0.0)


def load_sample_metadata_transforms(
    row: dict[str, Any], metadata_path_field: str
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, str] | None, dict[str, Any], str]:
    path = metadata_path_from_row(row, metadata_path_field)
    data = load_yaml(path)
    # The real-data metadata stores these in meters.
    T_map_lidar = matrix_from_yaml_key(data, "t_map_lidar", "m")
    T_lidar_camera_prior = matrix_from_yaml_key(data, "t_lidar_camera_prior", "m")
    camera = load_metadata_camera(data)
    return T_map_lidar, T_lidar_camera_prior, camera, data, str(path)


def load_map_pose_from_epnp_label(path: Path, pose_key: str, unit: str) -> tuple[np.ndarray, dict[str, Any]]:
    data = json.loads(path.read_text())
    if pose_key not in data:
        available_pose_keys = [
            key
            for key in data.keys()
            if "pose" in key.lower() or key.startswith("T_") or key in ("translation", "rotation")
        ]
        raise KeyError(
            f"{pose_key} not found in {path}. Pose-like keys available: {available_pose_keys}"
        )
    return matrix_3x4_or_4x4_to_transform(data[pose_key], unit), data


def load_pose_from_label_data(
    data: dict[str, Any], pose_key: str, unit: str, path_for_error: Path
) -> np.ndarray:
    if pose_key not in data:
        available_pose_keys = [
            key
            for key in data.keys()
            if "pose" in key.lower() or key.startswith("T_") or key in ("translation", "rotation")
        ]
        raise KeyError(
            f"{pose_key} not found in {path_for_error}. Pose-like keys available: {available_pose_keys}"
        )
    return matrix_3x4_or_4x4_to_transform(data[pose_key], unit)


def load_selected_samples(
    path: Path,
    max_samples: int | None,
    gigapose_frame_transform: np.ndarray | None,
    gigapose_frame_transform_side: str | None,
    epnp_map_pose_key: str | None,
    epnp_map_pose_unit: str,
    use_sample_metadata: bool,
    use_epnp_label_extrinsics: bool,
    epnp_camera_pose_key: str,
    epnp_camera_pose_unit: str,
    epnp_label_root_override: Path | None,
    epnp_label_dir_name: str | None,
    metadata_path_field: str,
    image_center_map_z_mode: str,
    session_z_scale: float,
) -> list[dict[str, Any]]:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                T_giga_raw = text_to_matrix(row["T_gigapose_cam_obj"])
                T_giga = apply_gigapose_frame_transform(
                    T_giga_raw,
                    gigapose_frame_transform,
                    gigapose_frame_transform_side,
                )
                label_data = {}
                if epnp_map_pose_key:
                    label_path = label_path_from_row(
                        row, epnp_label_root_override, epnp_label_dir_name
                    )
                    T_target, label_data = load_map_pose_from_epnp_label(
                        label_path, epnp_map_pose_key, epnp_map_pose_unit
                    )
                else:
                    label_path = Path(row.get("epnp_label_path", ""))
                    T_target = text_to_matrix(row["T_epnp_obj"])
                metadata_values = {}
                if use_sample_metadata:
                    T_map_lidar, T_lidar_camera_prior, camera, metadata, metadata_path = load_sample_metadata_transforms(
                        row, metadata_path_field
                    )
                    if camera is None:
                        K, D, distortion_model = None, None, "pinhole"
                    else:
                        K, D, distortion_model = camera
                    T_target_image = (
                        apply_map_z_mode(T_target, label_data, metadata, image_center_map_z_mode)
                        if epnp_map_pose_key
                        and image_center_map_z_mode
                        not in ("session_lidar_offset", "session_lidar_affine")
                        else T_target
                    )
                    metadata_values = {
                        "T_map_lidar": T_map_lidar,
                        "T_lidar_camera_prior": T_lidar_camera_prior,
                        "K": K,
                        "D": D,
                        "distortion_model": distortion_model,
                        "T_target_obj_image": T_target_image,
                        "metadata": metadata,
                        "metadata_path": metadata_path,
                    }
                elif use_epnp_label_extrinsics:
                    if not epnp_map_pose_key:
                        raise ValueError("--use-epnp-label-extrinsics requires --epnp-map-pose-key")
                    label_path = label_path_from_row(
                        row, epnp_label_root_override, epnp_label_dir_name
                    )
                    T_label_cam_obj = load_pose_from_label_data(
                        label_data,
                        epnp_camera_pose_key,
                        epnp_camera_pose_unit,
                        label_path,
                    )
                    T_map_cam_initial_sample = T_target @ np.linalg.inv(T_label_cam_obj)
                    camera = load_label_camera(label_data)
                    if camera is None:
                        K, D, distortion_model = None, None, "pinhole"
                    else:
                        K, D, distortion_model = camera
                    metadata_values = {
                        "T_map_cam_initial_sample": T_map_cam_initial_sample,
                        "T_epnp_camera_obj": T_label_cam_obj,
                        "K": K,
                        "D": D,
                        "distortion_model": distortion_model,
                    }
            except Exception as exc:
                print(f"Skipping malformed selected sample: {exc}")
                continue
            rows.append(
                {
                    **row,
                    "T_gigapose_cam_obj": T_giga,
                    "T_gigapose_cam_obj_raw": T_giga_raw,
                    "T_target_obj": T_target,
                    "target_pose_path": str(label_path),
                    "source_csv_epnp_label_path": row.get("epnp_label_path", ""),
                    "target_pose_key": epnp_map_pose_key or "selected_csv:T_epnp_obj",
                    "camera_pose_key": epnp_camera_pose_key if use_epnp_label_extrinsics else "",
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
    if use_sample_metadata and image_center_map_z_mode in ("session_lidar_offset", "session_lidar_affine"):
        add_session_lidar_z_stats(rows)
        for sample in rows:
            sample["T_target_obj_image"] = apply_map_z_mode(
                sample["T_target_obj"],
                {},
                sample["metadata"],
                image_center_map_z_mode,
                sample.get("session_lidar_z_offset_m"),
                sample.get("session_label_z_median"),
                sample.get("session_lidar_z_median"),
                session_z_scale,
            )
    return rows


def residual_vector(
    xi: np.ndarray,
    T_map_cam_initial: np.ndarray,
    samples: list[dict[str, Any]],
    translation_sigma_mm: float,
    translation_residual_components: str,
    rotation_sigma_deg: float,
    translation_prior_weight: float,
    rotation_prior_weight: float,
) -> np.ndarray:
    T_delta = se3_exp(xi)
    T_map_cam = T_delta @ T_map_cam_initial
    residuals = []
    rotation_sigma_rad = math.radians(rotation_sigma_deg)
    t_idx = translation_component_indices(translation_residual_components)

    for sample in samples:
        T_pred = T_map_cam @ sample["T_gigapose_cam_obj"]
        T_gt = sample["T_target_obj"]
        t_res = (T_pred[:3, 3] - T_gt[:3, 3])[t_idx] / translation_sigma_mm
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
    translation_residual_components: str,
    rotation_sigma_deg: float,
    image_center_weight: float,
    image_center_sigma_px: float,
    projection_model: str,
    translation_prior_weight: float,
    rotation_prior_weight: float,
) -> np.ndarray:
    T_delta = se3_exp(xi)
    residuals = []
    rotation_sigma_rad = math.radians(rotation_sigma_deg)
    t_idx = translation_component_indices(translation_residual_components)

    for sample in samples:
        T_lidar_camera = T_delta @ sample["T_lidar_camera_prior"]
        T_map_cam = sample["T_map_lidar"] @ T_lidar_camera
        T_pred = T_map_cam @ sample["T_gigapose_cam_obj"]
        T_gt = sample["T_target_obj"]
        t_res = (T_pred[:3, 3] - T_gt[:3, 3])[t_idx] / translation_sigma_mm
        r_res = so3_log(T_pred[:3, :3] @ T_gt[:3, :3].T) / rotation_sigma_rad
        residuals.extend(t_res.tolist())
        residuals.extend(r_res.tolist())

        if image_center_weight > 0:
            # least_squares requires the residual vector length to stay fixed
            # for every optimizer step. A projection can become invalid for a
            # trial update, so always append exactly two image residual values.
            uv_res = np.zeros(2, dtype=float)
            if sample.get("K") is not None:
                T_cam_obj_from_map = np.linalg.inv(T_map_cam) @ sample.get("T_target_obj_image", T_gt)
                uv_map, map_valid = project_origin(
                    T_cam_obj_from_map,
                    sample["K"],
                    sample.get("D"),
                    sample.get("distortion_model", "pinhole"),
                    projection_model,
                )
                uv_giga, giga_valid = project_origin(
                    sample["T_gigapose_cam_obj"],
                    sample["K"],
                    sample.get("D"),
                    sample.get("distortion_model", "pinhole"),
                    projection_model,
                )
                if map_valid and giga_valid and np.isfinite(uv_map).all() and np.isfinite(uv_giga).all():
                    uv_res = ((uv_map - uv_giga) / image_center_sigma_px) * image_center_weight
            residuals.extend(uv_res.tolist())

    residuals.extend((xi[:3] * rotation_prior_weight).tolist())
    residuals.extend(((xi[3:6] / translation_sigma_mm) * translation_prior_weight).tolist())
    return np.asarray(residuals, dtype=float)


def residual_vector_epnp_label_extrinsics(
    xi: np.ndarray,
    samples: list[dict[str, Any]],
    translation_sigma_mm: float,
    translation_residual_components: str,
    rotation_sigma_deg: float,
    image_center_weight: float,
    image_center_sigma_px: float,
    projection_model: str,
    translation_prior_weight: float,
    rotation_prior_weight: float,
) -> np.ndarray:
    T_delta = se3_exp(xi)
    residuals = []
    rotation_sigma_rad = math.radians(rotation_sigma_deg)
    t_idx = translation_component_indices(translation_residual_components)

    for sample in samples:
        T_map_cam = T_delta @ sample["T_map_cam_initial_sample"]
        T_pred = T_map_cam @ sample["T_gigapose_cam_obj"]
        T_gt = sample["T_target_obj"]
        t_res = (T_pred[:3, 3] - T_gt[:3, 3])[t_idx] / translation_sigma_mm
        r_res = so3_log(T_pred[:3, :3] @ T_gt[:3, :3].T) / rotation_sigma_rad
        residuals.extend(t_res.tolist())
        residuals.extend(r_res.tolist())

        if image_center_weight > 0:
            uv_res = np.zeros(2, dtype=float)
            if sample.get("K") is not None:
                T_cam_obj_from_map = np.linalg.inv(T_map_cam) @ T_gt
                uv_map, map_valid = project_origin(
                    T_cam_obj_from_map,
                    sample["K"],
                    sample.get("D"),
                    sample.get("distortion_model", "pinhole"),
                    projection_model,
                )
                uv_giga, giga_valid = project_origin(
                    sample["T_gigapose_cam_obj"],
                    sample["K"],
                    sample.get("D"),
                    sample.get("distortion_model", "pinhole"),
                    projection_model,
                )
                if map_valid and giga_valid and np.isfinite(uv_map).all() and np.isfinite(uv_giga).all():
                    uv_res = ((uv_map - uv_giga) / image_center_sigma_px) * image_center_weight
            residuals.extend(uv_res.tolist())

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


def compute_sample_metadata_errors(
    T_delta: np.ndarray,
    samples: list[dict[str, Any]],
    projection_model: str = "pinhole",
) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        T_lidar_camera = T_delta @ sample["T_lidar_camera_prior"]
        T_map_cam = sample["T_map_lidar"] @ T_lidar_camera
        T_pred = T_map_cam @ sample["T_gigapose_cam_obj"]
        T_gt = sample["T_target_obj"]
        image_center_error_px = float("nan")
        if sample.get("K") is not None:
            T_cam_obj_from_map = np.linalg.inv(T_map_cam) @ sample.get("T_target_obj_image", T_gt)
            uv_map, map_valid = project_origin(
                T_cam_obj_from_map,
                sample["K"],
                sample.get("D"),
                sample.get("distortion_model", "pinhole"),
                projection_model,
            )
            uv_giga, giga_valid = project_origin(
                sample["T_gigapose_cam_obj"],
                sample["K"],
                sample.get("D"),
                sample.get("distortion_model", "pinhole"),
                projection_model,
            )
            if map_valid and giga_valid and np.isfinite(uv_map).all() and np.isfinite(uv_giga).all():
                image_center_error_px = float(np.linalg.norm(uv_map - uv_giga))
        rows.append(
            {
                "match_key": sample.get("match_key", ""),
                "scene_id": sample.get("scene_id", ""),
                "im_id": sample.get("im_id", ""),
                "instance_id": sample.get("instance_id", ""),
                "score": sample.get("score", ""),
                "metadata_path": sample.get("metadata_path", ""),
                "target_pose_path": sample.get("target_pose_path", ""),
                "target_pose_key": sample.get("target_pose_key", ""),
                "translation_error_mm": float(np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3])),
                "rotation_error_deg": rotation_error_deg(T_pred[:3, :3], T_gt[:3, :3]),
                "image_center_error_px": image_center_error_px,
            }
        )
    return rows


def compute_sample_epnp_label_extrinsic_errors(
    T_delta: np.ndarray,
    samples: list[dict[str, Any]],
    projection_model: str = "pinhole",
) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        T_map_cam = T_delta @ sample["T_map_cam_initial_sample"]
        T_pred = T_map_cam @ sample["T_gigapose_cam_obj"]
        T_gt = sample["T_target_obj"]
        image_center_error_px = float("nan")
        if sample.get("K") is not None:
            T_cam_obj_from_map = np.linalg.inv(T_map_cam) @ T_gt
            uv_map, map_valid = project_origin(
                T_cam_obj_from_map,
                sample["K"],
                sample.get("D"),
                sample.get("distortion_model", "pinhole"),
                projection_model,
            )
            uv_giga, giga_valid = project_origin(
                sample["T_gigapose_cam_obj"],
                sample["K"],
                sample.get("D"),
                sample.get("distortion_model", "pinhole"),
                projection_model,
            )
            if map_valid and giga_valid and np.isfinite(uv_map).all() and np.isfinite(uv_giga).all():
                image_center_error_px = float(np.linalg.norm(uv_map - uv_giga))
        rows.append(
            {
                "match_key": sample.get("match_key", ""),
                "scene_id": sample.get("scene_id", ""),
                "im_id": sample.get("im_id", ""),
                "instance_id": sample.get("instance_id", ""),
                "score": sample.get("score", ""),
                "target_pose_path": sample.get("target_pose_path", ""),
                "source_csv_epnp_label_path": sample.get("source_csv_epnp_label_path", ""),
                "target_pose_key": sample.get("target_pose_key", ""),
                "camera_pose_key": sample.get("camera_pose_key", ""),
                "translation_error_mm": float(np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3])),
                "rotation_error_deg": rotation_error_deg(T_pred[:3, :3], T_gt[:3, :3]),
                "image_center_error_px": image_center_error_px,
            }
        )
    return rows


def summarize_errors(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    out = {}
    for key in ("translation_error_mm", "rotation_error_deg", "image_center_error_px"):
        if not rows or key not in rows[0]:
            continue
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

    (
        gigapose_frame_transform,
        gigapose_frame_transform_side,
        gigapose_frame_transform_path,
    ) = load_gigapose_frame_transform(
        args.gigapose_frame_transform_json,
        args.gigapose_frame_transform_side,
    )
    samples = load_selected_samples(
        args.selected_samples,
        args.max_samples,
        gigapose_frame_transform,
        gigapose_frame_transform_side,
        args.epnp_map_pose_key,
        args.epnp_map_pose_unit,
        args.use_sample_metadata,
        args.use_epnp_label_extrinsics,
        args.epnp_camera_pose_key,
        args.epnp_camera_pose_unit,
        args.epnp_label_root_override,
        args.epnp_label_dir_name,
        args.metadata_path_field,
        args.image_center_map_z_mode,
        args.session_z_scale,
    )
    if args.use_sample_metadata and args.use_epnp_label_extrinsics:
        raise SystemExit("Use either --use-sample-metadata or --use-epnp-label-extrinsics, not both.")

    if args.use_sample_metadata:
        T_initial = np.eye(4, dtype=float)
        before_rows = compute_sample_metadata_errors(T_initial, samples, args.projection_model)
        residual_fn = residual_vector_sample_metadata
        residual_args = (
            samples,
            args.translation_sigma_mm,
            args.translation_residual_components,
            args.rotation_sigma_deg,
            args.image_center_weight,
            args.image_center_sigma_px,
            args.projection_model,
            args.translation_prior_weight,
            args.rotation_prior_weight,
        )
    elif args.use_epnp_label_extrinsics:
        T_initial = np.eye(4, dtype=float)
        before_rows = compute_sample_epnp_label_extrinsic_errors(
            T_initial, samples, args.projection_model
        )
        residual_fn = residual_vector_epnp_label_extrinsics
        residual_args = (
            samples,
            args.translation_sigma_mm,
            args.translation_residual_components,
            args.rotation_sigma_deg,
            args.image_center_weight,
            args.image_center_sigma_px,
            args.projection_model,
            args.translation_prior_weight,
            args.rotation_prior_weight,
        )
    else:
        if args.initial_extrinsic is None:
            raise SystemExit(
                "--initial-extrinsic is required unless --use-sample-metadata "
                "or --use-epnp-label-extrinsics is set."
            )
        T_initial = load_initial_extrinsic(args.initial_extrinsic, args.initial_unit)
        before_rows = compute_sample_errors(T_initial, samples)
        residual_fn = residual_vector
        residual_args = (
            T_initial,
            samples,
            args.translation_sigma_mm,
            args.translation_residual_components,
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
        after_rows = compute_sample_metadata_errors(T_delta, samples, args.projection_model)
    elif args.use_epnp_label_extrinsics:
        after_rows = compute_sample_epnp_label_extrinsic_errors(
            T_delta, samples, args.projection_model
        )
    else:
        after_rows = compute_sample_errors(T_optimized, samples)

    if args.use_sample_metadata:
        optimization_mode = "sample_metadata_lidar_camera_prior"
    elif args.use_epnp_label_extrinsics:
        optimization_mode = "epnp_label_camera_extrinsics"
    else:
        optimization_mode = "static_T_map_cam"

    write_csv(args.output_dir / "errors_before_optimization.csv", before_rows)
    write_csv(args.output_dir / "errors_after_optimization.csv", after_rows)

    extrinsics = {
        "description": "Optimized map-to-camera extrinsic correction from selected GigaPose/EPnPv2 samples.",
        "translation_unit": "mm",
        "target_pose_source": args.epnp_map_pose_key or "selected_csv:T_epnp_obj",
        "camera_pose_source": args.epnp_camera_pose_key if args.use_epnp_label_extrinsics else None,
        "epnp_label_root_override": (
            str(args.epnp_label_root_override) if args.epnp_label_root_override is not None else None
        ),
        "epnp_label_dir_name": args.epnp_label_dir_name,
        "optimization_mode": optimization_mode,
        "gigapose_frame_transform_source": (
            str(gigapose_frame_transform_path)
            if gigapose_frame_transform_path is not None
            else None
        ),
        "gigapose_frame_transform_side": gigapose_frame_transform_side,
        "T_gigapose_frame_transform": (
            gigapose_frame_transform.tolist()
            if gigapose_frame_transform is not None
            else None
        ),
        "T_map_cam_initial": T_initial.tolist() if not args.use_sample_metadata and not args.use_epnp_label_extrinsics else None,
        "T_correction_left_multiply": T_delta.tolist(),
        "T_map_cam_optimized": T_optimized.tolist() if not args.use_sample_metadata and not args.use_epnp_label_extrinsics else None,
        "T_lidar_camera_correction_left_multiply": T_delta.tolist() if args.use_sample_metadata else None,
        "T_epnp_label_camera_correction_left_multiply": T_delta.tolist() if args.use_epnp_label_extrinsics else None,
        "correction_rotation_rpy_like_vector_rad": xi[:3].tolist(),
        "correction_translation_mm": xi[3:6].tolist(),
        "image_center_weight": args.image_center_weight,
        "image_center_sigma_px": args.image_center_sigma_px,
        "image_center_map_z_mode": args.image_center_map_z_mode,
        "session_z_scale": args.session_z_scale,
        "projection_model": args.projection_model,
        "note": (
            "Sample-metadata mode: apply T_lidar_camera_optimized = "
            "T_lidar_camera_correction_left_multiply @ t_lidar_camera_prior for each frame, "
            "then T_map_cam = t_map_lidar @ T_lidar_camera_optimized."
            if args.use_sample_metadata
            else (
                "EPnP-label-extrinsics mode: for each frame, derive "
                "T_map_cam_initial_i = T_map_object_raw_i @ inv(T_camera_object_i), then apply "
                "T_map_cam_optimized_i = T_epnp_label_camera_correction_left_multiply @ "
                "T_map_cam_initial_i."
                if args.use_epnp_label_extrinsics
                else "Use T_map_cam_optimized as camera-to-map transform if your pipeline expects T_map_cam."
            )
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
        "camera_pose_source": args.epnp_camera_pose_key if args.use_epnp_label_extrinsics else None,
        "epnp_label_root_override": (
            str(args.epnp_label_root_override) if args.epnp_label_root_override is not None else None
        ),
        "epnp_label_dir_name": args.epnp_label_dir_name,
        "optimization_mode": optimization_mode,
        "gigapose_frame_transform_source": (
            str(gigapose_frame_transform_path)
            if gigapose_frame_transform_path is not None
            else None
        ),
        "gigapose_frame_transform_side": gigapose_frame_transform_side,
        "translation_sigma_mm": args.translation_sigma_mm,
        "translation_residual_components": args.translation_residual_components,
        "rotation_sigma_deg": args.rotation_sigma_deg,
        "image_center_weight": args.image_center_weight,
        "image_center_sigma_px": args.image_center_sigma_px,
        "image_center_map_z_mode": args.image_center_map_z_mode,
        "session_z_scale": args.session_z_scale,
        "projection_model": args.projection_model,
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
