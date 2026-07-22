"""Optimize map-to-camera extrinsic correction from selected real labels.

This script uses selected samples produced by:

    python -m fine_tuning.select_real_label_candidates ...

Expected selected CSV columns:

    T_gigapose_cam_obj
    T_gigapose_aligned_epnp_obj  # preferred when present
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


For the ARCL/Assetto real folders, the ego vehicle moves, so each frame has its
own t_map_lidar metadata. The metadata t_lidar_camera_prior values may also
differ slightly because of rounding, serialization, or metadata-generation
differences.

In ``--use-sample-metadata`` mode, the slightly different recorded priors are
robustly combined to form one initialization for the camera being calibrated:

    T_lidar_camera_initial = representative(T_lidar_camera_prior_i)
    T_lidar_camera_optimized = exp(delta) @ T_lidar_camera_initial

For sample i, both object poses are compared in the LiDAR frame:

    T_lidar_object_gigapose
        = T_lidar_camera_optimized @ T_gigapose_cam_obj_i

    T_lidar_object_epnp
        = inv(T_map_lidar_target_i) @ T_map_object_raw_i
        

When the GigaPose CAD frame differs from the EPnP centered-object frame, the
updated optimizer automatically uses ``T_gigapose_aligned_epnp_obj`` when that
column is already present. Alternatively, pass the right-side transform from
the original selector:

    T_gigapose_aligned = T_gigapose @ X_right

The optimizer then uses ``T_gigapose_aligned`` everywhere. It refuses an
explicit aligned pose plus an external transform, preventing accidental double
alignment.

For the ARCL/Assetto real folders, the ego vehicle moves, so each frame has its
own ``t_map_lidar`` metadata.  In that case use ``--use-sample-metadata`` and
the optimized model becomes:

    T_map_object_raw_i ≈ T_map_lidar_target_i @ T_lidar_camera_optimized @ T_gigapose_cam_obj_i

``T_map_lidar_target_i`` is preprocessed before optimization. For mesh-z hybrid
EPnP labels, only its map-frame z translation is replaced by
``gt_xy_mesh_z_label.ego_z_compensation.corrected_z_m``. Optionally, its raw
LiDAR-time pose is first interpolated to the image timestamp using neighboring
metadata poses. The exact policy and diagnostics are saved with the optimized
extrinsics and reused by the matching selector.

The per-sample priors are used only to construct and diagnose the common
initial transform. The result is one reusable physical camera-LiDAR transform
per camera, optimized from all samples and dates for that camera.

The absolute optimized transform, its inverse, the correction, and all
per-frame map/camera compositions are saved for later reuse and reselection.


The reported translation error is the object-position disagreement in the LiDAR frame
It is not the camera–LiDAR calibration translation itself.

translation_error_mm: object-pose disagreement between GigaPose and EPnP.
correction_translation_norm_mm: how far the optimized camera–LiDAR calibration moved from its initial value.
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
        "--gigapose-pose-source",
        choices=("auto", "aligned", "raw"),
        default="auto",
        help=(
            "Which GigaPose pose from the selected-samples CSV to optimize. "
            "'auto' (recommended) uses T_gigapose_aligned_epnp_obj when that "
            "column exists, otherwise T_gigapose_cam_obj. 'aligned' requires "
            "the aligned column. 'raw' explicitly uses the unaligned prediction. "
            "When --gigapose-frame-transform-json is supplied, the raw pose is "
            "used and transformed exactly once."
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
        "--target-lidar-z-mode",
        choices=("raw", "epnp_corrected"),
        default="epnp_corrected",
        help=(
            "Map-frame z convention for t_map_lidar when constructing the "
            "EPnP target. 'epnp_corrected' replaces only t_map_lidar[2,3] "
            "with gt_xy_mesh_z_label.ego_z_compensation.corrected_z_m from "
            "the EPnP label. This is required for mesh-z hybrid labels."
        ),
    )
    parser.add_argument(
        "--allow-missing-corrected-lidar-z",
        action="store_true",
        help=(
            "With --target-lidar-z-mode epnp_corrected, allow labels without "
            "corrected_z_m to fall back to raw metadata z. By default the run "
            "fails instead of silently mixing vertical frames."
        ),
    )
    parser.add_argument(
        "--timestamp-alignment",
        choices=("raw", "interpolate_metadata"),
        default="raw",
        help=(
            "How to align t_map_lidar to the image timestamp. "
            "'interpolate_metadata' scans each session/camera metadata folder, "
            "interpolates translation and SO(3) rotation between bracketing "
            "LiDAR poses, and then applies the selected target-lidar z mode."
        ),
    )
    parser.add_argument(
        "--timestamp-max-bracket-gap-ms",
        type=float,
        default=200.0,
        help=(
            "Maximum time between the two LiDAR trajectory poses used to "
            "interpolate an image-time pose."
        ),
    )
    parser.add_argument(
        "--timestamp-fallback",
        choices=("error", "skip", "raw"),
        default="error",
        help=(
            "Behavior when timestamp interpolation cannot be performed. "
            "'skip' is recommended for calibration because it omits unaligned "
            "boundary samples; 'error' stops immediately; 'raw' keeps the "
            "original pose and records the fallback in diagnostics."
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
            "Squared-loss coefficient for the image-plane center residual. "
            "The least-squares residual is multiplied by sqrt(weight). In "
            "--use-sample-metadata or --use-epnp-label-extrinsics mode, this "
            "projects the map pose through the current extrinsic and keeps it "
            "close to the selected GigaPose projected center. Set to 0 to disable."
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
        default=0.0,
        help=(
            "Squared-loss coefficient for changing the extrinsic translation; "
            "the residual is multiplied by sqrt(weight). Larger values keep "
            "translation closer to the initial extrinsic."
        ),
    )
    parser.add_argument(
        "--rotation-prior-weight",
        type=float,
        default=0.0,
        help=(
            "Squared-loss coefficient for changing the extrinsic rotation; the "
            "residual is multiplied by sqrt(weight). Use a lower value than "
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


def selected_csv_fieldnames(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or [])


def resolve_gigapose_pose_source(
    path: Path,
    requested_source: str,
    external_transform: np.ndarray | None,
) -> tuple[str, bool]:
    """Choose one pose convention once for the complete optimization run."""

    fieldnames = selected_csv_fieldnames(path)
    has_raw = "T_gigapose_cam_obj" in fieldnames
    has_aligned = "T_gigapose_aligned_epnp_obj" in fieldnames
    if not has_raw:
        raise ValueError(f"{path} is missing required column T_gigapose_cam_obj")

    if external_transform is not None:
        if requested_source == "aligned":
            raise ValueError(
                "--gigapose-pose-source aligned cannot be combined with "
                "--gigapose-frame-transform-json; that would apply alignment twice."
            )
        return "raw_csv_plus_external_transform", has_aligned

    if requested_source == "aligned":
        if not has_aligned:
            raise ValueError(
                f"{path} does not contain T_gigapose_aligned_epnp_obj. "
                "Use --gigapose-pose-source raw or provide a frame-transform JSON."
            )
        return "aligned_csv", has_aligned
    if requested_source == "raw":
        return "raw_csv", has_aligned
    if requested_source != "auto":
        raise ValueError(f"Unknown GigaPose pose source: {requested_source!r}")
    return ("aligned_csv" if has_aligned else "raw_csv"), has_aligned


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


def loss_weight_scale(weight: float) -> float:
    """Convert a loss coefficient into its least-squares residual multiplier."""

    return math.sqrt(weight)


def image_center_residual(
    T_candidate_cam_obj: np.ndarray,
    T_gigapose_cam_obj: np.ndarray,
    sample: dict[str, Any],
    image_center_weight: float,
    image_center_sigma_px: float,
    projection_model: str,
) -> np.ndarray:
    """Return exactly two weighted residuals without rewarding invalid depth."""

    if image_center_weight <= 0 or sample.get("K") is None:
        return np.zeros(2, dtype=float)

    uv_candidate, candidate_valid = project_origin(
        T_candidate_cam_obj,
        sample["K"],
        sample.get("D"),
        sample.get("distortion_model", "pinhole"),
        projection_model,
    )
    uv_gigapose, gigapose_valid = project_origin(
        T_gigapose_cam_obj,
        sample["K"],
        sample.get("D"),
        sample.get("distortion_model", "pinhole"),
        projection_model,
    )
    weight_scale = loss_weight_scale(image_center_weight)
    if (
        candidate_valid
        and gigapose_valid
        and np.isfinite(uv_candidate).all()
        and np.isfinite(uv_gigapose).all()
    ):
        return (
            (uv_candidate - uv_gigapose)
            / image_center_sigma_px
            * weight_scale
        )

    # Returning zero for an invalid trial projection makes moving the object
    # behind the camera artificially attractive. Keep a fixed-length residual
    # and make the penalty grow smoothly with negative depth instead.
    candidate_z = float(T_candidate_cam_obj[2, 3])
    gigapose_z = float(T_gigapose_cam_obj[2, 3])
    min_depth = min(candidate_z, gigapose_z)
    depth_violation_m = max(0.0, -min_depth / 1000.0)
    penalty = weight_scale * (10.0 + depth_violation_m)
    return np.full(2, penalty, dtype=float)


def correction_prior_residuals(
    xi: np.ndarray,
    translation_sigma_mm: float,
    translation_prior_weight: float,
    rotation_prior_weight: float,
) -> list[float]:
    """Priors whose CLI weights are coefficients in the squared loss."""

    rotation = xi[:3] * loss_weight_scale(rotation_prior_weight)
    translation = (
        xi[3:6]
        / translation_sigma_mm
        * loss_weight_scale(translation_prior_weight)
    )
    return [*rotation.tolist(), *translation.tolist()]


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


def corrected_lidar_map_z_mm(label_data: dict[str, Any]) -> float | None:
    """Read the mesh-frame LiDAR altitude recorded by hybrid EPnP labels."""

    value = (
        label_data.get("gt_xy_mesh_z_label", {})
        .get("ego_z_compensation", {})
        .get("corrected_z_m")
    )
    if value is None:
        return None
    value_mm = float(value) * 1000.0
    return value_mm if np.isfinite(value_mm) else None


def _nested_items(
    value: Any, prefix: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], Any]]:
    items: list[tuple[tuple[str, ...], Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = (*prefix, str(key).lower())
            items.append((path, child))
            items.extend(_nested_items(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            items.extend(_nested_items(child, (*prefix, str(index))))
    return items


def _timestamp_value_ns(value: Any, key_path: tuple[str, ...]) -> int | None:
    """Convert common scalar/ROS timestamp representations to nanoseconds."""

    if isinstance(value, dict):
        sec = value.get("sec", value.get("secs"))
        nsec = value.get(
            "nanosec", value.get("nsec", value.get("nsecs", 0))
        )
        if sec is not None:
            try:
                return int(round(float(sec) * 1e9 + float(nsec)))
            except (TypeError, ValueError):
                return None

    number: float | None = None
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
    elif isinstance(value, str):
        stripped = value.strip()
        try:
            number = float(stripped)
        except ValueError:
            # Sensor file paths often carry the timestamp in their basename.
            tokens = re.findall(r"(?<!\d)(\d{10,})(?!\d)", stripped)
            if tokens:
                number = float(max(tokens, key=len))
    if number is None or not np.isfinite(number):
        return None

    joined = ".".join(key_path)
    if any(token in joined for token in ("_sec", "seconds", ".sec")) and not any(
        token in joined for token in ("nsec", "nanosec")
    ):
        number *= 1e9
    elif any(token in joined for token in ("_ms", "millisecond")):
        number *= 1e6
    elif any(token in joined for token in ("_us", "microsecond")):
        number *= 1e3
    # Unqualified timestamps in this dataset are already nanoseconds.
    return int(round(number))


def extract_sensor_timestamp_ns(
    metadata: dict[str, Any], sensor: str, metadata_path: str | Path
) -> int | None:
    """Find an image or LiDAR timestamp without assuming one YAML layout."""

    sensor_tokens = {
        "image": ("image", "camera", "rgb", "color"),
        "lidar": ("lidar", "pointcloud", "point_cloud", "points"),
    }[sensor]
    candidates: list[tuple[int, int]] = []
    for key_path, value in _nested_items(metadata):
        joined = ".".join(key_path)
        if not any(token in joined for token in sensor_tokens):
            continue
        if not any(
            token in joined
            for token in ("timestamp", "stamp", "time_ns", "file", "path")
        ):
            continue
        timestamp_ns = _timestamp_value_ns(value, key_path)
        if timestamp_ns is None:
            continue
        score = 0
        if "timestamp_ns" in joined or "time_ns" in joined:
            score += 8
        if "timestamp" in joined:
            score += 4
        if "stamp" in joined:
            score += 2
        if key_path and any(token in key_path[-1] for token in sensor_tokens):
            score += 2
        candidates.append((score, timestamp_ns))

    if candidates:
        return max(candidates, key=lambda item: item[0])[1]

    if sensor == "image":
        # sample_<timestamp>_<instance>.yaml uses the image timestamp.
        match = re.match(r"sample_(\d+)_\d+$", Path(metadata_path).stem)
        if match:
            return int(match.group(1))
    return None


def interpolate_transform(
    T0: np.ndarray, T1: np.ndarray, alpha: float
) -> np.ndarray:
    """Interpolate SE(3): linear translation and geodesic SO(3) rotation."""

    alpha = float(np.clip(alpha, 0.0, 1.0))
    out = np.eye(4, dtype=float)
    out[:3, 3] = (1.0 - alpha) * T0[:3, 3] + alpha * T1[:3, 3]
    relative = T0[:3, :3].T @ T1[:3, :3]
    out[:3, :3] = T0[:3, :3] @ so3_exp(alpha * so3_log(relative))
    return out


def load_metadata_lidar_trajectory(
    metadata_dir: Path,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, int]]:
    """Load the LiDAR-time ego trajectory represented by a metadata folder."""

    poses_by_timestamp: dict[int, np.ndarray] = {}
    diagnostics = {
        "metadata_files_scanned": 0,
        "trajectory_poses_loaded": 0,
        "trajectory_missing_lidar_timestamp": 0,
        "trajectory_malformed_metadata": 0,
    }
    for path in sorted(metadata_dir.glob("sample_*.yaml")):
        diagnostics["metadata_files_scanned"] += 1
        try:
            metadata = load_yaml(path)
            timestamp_ns = extract_sensor_timestamp_ns(
                metadata, "lidar", path
            )
            if timestamp_ns is None:
                diagnostics["trajectory_missing_lidar_timestamp"] += 1
                continue
            pose = matrix_from_yaml_key(metadata, "t_map_lidar", "m")
            poses_by_timestamp.setdefault(timestamp_ns, pose)
        except Exception:
            diagnostics["trajectory_malformed_metadata"] += 1

    timestamps = np.asarray(sorted(poses_by_timestamp), dtype=np.int64)
    poses = [poses_by_timestamp[int(timestamp)] for timestamp in timestamps]
    diagnostics["trajectory_poses_loaded"] = len(poses)
    return timestamps, poses, diagnostics


def prepare_target_map_lidar_transforms(
    samples: list[dict[str, Any]],
    target_lidar_z_mode: str,
    allow_missing_corrected_lidar_z: bool,
    timestamp_alignment: str,
    timestamp_max_bracket_gap_ms: float,
    timestamp_fallback: str,
) -> dict[str, Any]:
    """Build vertically and temporally consistent map<-LiDAR target poses."""

    trajectory_cache: dict[
        Path, tuple[np.ndarray, list[np.ndarray], dict[str, int]]
    ] = {}
    diagnostics: dict[str, Any] = {
        "target_lidar_z_mode": target_lidar_z_mode,
        "timestamp_alignment": timestamp_alignment,
        "samples_total": len(samples),
        "samples_corrected_z": 0,
        "samples_missing_corrected_z": 0,
        "samples_timestamp_interpolated": 0,
        "samples_timestamp_skipped": 0,
        "samples_timestamp_raw_fallback": 0,
        "timestamp_failures": {},
    }
    offsets_ms: list[float] = []
    bracket_gaps_ms: list[float] = []
    time_translation_shifts_mm: list[float] = []
    time_rotation_shifts_deg: list[float] = []
    z_replacements_mm: list[float] = []
    prepared_samples: list[dict[str, Any]] = []

    for sample in samples:
        raw = np.asarray(sample["T_map_lidar"], dtype=float).reshape(4, 4)
        time_aligned = raw.copy()
        metadata_path = Path(sample["metadata_path"])
        metadata = sample["metadata"]
        image_timestamp_ns = extract_sensor_timestamp_ns(
            metadata, "image", metadata_path
        )
        lidar_timestamp_ns = extract_sensor_timestamp_ns(
            metadata, "lidar", metadata_path
        )
        sample["image_timestamp_ns"] = image_timestamp_ns
        sample["lidar_timestamp_ns"] = lidar_timestamp_ns
        if image_timestamp_ns is not None and lidar_timestamp_ns is not None:
            offsets_ms.append(
                (image_timestamp_ns - lidar_timestamp_ns) / 1e6
            )

        if timestamp_alignment == "interpolate_metadata":
            failure: str | None = None
            metadata_dir = metadata_path.parent
            if metadata_dir not in trajectory_cache:
                trajectory_cache[metadata_dir] = load_metadata_lidar_trajectory(
                    metadata_dir
                )
            timestamps, poses, _ = trajectory_cache[metadata_dir]
            if image_timestamp_ns is None:
                failure = "missing_image_timestamp"
            elif timestamps.size < 2:
                failure = "insufficient_trajectory"
            else:
                right = int(np.searchsorted(timestamps, image_timestamp_ns))
                if (
                    right < timestamps.size
                    and int(timestamps[right]) == image_timestamp_ns
                ):
                    time_aligned = poses[right].copy()
                    sample["timestamp_interpolation_left_ns"] = (
                        image_timestamp_ns
                    )
                    sample["timestamp_interpolation_right_ns"] = (
                        image_timestamp_ns
                    )
                    sample["timestamp_interpolation_alpha"] = 0.0
                    sample["timestamp_alignment_status"] = "exact"
                    diagnostics["samples_timestamp_interpolated"] += 1
                elif right == 0 or right >= timestamps.size:
                    failure = "image_timestamp_not_bracketed"
                else:
                    left = right - 1
                    t0, t1 = int(timestamps[left]), int(timestamps[right])
                    gap_ms = (t1 - t0) / 1e6
                    if gap_ms > timestamp_max_bracket_gap_ms:
                        failure = "trajectory_bracket_too_wide"
                    else:
                        alpha = (image_timestamp_ns - t0) / float(t1 - t0)
                        time_aligned = interpolate_transform(
                            poses[left], poses[right], alpha
                        )
                        sample["timestamp_interpolation_left_ns"] = t0
                        sample["timestamp_interpolation_right_ns"] = t1
                        sample["timestamp_interpolation_alpha"] = float(alpha)
                        sample["timestamp_alignment_status"] = "interpolated"
                        diagnostics["samples_timestamp_interpolated"] += 1
                        bracket_gaps_ms.append(gap_ms)
            if failure is not None:
                failures = diagnostics["timestamp_failures"]
                failures[failure] = failures.get(failure, 0) + 1
                sample["timestamp_alignment_status"] = f"raw_fallback:{failure}"
                if timestamp_fallback == "error":
                    raise ValueError(
                        f"Cannot timestamp-align {metadata_path}: {failure}. "
                        "Use --timestamp-fallback skip to omit this sample, or "
                        "raw only if an explicitly recorded fallback is acceptable."
                    )
                if timestamp_fallback == "skip":
                    diagnostics["samples_timestamp_skipped"] += 1
                    continue
                diagnostics["samples_timestamp_raw_fallback"] += 1
        else:
            sample["timestamp_alignment_status"] = "raw"

        sample["T_map_lidar_raw"] = raw.copy()
        sample["T_map_lidar_time_aligned"] = time_aligned.copy()
        time_translation_shift_mm = float(
            np.linalg.norm(time_aligned[:3, 3] - raw[:3, 3])
        )
        time_rotation_shift_deg = rotation_error_deg(
            time_aligned[:3, :3], raw[:3, :3]
        )
        sample["timestamp_alignment_translation_shift_mm"] = (
            time_translation_shift_mm
        )
        sample["timestamp_alignment_rotation_shift_deg"] = (
            time_rotation_shift_deg
        )
        if sample["timestamp_alignment_status"] == "interpolated":
            time_translation_shifts_mm.append(time_translation_shift_mm)
            time_rotation_shifts_deg.append(time_rotation_shift_deg)
        target = time_aligned.copy()
        corrected_z_mm = corrected_lidar_map_z_mm(sample.get("label_data", {}))
        sample["corrected_lidar_map_z_mm"] = corrected_z_mm
        if target_lidar_z_mode == "epnp_corrected":
            if corrected_z_mm is None:
                diagnostics["samples_missing_corrected_z"] += 1
                if not allow_missing_corrected_lidar_z:
                    raise ValueError(
                        f"{sample.get('target_pose_path')} has no "
                        "gt_xy_mesh_z_label.ego_z_compensation.corrected_z_m. "
                        "Use --allow-missing-corrected-lidar-z only to permit "
                        "an explicitly reported raw-z fallback."
                    )
            else:
                z_replacement_mm = float(corrected_z_mm - target[2, 3])
                target[2, 3] = corrected_z_mm
                sample["target_lidar_z_replacement_mm"] = z_replacement_mm
                z_replacements_mm.append(z_replacement_mm)
                diagnostics["samples_corrected_z"] += 1
        sample["T_map_lidar_target"] = target
        prepared_samples.append(sample)

    samples[:] = prepared_samples
    diagnostics["samples_used"] = len(samples)
    if not samples:
        raise ValueError(
            "Target LiDAR preprocessing removed every sample. Inspect timestamp "
            "fields, trajectory availability, and the bracket-gap threshold."
        )

    for _, _, trajectory_diagnostics in trajectory_cache.values():
        for key, value in trajectory_diagnostics.items():
            diagnostics[key] = diagnostics.get(key, 0) + value
    if offsets_ms:
        values = np.asarray(offsets_ms, dtype=float)
        diagnostics.update(
            {
                "image_minus_lidar_offset_ms_mean": float(np.mean(values)),
                "image_minus_lidar_offset_ms_median": float(np.median(values)),
                "image_minus_lidar_offset_ms_p90_abs": float(
                    np.percentile(np.abs(values), 90)
                ),
            }
        )
    if bracket_gaps_ms:
        values = np.asarray(bracket_gaps_ms, dtype=float)
        diagnostics.update(
            {
                "trajectory_bracket_gap_ms_median": float(np.median(values)),
                "trajectory_bracket_gap_ms_max": float(np.max(values)),
            }
        )
    for name, values in (
        ("timestamp_translation_shift_mm", time_translation_shifts_mm),
        ("timestamp_rotation_shift_deg", time_rotation_shifts_deg),
        ("target_lidar_z_replacement_mm", z_replacements_mm),
    ):
        if not values:
            continue
        array = np.asarray(values, dtype=float)
        diagnostics[f"{name}_mean"] = float(np.mean(array))
        diagnostics[f"{name}_median"] = float(np.median(array))
        diagnostics[f"{name}_p90_abs"] = float(
            np.percentile(np.abs(array), 90)
        )
    return diagnostics


def resolve_target_map_lidar(sample: dict[str, Any]) -> np.ndarray:
    """Return the preprocessed map<-LiDAR pose used by target residuals."""

    return np.asarray(
        sample.get("T_map_lidar_target", sample["T_map_lidar"]), dtype=float
    ).reshape(4, 4)


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


def resolve_lidar_camera_initial(
    samples: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, float]]:
    """Construct one robust initial calibration from all sample priors."""

    priors = np.asarray(
        [sample["T_lidar_camera_prior"] for sample in samples], dtype=float
    )
    T_initial = np.eye(4, dtype=float)
    T_initial[:3, 3] = np.median(priors[:, :3, 3], axis=0)
    rotation_mean = np.mean(priors[:, :3, :3], axis=0)
    U, _, Vt = np.linalg.svd(rotation_mean)
    rotation_initial = U @ Vt
    if np.linalg.det(rotation_initial) < 0:
        U[:, -1] *= -1.0
        rotation_initial = U @ Vt
    T_initial[:3, :3] = rotation_initial
    translation = np.asarray(
        [np.linalg.norm(T[:3, 3] - T_initial[:3, 3]) for T in priors]
    )
    rotation = np.asarray(
        [rotation_error_deg(T[:3, :3], T_initial[:3, :3]) for T in priors]
    )
    summary = {
        "lidar_camera_prior_translation_spread_mm_median": float(
            np.median(translation)
        ),
        "lidar_camera_prior_translation_spread_mm_max": float(
            np.max(translation)
        ),
        "lidar_camera_prior_rotation_spread_deg_median": float(
            np.median(rotation)
        ),
        "lidar_camera_prior_rotation_spread_deg_max": float(np.max(rotation)),
    }
    print(
        "Constructed one T_lidar_camera_initial from",
        len(samples),
        "sample priors; spread around that initialization:",
        json.dumps(summary),
    )
    return T_initial, summary


def metadata_camera_name(metadata_path: str | Path) -> str:
    path = Path(metadata_path)
    return path.parent.parent.name if path.parent.name == "metadata" else path.parent.name


def validate_single_camera(samples: list[dict[str, Any]]) -> str:
    cameras = sorted(
        {
            metadata_camera_name(sample["metadata_path"])
            for sample in samples
            if sample.get("metadata_path")
        }
    )
    if len(cameras) != 1:
        raise ValueError(
            "Camera-LiDAR calibration must be run separately for each camera; "
            f"found camera groups {cameras or ['unknown']}."
        )
    return cameras[0]


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
    gigapose_pose_source: str,
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
    target_lidar_z_mode: str,
    allow_missing_corrected_lidar_z: bool,
    timestamp_alignment: str,
    timestamp_max_bracket_gap_ms: float,
    timestamp_fallback: str,
) -> list[dict[str, Any]]:
    rows = []
    malformed_count = 0
    malformed_examples: list[str] = []
    with path.open(newline="") as f:
        for row_number, row in enumerate(csv.DictReader(f), start=2):
            try:
                T_giga_raw = text_to_matrix(row["T_gigapose_cam_obj"])
                if gigapose_pose_source == "aligned_csv":
                    value = row.get("T_gigapose_aligned_epnp_obj")
                    if not value:
                        raise ValueError(
                            "Selected aligned pose source, but this row has no "
                            "T_gigapose_aligned_epnp_obj value."
                        )
                    T_giga = text_to_matrix(value)
                elif gigapose_pose_source == "raw_csv":
                    T_giga = T_giga_raw
                elif gigapose_pose_source == "raw_csv_plus_external_transform":
                    T_giga = apply_gigapose_frame_transform(
                        T_giga_raw,
                        gigapose_frame_transform,
                        gigapose_frame_transform_side,
                    )
                else:
                    raise ValueError(
                        f"Unknown resolved GigaPose pose source: {gigapose_pose_source}"
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
                if (
                    gigapose_pose_source == "aligned_csv"
                    and not row.get("T_gigapose_aligned_epnp_obj")
                ):
                    raise ValueError(
                        f"{path}:{row_number} is missing the aligned GigaPose "
                        "pose required by this run. Refusing to mix aligned and "
                        "raw pose conventions."
                    ) from exc
                malformed_count += 1
                if len(malformed_examples) < 5:
                    malformed_examples.append(f"row {row_number}: {exc}")
                continue
            rows.append(
                {
                    **row,
                    "T_gigapose_cam_obj": T_giga,
                    "T_gigapose_cam_obj_raw": T_giga_raw,
                    "gigapose_pose_source": gigapose_pose_source,
                    "T_target_obj": T_target,
                    "target_pose_path": str(label_path),
                    "source_csv_epnp_label_path": row.get("epnp_label_path", ""),
                    "target_pose_key": epnp_map_pose_key or "selected_csv:T_epnp_obj",
                    "camera_pose_key": epnp_camera_pose_key if use_epnp_label_extrinsics else "",
                    "label_data": label_data,
                    **metadata_values,
                }
            )
            if max_samples is not None and len(rows) >= max_samples:
                break
    if malformed_count:
        print(
            f"Skipped {malformed_count} malformed selected samples. "
            f"First {len(malformed_examples)} errors:"
        )
        for example in malformed_examples:
            print(f"  {example}")
    if not rows:
        if epnp_map_pose_key:
            example_text = (
                f" First error: {malformed_examples[0]}"
                if malformed_examples
                else ""
            )
            raise ValueError(
                f"No valid selected samples found in {path}. Check that each "
                f"epnp_label_path exists and contains {epnp_map_pose_key}."
                f"{example_text}"
            )
        raise ValueError(f"No valid selected samples found in {path}")
    if use_sample_metadata:
        target_preprocessing_summary = prepare_target_map_lidar_transforms(
            rows,
            target_lidar_z_mode,
            allow_missing_corrected_lidar_z,
            timestamp_alignment,
            timestamp_max_bracket_gap_ms,
            timestamp_fallback,
        )
        for sample in rows:
            sample["target_preprocessing_summary"] = (
                target_preprocessing_summary
            )
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

    residuals.extend(
        correction_prior_residuals(
            xi,
            translation_sigma_mm,
            translation_prior_weight,
            rotation_prior_weight,
        )
    )
    return np.asarray(residuals, dtype=float)


def residual_vector_sample_metadata(
    xi: np.ndarray,
    T_lidar_camera_initial: np.ndarray,
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
    """Optimize one fixed LiDAR<-camera transform from all samples.

    The optimized transform is

        T_lidar_camera(xi)
            = exp(xi) @ T_lidar_camera_initial

    For every sample, both object poses are expressed in the LiDAR frame:

        GigaPose:
            T_lidar_object_pred
                = T_lidar_camera(xi) @ T_camera_object_gigapose

        EPnP:
            T_lidar_object_target
                = inv(T_map_lidar) @ T_map_object_epnp

    The slightly different metadata priors contribute to the robust common
    initialization; the optimization variable is one calibration.
    """

    T_delta = se3_exp(xi)
    T_lidar_camera = T_delta @ T_lidar_camera_initial

    residuals: list[float] = []
    rotation_sigma_rad = math.radians(rotation_sigma_deg)
    t_idx = translation_component_indices(translation_residual_components)

    for sample in samples:
        T_map_lidar = resolve_target_map_lidar(sample)
        T_map_object = sample["T_target_obj"]
        T_camera_object_gigapose = sample["T_gigapose_cam_obj"]

        # EPnP/reference branch:
        # object -> map -> LiDAR
        T_lidar_object_target = (
            np.linalg.inv(T_map_lidar) @ T_map_object
        )

        # GigaPose/prediction branch:
        # object -> camera -> LiDAR
        T_lidar_object_pred = (
            T_lidar_camera @ T_camera_object_gigapose
        )

        translation_residual = (
            T_lidar_object_pred[:3, 3]
            - T_lidar_object_target[:3, 3]
        )[t_idx] / translation_sigma_mm

        rotation_residual = so3_log(
            T_lidar_object_pred[:3, :3]
            @ T_lidar_object_target[:3, :3].T
        ) / rotation_sigma_rad

        residuals.extend(translation_residual.tolist())
        residuals.extend(rotation_residual.tolist())

        if image_center_weight > 0:
            # The image-center term still needs a camera-frame pose. Construct
            # the current map<-camera transform using the optimized fixed
            # LiDAR<-camera extrinsic and this frame's map<-LiDAR localization.
            T_map_camera = T_map_lidar @ T_lidar_camera

            T_camera_object_from_map = (
                np.linalg.inv(T_map_camera)
                @ sample.get("T_target_obj_image", T_map_object)
            )

            residuals.extend(
                image_center_residual(
                    T_camera_object_from_map,
                    T_camera_object_gigapose,
                    sample,
                    image_center_weight,
                    image_center_sigma_px,
                    projection_model,
                ).tolist()
            )

    # These terms are zero by default. Nonzero CLI weights explicitly bias the
    # solution toward the initial t_lidar_camera_prior.
    residuals.extend(
        correction_prior_residuals(
            xi,
            translation_sigma_mm,
            translation_prior_weight,
            rotation_prior_weight,
        )
    )

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
            T_cam_obj_from_map = np.linalg.inv(T_map_cam) @ T_gt
            residuals.extend(
                image_center_residual(
                    T_cam_obj_from_map,
                    sample["T_gigapose_cam_obj"],
                    sample,
                    image_center_weight,
                    image_center_sigma_px,
                    projection_model,
                ).tolist()
            )

    residuals.extend(
        correction_prior_residuals(
            xi,
            translation_sigma_mm,
            translation_prior_weight,
            rotation_prior_weight,
        )
    )
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
                "gigapose_pose_source": sample.get("gigapose_pose_source", ""),
                "translation_error_mm": float(np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3])),
                "rotation_error_deg": rotation_error_deg(T_pred[:3, :3], T_gt[:3, :3]),
            }
        )
    return rows


def compute_sample_metadata_errors(
    T_lidar_camera: np.ndarray,
    samples: list[dict[str, Any]],
    projection_model: str = "pinhole",
) -> list[dict[str, Any]]:
    """Evaluate one fixed LiDAR<-camera calibration over all samples."""

    rows: list[dict[str, Any]] = []

    for sample in samples:
        T_map_lidar = resolve_target_map_lidar(sample)
        T_map_object = sample["T_target_obj"]
        T_camera_object_gigapose = sample["T_gigapose_cam_obj"]

        # Reference pose expressed in LiDAR coordinates.
        T_lidar_object_target = (
            np.linalg.inv(T_map_lidar) @ T_map_object
        )

        # GigaPose pose expressed in LiDAR coordinates using the candidate
        # camera-LiDAR extrinsic.
        T_lidar_object_pred = (
            T_lidar_camera @ T_camera_object_gigapose
        )

        # translation_error_mm = float(
        #     np.linalg.norm(
        #         T_lidar_object_pred[:3, 3]
        #         - T_lidar_object_target[:3, 3]
        #     )
        # )
        translation_difference = (
            T_lidar_object_pred[:3, 3]
            - T_lidar_object_target[:3, 3]
        )

        translation_error_xy_mm = float(
            np.linalg.norm(translation_difference[:2])
        )

        translation_error_z_mm = float(
            abs(translation_difference[2])
        )

        translation_error_xyz_mm = float(
            np.linalg.norm(translation_difference)
        )
        

        rotation_error = rotation_error_deg(
            T_lidar_object_pred[:3, :3],
            T_lidar_object_target[:3, :3],
        )

        image_center_error_px = float("nan")

        if sample.get("K") is not None:
            T_map_camera = T_map_lidar @ T_lidar_camera

            T_camera_object_from_map = (
                np.linalg.inv(T_map_camera)
                @ sample.get("T_target_obj_image", T_map_object)
            )

            uv_map, map_valid = project_origin(
                T_camera_object_from_map,
                sample["K"],
                sample.get("D"),
                sample.get("distortion_model", "pinhole"),
                projection_model,
            )

            uv_gigapose, gigapose_valid = project_origin(
                T_camera_object_gigapose,
                sample["K"],
                sample.get("D"),
                sample.get("distortion_model", "pinhole"),
                projection_model,
            )

            if (
                map_valid
                and gigapose_valid
                and np.isfinite(uv_map).all()
                and np.isfinite(uv_gigapose).all()
            ):
                image_center_error_px = float(
                    np.linalg.norm(uv_map - uv_gigapose)
                )

        rows.append(
            {
                "match_key": sample.get("match_key", ""),
                "scene_id": sample.get("scene_id", ""),
                "im_id": sample.get("im_id", ""),
                "instance_id": sample.get("instance_id", ""),
                "score": sample.get("score", ""),
                "gigapose_pose_source": sample.get(
                    "gigapose_pose_source", ""
                ),
                "metadata_path": sample.get("metadata_path", ""),
                "target_pose_path": sample.get("target_pose_path", ""),
                "target_pose_key": sample.get("target_pose_key", ""),
                "translation_error_mm": translation_error_xyz_mm,
                "translation_error_xy_mm": translation_error_xy_mm,
                "translation_error_z_mm": translation_error_z_mm,
                "rotation_error_deg": rotation_error,
                "image_center_error_px": image_center_error_px,
                "T_lidar_object_pred": matrix_to_text(
                    T_lidar_object_pred
                ),
                "T_lidar_object_target": matrix_to_text(
                    T_lidar_object_target
                ),
            }
        )

    return rows


def optimized_sample_extrinsic_rows(
    T_lidar_camera_optimized: np.ndarray,
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return all reusable per-sample extrinsics in both directions."""

    rows: list[dict[str, Any]] = []
    for sample in samples:
        T_map_lidar_raw = np.asarray(
            sample.get("T_map_lidar_raw", sample["T_map_lidar"]), dtype=float
        ).reshape(4, 4)
        T_map_lidar = resolve_target_map_lidar(sample)
        T_lidar_camera_prior = np.asarray(
            sample["T_lidar_camera_prior"], dtype=float
        ).reshape(4, 4)
        T_camera_lidar_optimized = np.linalg.inv(T_lidar_camera_optimized)
        T_map_camera_optimized = T_map_lidar @ T_lidar_camera_optimized
        T_camera_map_optimized = np.linalg.inv(T_map_camera_optimized)
        rows.append(
            {
                "match_key": sample.get("match_key", ""),
                "scene_id": sample.get("scene_id", ""),
                "im_id": sample.get("im_id", ""),
                "instance_id": sample.get("instance_id", ""),
                "metadata_path": sample.get("metadata_path", ""),
                "target_pose_path": sample.get("target_pose_path", ""),
                "T_map_lidar": matrix_to_text(T_map_lidar_raw),
                "T_map_lidar_raw": matrix_to_text(T_map_lidar_raw),
                "T_map_lidar_time_aligned": matrix_to_text(
                    np.asarray(
                        sample.get("T_map_lidar_time_aligned", T_map_lidar_raw),
                        dtype=float,
                    ).reshape(4, 4)
                ),
                "T_map_lidar_target": matrix_to_text(T_map_lidar),
                "T_lidar_map": matrix_to_text(np.linalg.inv(T_map_lidar)),
                "corrected_lidar_map_z_mm": sample.get(
                    "corrected_lidar_map_z_mm", ""
                ),
                "image_timestamp_ns": sample.get("image_timestamp_ns", ""),
                "lidar_timestamp_ns": sample.get("lidar_timestamp_ns", ""),
                "timestamp_alignment_status": sample.get(
                    "timestamp_alignment_status", ""
                ),
                "timestamp_alignment_translation_shift_mm": sample.get(
                    "timestamp_alignment_translation_shift_mm", ""
                ),
                "timestamp_alignment_rotation_shift_deg": sample.get(
                    "timestamp_alignment_rotation_shift_deg", ""
                ),
                "target_lidar_z_replacement_mm": sample.get(
                    "target_lidar_z_replacement_mm", ""
                ),
                "timestamp_interpolation_left_ns": sample.get(
                    "timestamp_interpolation_left_ns", ""
                ),
                "timestamp_interpolation_right_ns": sample.get(
                    "timestamp_interpolation_right_ns", ""
                ),
                "timestamp_interpolation_alpha": sample.get(
                    "timestamp_interpolation_alpha", ""
                ),
                "T_lidar_camera_prior": matrix_to_text(T_lidar_camera_prior),
                "T_lidar_camera_optimized": matrix_to_text(
                    T_lidar_camera_optimized
                ),
                "T_camera_lidar_optimized": matrix_to_text(
                    T_camera_lidar_optimized
                ),
                "T_map_camera_optimized": matrix_to_text(T_map_camera_optimized),
                "T_camera_map_optimized": matrix_to_text(T_camera_map_optimized),
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
                "gigapose_pose_source": sample.get("gigapose_pose_source", ""),
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
    for key in (
    "translation_error_mm",
    "translation_error_xy_mm",
    "translation_error_z_mm",
    "rotation_error_deg",
    "image_center_error_px",
):
        if not rows or key not in rows[0]:
            continue
        vals = np.asarray([float(row[key]) for row in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        out[f"{prefix}_{key}_mean"] = float(np.mean(vals))
        out[f"{prefix}_{key}_median"] = float(np.median(vals))
        out[f"{prefix}_{key}_p90"] = float(np.percentile(vals, 90))
        out[f"{prefix}_{key}_valid_samples"] = int(vals.size)
    return out


def summarize_gigapose_pose_inputs(samples: list[dict[str, Any]]) -> dict[str, float]:
    raw_selected_translation = []
    raw_selected_rotation = []
    selected_epnp_translation = []
    selected_epnp_rotation = []
    raw_epnp_translation = []
    raw_epnp_rotation = []

    for sample in samples:
        raw = sample["T_gigapose_cam_obj_raw"]
        selected = sample["T_gigapose_cam_obj"]
        raw_selected_translation.append(
            float(np.linalg.norm(raw[:3, 3] - selected[:3, 3]))
        )
        raw_selected_rotation.append(
            rotation_error_deg(raw[:3, :3], selected[:3, :3])
        )
        epnp = sample.get("T_epnp_camera_obj")
        if epnp is not None:
            selected_epnp_translation.append(
                float(np.linalg.norm(selected[:3, 3] - epnp[:3, 3]))
            )
            selected_epnp_rotation.append(
                rotation_error_deg(selected[:3, :3], epnp[:3, :3])
            )
            raw_epnp_translation.append(
                float(np.linalg.norm(raw[:3, 3] - epnp[:3, 3]))
            )
            raw_epnp_rotation.append(
                rotation_error_deg(raw[:3, :3], epnp[:3, :3])
            )

    def add_stats(
        output: dict[str, float], name: str, values: list[float]
    ) -> None:
        if not values:
            return
        array = np.asarray(values, dtype=float)
        output[f"{name}_mean"] = float(np.mean(array))
        output[f"{name}_median"] = float(np.median(array))
        output[f"{name}_p90"] = float(np.percentile(array, 90))

    output: dict[str, float] = {}
    add_stats(
        output,
        "input_raw_to_selected_translation_mm",
        raw_selected_translation,
    )
    add_stats(
        output,
        "input_raw_to_selected_rotation_deg",
        raw_selected_rotation,
    )
    add_stats(
        output,
        "input_selected_to_epnp_translation_mm",
        selected_epnp_translation,
    )
    add_stats(
        output,
        "input_selected_to_epnp_rotation_deg",
        selected_epnp_rotation,
    )
    add_stats(
        output,
        "input_raw_to_epnp_translation_mm",
        raw_epnp_translation,
    )
    add_stats(
        output,
        "input_raw_to_epnp_rotation_deg",
        raw_epnp_rotation,
    )
    return output


def validate_optimization_args(args: argparse.Namespace) -> None:
    if args.translation_sigma_mm <= 0:
        raise ValueError("--translation-sigma-mm must be positive")
    if args.rotation_sigma_deg <= 0:
        raise ValueError("--rotation-sigma-deg must be positive")
    if args.image_center_sigma_px <= 0:
        raise ValueError("--image-center-sigma-px must be positive")
    for name in (
        "image_center_weight",
        "translation_prior_weight",
        "rotation_prior_weight",
    ):
        if float(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive when provided")
    if args.timestamp_max_bracket_gap_ms <= 0:
        raise ValueError("--timestamp-max-bracket-gap-ms must be positive")
    if not args.use_sample_metadata and args.timestamp_alignment != "raw":
        raise ValueError(
            "Timestamp preprocessing requires "
            "--use-sample-metadata."
        )


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
    validate_optimization_args(args)
    if args.use_sample_metadata and args.use_epnp_label_extrinsics:
        raise SystemExit(
            "Use either --use-sample-metadata or "
            "--use-epnp-label-extrinsics, not both."
        )
    if (
        args.use_sample_metadata
        and args.target_lidar_z_mode == "epnp_corrected"
        and args.image_center_weight > 0
        and args.image_center_map_z_mode != "raw"
    ):
        raise ValueError(
            "Use --image-center-map-z-mode raw with "
            "--target-lidar-z-mode epnp_corrected. The corrected LiDAR z and "
            "raw mesh-z object pose are already in the same vertical frame; "
            "applying a second image-center z conversion would be inconsistent."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    (
        gigapose_frame_transform,
        gigapose_frame_transform_side,
        gigapose_frame_transform_path,
    ) = load_gigapose_frame_transform(
        args.gigapose_frame_transform_json,
        args.gigapose_frame_transform_side,
    )
    gigapose_pose_source, aligned_column_present = resolve_gigapose_pose_source(
        args.selected_samples,
        args.gigapose_pose_source,
        gigapose_frame_transform,
    )
    gigapose_pose_column = {
        "aligned_csv": "T_gigapose_aligned_epnp_obj",
        "raw_csv": "T_gigapose_cam_obj",
        "raw_csv_plus_external_transform": (
            "T_gigapose_cam_obj + external frame transform"
        ),
    }[gigapose_pose_source]
    print(
        "GigaPose optimization pose source:",
        gigapose_pose_source,
        "(aligned CSV column present:",
        aligned_column_present,
        ")",
    )
    samples = load_selected_samples(
        args.selected_samples,
        args.max_samples,
        gigapose_pose_source,
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
        args.target_lidar_z_mode,
        args.allow_missing_corrected_lidar_z,
        args.timestamp_alignment,
        args.timestamp_max_bracket_gap_ms,
        args.timestamp_fallback,
    )
    target_preprocessing_summary = (
        samples[0].get("target_preprocessing_summary", {}) if samples else {}
    )
    calibrated_camera = validate_single_camera(samples) if args.use_sample_metadata else None
    if args.use_sample_metadata:
        T_metadata_initial, prior_variation_summary = resolve_lidar_camera_initial(
            samples
        )
    else:
        T_metadata_initial, prior_variation_summary = None, {}

    intrinsics_samples = sum(sample.get("K") is not None for sample in samples)
    valid_gigapose_depth_samples = sum(
        float(sample["T_gigapose_cam_obj"][2, 3]) > 1e-6
        for sample in samples
    )
    if args.image_center_weight > 0 and (
        args.use_sample_metadata or args.use_epnp_label_extrinsics
    ):
        if intrinsics_samples == 0:
            raise ValueError(
                "--image-center-weight is positive, but none of the selected "
                "samples contains camera intrinsics."
            )
        if intrinsics_samples < len(samples):
            print(
                "WARNING:",
                len(samples) - intrinsics_samples,
                "samples have no camera intrinsics and will not contribute an "
                "image-center residual.",
            )
        if valid_gigapose_depth_samples < len(samples):
            print(
                "WARNING:",
                len(samples) - valid_gigapose_depth_samples,
                "selected GigaPose poses have non-positive center depth.",
            )
    elif args.image_center_weight > 0:
        print(
            "WARNING: --image-center-weight is ignored in static "
            "T_map_cam mode."
        )

    if args.use_sample_metadata:
        # All priors form one robust initial guess. Every pose pair then
        # constrains the same fixed physical camera-LiDAR calibration.
        assert T_metadata_initial is not None
        T_initial = T_metadata_initial

        before_rows = compute_sample_metadata_errors(
            T_initial,
            samples,
            args.projection_model,
        )

        residual_fn = residual_vector_sample_metadata
        residual_args = (
            T_initial,
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
        after_rows = compute_sample_metadata_errors(
            T_optimized,
            samples,
            args.projection_model,
        )
    elif args.use_epnp_label_extrinsics:
        after_rows = compute_sample_epnp_label_extrinsic_errors(
            T_delta, samples, args.projection_model
        )
    else:
        after_rows = compute_sample_errors(T_optimized, samples)

    if args.use_sample_metadata:
        optimization_mode = "fixed_lidar_camera_from_robust_metadata_initial"
    elif args.use_epnp_label_extrinsics:
        optimization_mode = "epnp_label_camera_extrinsics"
    else:
        optimization_mode = "static_T_map_cam"

    write_csv(args.output_dir / "errors_before_optimization.csv", before_rows)
    write_csv(args.output_dir / "errors_after_optimization.csv", after_rows)
    per_sample_extrinsics_path: Path | None = None
    if args.use_sample_metadata:
        per_sample_extrinsics_path = (
            args.output_dir / "optimized_extrinsics_per_sample.csv"
        )
        write_csv(
            per_sample_extrinsics_path,
            optimized_sample_extrinsic_rows(T_optimized, samples),
        )

    before_summary = summarize_errors(before_rows, "before")
    after_summary = summarize_errors(after_rows, "after")
    input_pose_summary = summarize_gigapose_pose_inputs(samples)
    warnings: list[str] = []
    if aligned_column_present and gigapose_pose_source == "raw_csv":
        warnings.append(
            "The selected CSV contains aligned GigaPose poses, but raw poses "
            "were explicitly requested. This is usually wrong for EPnP-frame "
            "extrinsic optimization."
        )
    if not result.success:
        warnings.append(f"Optimizer did not report success: {result.message}")
    if args.use_sample_metadata:
        timestamp_offset_p90 = target_preprocessing_summary.get(
            "image_minus_lidar_offset_ms_p90_abs"
        )
        if (
            args.timestamp_alignment == "raw"
            and timestamp_offset_p90 is not None
            and float(timestamp_offset_p90) > 5.0
        ):
            warnings.append(
                "Image/LiDAR timestamp offset p90 is "
                f"{float(timestamp_offset_p90):.3f} ms, but timestamp alignment "
                "is raw; the extrinsic may absorb vehicle motion."
            )
        fallback_count = int(
            target_preprocessing_summary.get(
                "samples_timestamp_raw_fallback", 0
            )
        )
        if fallback_count:
            warnings.append(
                f"{fallback_count} samples used raw timestamp fallback; inspect "
                "target_map_lidar_preprocessing.timestamp_failures."
            )
    for metric in (
        "translation_error_mm_median",
        "rotation_error_deg_median",
        "image_center_error_px_median",
    ):
        before_value = before_summary.get(f"before_{metric}")
        after_value = after_summary.get(f"after_{metric}")
        if (
            before_value is not None
            and after_value is not None
            and after_value > before_value
        ):
            warnings.append(
                f"{metric} increased from {before_value:.6g} to "
                f"{after_value:.6g}."
            )

    extrinsics = {
        "description": "Optimized map-to-camera extrinsic correction from selected GigaPose/EPnPv2 samples.",
        "translation_unit": "mm",
        "gigapose_pose_source": gigapose_pose_source,
        "gigapose_pose_column": gigapose_pose_column,
        "aligned_gigapose_column_present": aligned_column_present,
        "loss_weight_semantics": (
            "CLI weights are squared-loss coefficients; residuals are multiplied "
            "by sqrt(weight)."
        ),
        "target_pose_source": args.epnp_map_pose_key or "selected_csv:T_epnp_obj",
        "camera_pose_source": args.epnp_camera_pose_key if args.use_epnp_label_extrinsics else None,
        "epnp_label_root_override": (
            str(args.epnp_label_root_override) if args.epnp_label_root_override is not None else None
        ),
        "epnp_label_dir_name": args.epnp_label_dir_name,
        "optimization_mode": optimization_mode,
        "calibrated_camera": calibrated_camera,
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
        "T_map_cam_initial": (
            T_initial.tolist()
            if not args.use_sample_metadata
            and not args.use_epnp_label_extrinsics
            else None
        ),
        "T_correction_left_multiply": T_delta.tolist(),
        "T_map_cam_optimized": (
            T_optimized.tolist()
            if not args.use_sample_metadata
            and not args.use_epnp_label_extrinsics
            else None
        ),
        "metadata_prior_policy": (
            "robust_common_initial_then_optimize_one_fixed_transform"
            if args.use_sample_metadata
            else None
        ),
        "metadata_prior_variation": prior_variation_summary,
        "target_map_lidar_preprocessing": target_preprocessing_summary,
        "target_lidar_z_mode": args.target_lidar_z_mode,
        "allow_missing_corrected_lidar_z": (
            args.allow_missing_corrected_lidar_z
        ),
        "timestamp_alignment": args.timestamp_alignment,
        "timestamp_max_bracket_gap_ms": args.timestamp_max_bracket_gap_ms,
        "timestamp_fallback": args.timestamp_fallback,
        "T_lidar_camera_initial": (
            T_initial.tolist() if args.use_sample_metadata else None
        ),
        "T_lidar_camera_correction_left_multiply": (
            T_delta.tolist()
            if args.use_sample_metadata
            else None
        ),
        "T_lidar_camera_optimized": (
            T_optimized.tolist() if args.use_sample_metadata else None
        ),
        "T_camera_lidar_optimized": (
            np.linalg.inv(T_optimized).tolist()
            if args.use_sample_metadata
            else None
        ),
        "optimized_extrinsics_per_sample_csv": (
            str(per_sample_extrinsics_path)
            if per_sample_extrinsics_path is not None
            else None
        ),
        "T_epnp_label_camera_correction_left_multiply": (
            T_delta.tolist()
            if args.use_epnp_label_extrinsics
            else None
        ),
        "correction_rotation_rotvec_rad": xi[:3].tolist(),
        "correction_rotation_rpy_like_vector_rad": xi[:3].tolist(),
        "correction_translation_mm": xi[3:6].tolist(),
        "image_center_weight": args.image_center_weight,
        "image_center_sigma_px": args.image_center_sigma_px,
        "image_center_map_z_mode": args.image_center_map_z_mode,
        "session_z_scale": args.session_z_scale,
        "projection_model": args.projection_model,
        "note": (
            "Sample-metadata mode first constructs T_map_lidar_target by "
            "applying the saved vertical-frame and timestamp-alignment policy. "
            "It robustly combines all metadata priors into "
            "T_lidar_camera_initial, then estimates one fixed physical "
            "T_lidar_camera_optimized for the calibrated camera. For any frame "
            "i, use T_map_camera_i = T_map_lidar_target_i @ "
            "T_lidar_camera_optimized. The absolute optimized transform is "
            "reusable across dates for the same unchanged camera calibration."
            if args.use_sample_metadata
            else (
                "EPnP-label-extrinsics mode: for each frame, derive "
                "T_map_cam_initial_i = T_map_object_raw_i @ "
                "inv(T_camera_object_i), then apply "
                "T_map_cam_optimized_i = "
                "T_epnp_label_camera_correction_left_multiply @ "
                "T_map_cam_initial_i. GigaPose poses are first put into the "
                "EPnP object-frame convention using gigapose_pose_source."
                if args.use_epnp_label_extrinsics
                else (
                    "Use T_map_cam_optimized as camera-to-map transform "
                    "if your pipeline expects T_map_cam."
                )
            )
        ),
    }
    (args.output_dir / "optimized_extrinsics.json").write_text(json.dumps(extrinsics, indent=2))

    report = {
        "selected_samples": len(samples),
        "optimizer_success": bool(result.success),
        "optimizer_message": result.message,
        "optimizer_cost": float(result.cost),
        "optimizer_nfev": int(result.nfev),
        "optimizer_optimality": float(result.optimality),
        "robust_loss": args.robust_loss,
        "gigapose_pose_source": gigapose_pose_source,
        "gigapose_pose_column": gigapose_pose_column,
        "aligned_gigapose_column_present": aligned_column_present,
        "loss_weight_semantics": (
            "CLI weights are squared-loss coefficients; residuals are multiplied "
            "by sqrt(weight)."
        ),
        "target_pose_source": args.epnp_map_pose_key or "selected_csv:T_epnp_obj",
        "camera_pose_source": args.epnp_camera_pose_key if args.use_epnp_label_extrinsics else None,
        "epnp_label_root_override": (
            str(args.epnp_label_root_override) if args.epnp_label_root_override is not None else None
        ),
        "epnp_label_dir_name": args.epnp_label_dir_name,
        "optimization_mode": optimization_mode,
        "calibrated_camera": calibrated_camera,
        "metadata_prior_policy": (
            "robust_common_initial_then_optimize_one_fixed_transform"
            if args.use_sample_metadata
            else None
        ),
        "target_map_lidar_preprocessing": target_preprocessing_summary,
        "target_lidar_z_mode": args.target_lidar_z_mode,
        "allow_missing_corrected_lidar_z": (
            args.allow_missing_corrected_lidar_z
        ),
        "timestamp_alignment": args.timestamp_alignment,
        "timestamp_max_bracket_gap_ms": args.timestamp_max_bracket_gap_ms,
        "timestamp_fallback": args.timestamp_fallback,
        "optimized_extrinsics_per_sample_csv": (
            str(per_sample_extrinsics_path)
            if per_sample_extrinsics_path is not None
            else None
        ),
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
        "samples_with_intrinsics": intrinsics_samples,
        "samples_with_positive_gigapose_center_depth": (
            valid_gigapose_depth_samples
        ),
        "correction_translation_norm_mm": float(np.linalg.norm(xi[3:6])),
        "correction_rotation_norm_deg": float(np.degrees(np.linalg.norm(xi[:3]))),
        **input_pose_summary,
        **prior_variation_summary,
        **before_summary,
        **after_summary,
        "warnings": warnings,
    }
    (args.output_dir / "optimization_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"Wrote optimized extrinsics to {args.output_dir / 'optimized_extrinsics.json'}")


if __name__ == "__main__":
    main()
