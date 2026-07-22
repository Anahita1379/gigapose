"""Optimize camera-LiDAR extrinsics with an explicit raw-to-centered CAD transform.

This is a separate, focused alternative to
``fine_tuning.optimize_camera_lidar_extrinsics``.  The original optimizer is
left unchanged.

Pose convention
---------------
``T_A_B`` maps coordinates from frame B into frame A.  For every selected
sample this optimizer constructs

    T_lidar_object_centered_target
        = inv(T_map_lidar)
          @ T_map_object_raw
          @ T_object_raw_object_centered

and compares it to the raw GigaPose centered-object pose expressed in LiDAR:

    T_lidar_object_centered_pred
        = T_lidar_camera
          @ T_camera_object_centered_gigapose

The optimized variable is one fixed physical ``T_lidar_camera`` for one camera.
The inverse ``T_camera_lidar`` is saved as well.

Unlike the older experimental optimizer, this script deliberately:

* uses the stored per-sample ``T_map_lidar`` directly;
* performs no image/LiDAR timestamp interpolation;
* performs no map-Z replacement;
* uses raw ``T_gigapose_cam_obj`` rather than an empirical aligned pose; and
* explicitly converts ``T_map_object_raw`` to the centered object convention.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from fine_tuning import optimize_camera_lidar_extrinsics as base


DEFAULT_RAW_OBJECT_CENTER_M = np.asarray(
    [-0.2411941141, 0.0009010172, 0.3329219520], dtype=float
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize one camera-LiDAR transform using raw map-object labels "
            "and raw GigaPose centered-object predictions."
        )
    )
    parser.add_argument("--selected-samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--use-sample-metadata",
        action="store_true",
        help=(
            "Accepted for command compatibility. This mode is always enabled "
            "because the optimizer requires t_map_lidar and "
            "t_lidar_camera_prior from each sample metadata YAML."
        ),
    )
    parser.add_argument(
        "--metadata-path-field",
        default="sample_metadata_path",
        help="CSV field containing the source metadata YAML path.",
    )
    parser.add_argument(
        "--epnp-map-pose-key",
        default="T_map_object_raw",
        help="Raw map-frame object pose key in each EPnP label JSON.",
    )
    parser.add_argument(
        "--epnp-map-pose-unit",
        choices=("auto", "m", "mm"),
        default="m",
    )
    parser.add_argument("--epnp-label-root-override", type=Path, default=None)
    parser.add_argument("--epnp-label-dir-name", default=None)
    parser.add_argument(
        "--raw-object-center-m",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=DEFAULT_RAW_OBJECT_CENTER_M.tolist(),
        help=(
            "Coordinates, in metres, of the centered-object origin expressed "
            "in the raw object frame. Default: -0.2411941141 0.0009010172 "
            "0.3329219520."
        ),
    )
    parser.add_argument(
        "--gigapose-pose-source",
        choices=("raw",),
        default="raw",
        help=(
            "Only raw is accepted. The explicit CAD origin transform replaces "
            "the empirical aligned-pose transform."
        ),
    )
    parser.add_argument(
        "--timestamp-alignment",
        choices=("raw",),
        default="raw",
        help="Only raw is accepted: use each stored t_map_lidar directly.",
    )
    parser.add_argument(
        "--target-lidar-z-mode",
        choices=("raw",),
        default="raw",
        help="Only raw is accepted: do not replace the map-Z of t_map_lidar.",
    )
    parser.add_argument(
        "--translation-residual-components",
        choices=("xyz", "xy", "xz", "yz", "x", "y", "z"),
        default="xyz",
    )
    parser.add_argument("--translation-sigma-mm", type=float, default=1000.0)
    parser.add_argument("--rotation-sigma-deg", type=float, default=10.0)
    parser.add_argument(
        "--image-center-weight",
        type=float,
        default=0.0,
        help=(
            "Optional squared-loss coefficient for agreement between the "
            "map-derived centered pose origin and the GigaPose centered origin."
        ),
    )
    parser.add_argument("--image-center-sigma-px", type=float, default=50.0)
    parser.add_argument(
        "--projection-model",
        choices=("metadata", "pinhole"),
        default="metadata",
    )
    parser.add_argument("--translation-prior-weight", type=float, default=0.0)
    parser.add_argument("--rotation-prior-weight", type=float, default=0.0)
    parser.add_argument(
        "--robust-loss",
        choices=("linear", "soft_l1", "huber", "cauchy", "arctan"),
        default="soft_l1",
    )
    parser.add_argument("--max-nfev", type=int, default=500)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.max_nfev <= 0:
        raise ValueError("--max-nfev must be positive")
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
    center = np.asarray(args.raw_object_center_m, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("--raw-object-center-m must contain three finite values")


def raw_from_centered_transform(center_m: np.ndarray) -> np.ndarray:
    """Return T_object_raw_object_centered in the optimizer's millimetres."""

    center_m = np.asarray(center_m, dtype=float).reshape(3)
    if not np.isfinite(center_m).all():
        raise ValueError("Raw object center must be finite")
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = center_m * 1000.0
    return transform


def apply_centered_object_convention(
    samples: list[dict[str, Any]],
    T_object_raw_object_centered: np.ndarray,
) -> None:
    """Convert every raw map-object target to the GigaPose centered convention."""

    conversion = np.asarray(T_object_raw_object_centered, dtype=float).reshape(4, 4)
    for sample in samples:
        T_map_object_raw = np.asarray(sample["T_target_obj"], dtype=float).reshape(4, 4)
        T_map_object_centered = T_map_object_raw @ conversion
        sample["T_map_object_raw"] = T_map_object_raw
        sample["T_map_object_centered"] = T_map_object_centered
        sample["T_object_raw_object_centered"] = conversion
        sample["T_target_obj"] = T_map_object_centered
        # The optional image-center residual must use the same centered origin.
        sample["T_target_obj_image"] = T_map_object_centered
        sample["target_pose_key"] = (
            f"{sample.get('target_pose_key', 'T_map_object_raw')} "
            "@ T_object_raw_object_centered"
        )


def residual_vector_centered(
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
    """Residual for one fixed LiDAR<-camera transform and centered targets."""

    T_lidar_camera = base.se3_exp(xi) @ T_lidar_camera_initial
    rotation_sigma_rad = math.radians(rotation_sigma_deg)
    translation_indices = base.translation_component_indices(
        translation_residual_components
    )
    residuals: list[float] = []

    for sample in samples:
        # Directly use the stored map<-LiDAR transform. No timestamp or Z rewrite.
        T_map_lidar = np.asarray(sample["T_map_lidar"], dtype=float).reshape(4, 4)
        T_map_object_centered = np.asarray(
            sample["T_map_object_centered"], dtype=float
        ).reshape(4, 4)
        T_camera_object_centered_gigapose = np.asarray(
            sample["T_gigapose_cam_obj"], dtype=float
        ).reshape(4, 4)

        T_lidar_object_centered_target = (
            np.linalg.inv(T_map_lidar) @ T_map_object_centered
        )
        T_lidar_object_centered_pred = (
            T_lidar_camera @ T_camera_object_centered_gigapose
        )

        translation_residual = (
            T_lidar_object_centered_pred[:3, 3] - T_lidar_object_centered_target[:3, 3]
        )[translation_indices] / translation_sigma_mm
        rotation_residual = (
            base.so3_log(
                T_lidar_object_centered_pred[:3, :3]
                @ T_lidar_object_centered_target[:3, :3].T
            )
            / rotation_sigma_rad
        )
        residuals.extend(translation_residual.tolist())
        residuals.extend(rotation_residual.tolist())

        if image_center_weight > 0 and sample.get("K") is not None:
            T_map_camera = T_map_lidar @ T_lidar_camera
            T_camera_object_centered_from_map = (
                np.linalg.inv(T_map_camera) @ T_map_object_centered
            )
            residuals.extend(
                base.image_center_residual(
                    T_camera_object_centered_from_map,
                    T_camera_object_centered_gigapose,
                    sample,
                    image_center_weight,
                    image_center_sigma_px,
                    projection_model,
                ).tolist()
            )

    residuals.extend(
        base.correction_prior_residuals(
            xi,
            translation_sigma_mm,
            translation_prior_weight,
            rotation_prior_weight,
        )
    )
    return np.asarray(residuals, dtype=float)


def centered_per_sample_rows(
    T_lidar_camera_optimized: np.ndarray,
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Save the exact raw, centered, target, prediction, and extrinsic poses."""

    T_camera_lidar_optimized = np.linalg.inv(T_lidar_camera_optimized)
    rows: list[dict[str, Any]] = []
    for sample in samples:
        T_map_lidar = np.asarray(sample["T_map_lidar"], dtype=float).reshape(4, 4)
        T_map_object_raw = np.asarray(sample["T_map_object_raw"], dtype=float)
        T_map_object_centered = np.asarray(sample["T_map_object_centered"], dtype=float)
        T_camera_object_gigapose = np.asarray(sample["T_gigapose_cam_obj"], dtype=float)
        T_lidar_object_target = np.linalg.inv(T_map_lidar) @ T_map_object_centered
        T_lidar_object_pred = T_lidar_camera_optimized @ T_camera_object_gigapose
        T_camera_object_from_map = T_camera_lidar_optimized @ T_lidar_object_target
        T_map_camera_optimized = T_map_lidar @ T_lidar_camera_optimized
        rows.append(
            {
                "match_key": sample.get("match_key", ""),
                "scene_id": sample.get("scene_id", ""),
                "im_id": sample.get("im_id", ""),
                "instance_id": sample.get("instance_id", ""),
                "score": sample.get("score", ""),
                "metadata_path": sample.get("metadata_path", ""),
                "epnp_label_path": sample.get("target_pose_path", ""),
                "T_map_lidar_used_directly": base.matrix_to_text(T_map_lidar),
                "T_map_object_raw": base.matrix_to_text(T_map_object_raw),
                "T_object_raw_object_centered": base.matrix_to_text(
                    sample["T_object_raw_object_centered"]
                ),
                "T_map_object_centered": base.matrix_to_text(T_map_object_centered),
                "T_camera_object_centered_gigapose": base.matrix_to_text(
                    T_camera_object_gigapose
                ),
                "T_lidar_object_centered_target": base.matrix_to_text(
                    T_lidar_object_target
                ),
                "T_lidar_object_centered_pred": base.matrix_to_text(
                    T_lidar_object_pred
                ),
                "T_camera_object_centered_from_map_optimized": base.matrix_to_text(
                    T_camera_object_from_map
                ),
                "T_lidar_camera_prior": base.matrix_to_text(
                    sample["T_lidar_camera_prior"]
                ),
                "T_lidar_camera_optimized": base.matrix_to_text(
                    T_lidar_camera_optimized
                ),
                "T_camera_lidar_optimized": base.matrix_to_text(
                    T_camera_lidar_optimized
                ),
                "T_map_camera_optimized": base.matrix_to_text(T_map_camera_optimized),
                "T_camera_map_optimized": base.matrix_to_text(
                    np.linalg.inv(T_map_camera_optimized)
                ),
            }
        )
    return rows


def metric_warnings(before: dict[str, float], after: dict[str, float]) -> list[str]:
    warnings: list[str] = []
    for metric in (
        "translation_error_mm_median",
        "translation_error_xy_mm_median",
        "translation_error_z_mm_median",
        "rotation_error_deg_median",
        "image_center_error_px_median",
    ):
        old = before.get(f"before_{metric}")
        new = after.get(f"after_{metric}")
        if old is not None and new is not None and new > old:
            warnings.append(f"{metric} increased from {old:.6g} to {new:.6g}.")
    return warnings


def main() -> None:
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise SystemExit(
            "This optimizer requires scipy. Install scipy in the GigaPose environment."
        ) from exc

    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_center_m = np.asarray(args.raw_object_center_m, dtype=float)
    T_object_raw_object_centered = raw_from_centered_transform(raw_center_m)

    # The base loader handles CSV/JSON/YAML parsing and unit conversion. Every
    # policy that previously altered t_map_lidar is fixed to the direct/raw path.
    samples = base.load_selected_samples(
        args.selected_samples,
        args.max_samples,
        "raw_csv",
        None,
        None,
        args.epnp_map_pose_key,
        args.epnp_map_pose_unit,
        True,
        False,
        "T_camera_object_centered",
        "auto",
        args.epnp_label_root_override,
        args.epnp_label_dir_name,
        args.metadata_path_field,
        "raw",
        1.0,
        "raw",
        False,
        "raw",
        1.0,
        "error",
        False,
        1.0,
        1.0,
    )
    calibrated_camera = base.validate_single_camera(samples)
    apply_centered_object_convention(samples, T_object_raw_object_centered)

    T_lidar_camera_initial, prior_summary = base.resolve_lidar_camera_initial(samples)
    before_rows = base.compute_sample_metadata_errors(
        T_lidar_camera_initial, samples, args.projection_model
    )

    residual_args = (
        T_lidar_camera_initial,
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
    result = least_squares(
        residual_vector_centered,
        x0=np.zeros(6, dtype=float),
        args=residual_args,
        loss=args.robust_loss,
        max_nfev=args.max_nfev,
    )
    xi = np.asarray(result.x, dtype=float)
    T_correction = base.se3_exp(xi)
    T_lidar_camera_optimized = T_correction @ T_lidar_camera_initial
    T_camera_lidar_initial = np.linalg.inv(T_lidar_camera_initial)
    T_camera_lidar_optimized = np.linalg.inv(T_lidar_camera_optimized)

    after_rows = base.compute_sample_metadata_errors(
        T_lidar_camera_optimized, samples, args.projection_model
    )
    before_summary = base.summarize_errors(before_rows, "before")
    after_summary = base.summarize_errors(after_rows, "after")
    warnings = metric_warnings(before_summary, after_summary)
    if not result.success:
        warnings.append(f"Optimizer did not report success: {result.message}")

    before_path = args.output_dir / "errors_before_optimization.csv"
    after_path = args.output_dir / "errors_after_optimization.csv"
    per_sample_path = args.output_dir / "optimized_extrinsics_per_sample.csv"
    base.write_csv(before_path, before_rows)
    base.write_csv(after_path, after_rows)
    base.write_csv(
        per_sample_path,
        centered_per_sample_rows(T_lidar_camera_optimized, samples),
    )

    preprocessing = {
        "timestamp_policy": "use_stored_T_map_lidar_directly",
        "timestamp_interpolation": False,
        "lidar_timestamp_imputation": False,
        "map_lidar_z_replacement": False,
        "gigapose_pose_source": "T_gigapose_cam_obj (raw centered-object pose)",
        "epnp_pose_source": args.epnp_map_pose_key,
        "raw_object_center_m": raw_center_m.tolist(),
        "raw_object_center_mm": (raw_center_m * 1000.0).tolist(),
        "T_object_raw_object_centered": T_object_raw_object_centered.tolist(),
        "target_equation": (
            "inv(T_map_lidar) @ T_map_object_raw @ T_object_raw_object_centered"
        ),
        "prediction_equation": ("T_lidar_camera @ T_camera_object_centered_gigapose"),
    }
    extrinsics = {
        "format_version": 1,
        "description": (
            "One fixed camera-LiDAR calibration optimized after explicitly "
            "converting EPnP raw-object poses into the GigaPose centered-object "
            "coordinate convention."
        ),
        "translation_unit": "mm",
        "calibrated_camera": calibrated_camera,
        "optimization_mode": "direct_map_lidar_raw_object_to_centered_gigapose",
        "preprocessing": preprocessing,
        "metadata_prior_policy": (
            "robust_common_initial_then_optimize_one_fixed_transform"
        ),
        "metadata_prior_variation": prior_summary,
        "T_lidar_camera_initial": T_lidar_camera_initial.tolist(),
        "T_camera_lidar_initial": T_camera_lidar_initial.tolist(),
        "T_lidar_camera_correction_left_multiply": T_correction.tolist(),
        "T_lidar_camera_optimized": T_lidar_camera_optimized.tolist(),
        "T_camera_lidar_optimized": T_camera_lidar_optimized.tolist(),
        "correction_rotation_rotvec_rad": xi[:3].tolist(),
        "correction_translation_mm": xi[3:6].tolist(),
        "optimized_extrinsics_per_sample_csv": str(per_sample_path),
        "errors_before_csv": str(before_path),
        "errors_after_csv": str(after_path),
    }
    extrinsics_path = args.output_dir / "optimized_extrinsics.json"
    extrinsics_path.write_text(json.dumps(extrinsics, indent=2))

    report = {
        "selected_samples": len(samples),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_cost": float(result.cost),
        "optimizer_nfev": int(result.nfev),
        "optimizer_optimality": float(result.optimality),
        "robust_loss": args.robust_loss,
        "calibrated_camera": calibrated_camera,
        "translation_sigma_mm": args.translation_sigma_mm,
        "translation_residual_components": args.translation_residual_components,
        "rotation_sigma_deg": args.rotation_sigma_deg,
        "image_center_weight": args.image_center_weight,
        "image_center_sigma_px": args.image_center_sigma_px,
        "projection_model": args.projection_model,
        "translation_prior_weight": args.translation_prior_weight,
        "rotation_prior_weight": args.rotation_prior_weight,
        "correction_translation_norm_mm": float(np.linalg.norm(xi[3:6])),
        "correction_rotation_norm_deg": float(np.degrees(np.linalg.norm(xi[:3]))),
        **preprocessing,
        **prior_summary,
        **before_summary,
        **after_summary,
        "warnings": warnings,
    }
    report_path = args.output_dir / "optimization_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"Wrote optimized extrinsics to {extrinsics_path}")


if __name__ == "__main__":
    main()
