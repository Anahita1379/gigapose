"""Robustly estimate one fixed LiDAR<-camera transform from EPnP anchors."""
from __future__ import annotations

import argparse
import numpy as np
from scipy.optimize import least_squares

from .geometry import as_pose, pose_error, so3_exp
from .io import write_json
from .trajectory import load


def _validate_trajectory_contract(rows):
    hybrid_rows = [
        row
        for row in rows
        if "mesh_z_hybrid" in str(row.get("epnp_label_path", "")).lower()
    ]
    stale_rows = [
        row
        for row in hybrid_rows
        if row.get("target_lidar_z_mode") != "epnp_corrected"
        or row.get("corrected_lidar_map_z_m") is None
    ]
    if stale_rows:
        raise ValueError(
            f"{len(stale_rows)} of {len(hybrid_rows)} mesh-Z hybrid rows were "
            "created by the old incompatible observation adapter: corrected "
            "LiDAR-map Z provenance is missing. Rebuild observations, then "
            "regenerate initial_trajectories_iteration_0.jsonl and "
            "refined_iteration_0.jsonl before optimizing the extrinsic."
        )


def _representative_prior(rows):
    priors = np.stack([as_pose(row["T_lidar_camera_initial"]) for row in rows])
    output = np.eye(4)
    output[:3, 3] = np.median(priors[:, :3, 3], axis=0)
    u, _, vt = np.linalg.svd(np.mean(priors[:, :3, :3], axis=0))
    output[:3, :3] = u @ vt
    if np.linalg.det(output[:3, :3]) < 0:
        u[:, -1] *= -1
        output[:3, :3] = u @ vt
    return output


def _updated(prior, xi):
    delta = np.eye(4)
    delta[:3, :3] = so3_exp(xi[:3])
    delta[:3, 3] = xi[3:]
    return delta @ prior


def _target(row):
    value = row.get("T_map_object_centered_epnp")
    if value is None:
        value = row.get("T_map_object_centered_refined")
    return as_pose(value)


def _sample_errors(rows, extrinsic):
    values = []
    for row in rows:
        predicted = (
            as_pose(row["T_map_lidar"])
            @ extrinsic
            @ as_pose(row["T_camera_object_centered_gigapose"])
        )
        error = pose_error(_target(row), predicted)
        values.append(
            [np.linalg.norm(error[:3]), np.degrees(np.linalg.norm(error[3:]))]
        )
    return np.asarray(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-lidar-points", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--translation-sigma-m", type=float, default=1.0)
    parser.add_argument("--rotation-sigma-deg", type=float, default=10.0)
    parser.add_argument("--translation-prior-weight", type=float, default=0.05)
    parser.add_argument("--rotation-prior-weight", type=float, default=0.05)
    parser.add_argument(
        "--max-correction-translation-m", type=float, default=2.0
    )
    parser.add_argument(
        "--max-correction-rotation-deg", type=float, default=10.0
    )
    parser.add_argument(
        "--allow-large-correction",
        action="store_true",
        help=(
            "Expose an optimized transform even when it exceeds the safety "
            "limits. Intended only for deliberate diagnostics."
        ),
    )
    args = parser.parse_args()
    rows = load(args.trajectories)
    _validate_trajectory_contract(rows)
    usable = []
    for row in rows:
        point_count = int(float(row.get("lidar_point_count", 0) or 0))
        independent = (
            str(row.get("independent_support", "")).lower()
            in ("1", "true", "yes", "lidar")
            or point_count >= args.min_lidar_points
        )
        if (
            independent
            and row.get("T_map_lidar") is not None
            and row.get("T_camera_object_centered_gigapose") is not None
            and (
                row.get("T_map_object_centered_epnp") is not None
                or row.get("T_map_object_centered_refined") is not None
            )
        ):
            usable.append(row)
    if not usable:
        raise ValueError(
            "No independent calibration samples. EPnP-adapted observations "
            "set independent_support automatically."
        )
    prior = _representative_prior(usable)
    rotation_sigma = np.radians(args.rotation_sigma_deg)

    def residual(xi):
        extrinsic = _updated(prior, xi)
        values = []
        for row in usable:
            predicted = (
                as_pose(row["T_map_lidar"])
                @ extrinsic
                @ as_pose(row["T_camera_object_centered_gigapose"])
            )
            error = pose_error(_target(row), predicted)
            values.extend((error[:3] / args.translation_sigma_m).tolist())
            values.extend((error[3:] / rotation_sigma).tolist())
        values.extend(
            (
                np.sqrt(args.rotation_prior_weight)
                * xi[:3]
                / rotation_sigma
            ).tolist()
        )
        values.extend(
            (
                np.sqrt(args.translation_prior_weight)
                * xi[3:]
                / args.translation_sigma_m
            ).tolist()
        )
        return np.asarray(values)

    result = least_squares(
        residual,
        np.zeros(6),
        loss="soft_l1",
        max_nfev=args.max_iterations,
    )
    optimized = _updated(prior, result.x)
    before = _sample_errors(usable, prior)
    after = _sample_errors(usable, optimized)
    correction_translation_norm_m = float(np.linalg.norm(result.x[3:]))
    correction_rotation_deg = float(
        np.degrees(np.linalg.norm(result.x[:3]))
    )
    correction_within_limits = bool(
        correction_translation_norm_m <= args.max_correction_translation_m
        and correction_rotation_deg <= args.max_correction_rotation_deg
    )
    calibration_valid = bool(result.success and correction_within_limits)
    report = {
        "format": "teacher_fixed_extrinsic_v2",
        "optimization_mode": "fixed_lidar_camera_from_epnp_hybrid_anchors",
        "translation_unit": "m",
        "sample_count": len(usable),
        "independent_support_required": True,
        "target_preference": "T_map_object_centered_epnp_then_refined",
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_nfev": int(result.nfev),
        "calibration_valid_for_reuse": calibration_valid,
        "correction_within_safety_limits": correction_within_limits,
        "max_correction_translation_m": args.max_correction_translation_m,
        "max_correction_rotation_deg": args.max_correction_rotation_deg,
        "before_translation_error_m_median": float(np.median(before[:, 0])),
        "after_translation_error_m_median": float(np.median(after[:, 0])),
        "before_rotation_error_deg_median": float(np.median(before[:, 1])),
        "after_rotation_error_deg_median": float(np.median(after[:, 1])),
        "T_lidar_camera_initial": prior.tolist(),
        "T_lidar_camera_candidate": optimized.tolist(),
        "T_camera_lidar_candidate": np.linalg.inv(optimized).tolist(),
        "correction_rotvec_rad": result.x[:3].tolist(),
        "correction_translation_m": result.x[3:].tolist(),
        "correction_translation_norm_m": correction_translation_norm_m,
        "correction_rotation_deg": correction_rotation_deg,
        "used_rows": [row.get("prediction_row") for row in usable],
    }
    if calibration_valid or args.allow_large_correction:
        report["T_lidar_camera_optimized"] = optimized.tolist()
        report["T_camera_lidar_optimized"] = np.linalg.inv(optimized).tolist()
    write_json(args.output, report)
    if not correction_within_limits and not args.allow_large_correction:
        raise SystemExit(
            "Rejected implausible fixed-extrinsic correction: "
            f"translation={correction_translation_norm_m:.3f} m "
            f"(limit {args.max_correction_translation_m:.3f} m), "
            f"rotation={correction_rotation_deg:.3f} deg "
            f"(limit {args.max_correction_rotation_deg:.3f} deg). "
            "The diagnostic report was written, but it intentionally has no "
            "T_lidar_camera_optimized field. Check frame and unit conventions."
        )
    print(
        f"Optimized fixed extrinsic from {len(usable)} independent samples; "
        f"median translation error {np.median(before[:, 0]):.3f} -> "
        f"{np.median(after[:, 0]):.3f} m"
    )


if __name__ == "__main__":
    main()
