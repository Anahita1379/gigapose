"""Optimize one fixed camera-to-LiDAR transform for one camera.

For selected sample i, all translations are in millimetres:

    G_i = T_camera_object_centered_gigapose
    M_i = T_map_object_raw_epnp
    L_i = T_map_lidar_metadata
    C   = T_object_raw_object_centered
    X   = T_lidar_camera

The centered EPnP target in LiDAR coordinates is

    Q_i = inv(L_i) @ M_i @ C

and the prediction in LiDAR coordinates is

    P_i(X) = X @ G_i.

One shared X is optimized for every selected row belonging to the requested
camera.  No GigaPose-to-EPnP frame transform is fitted.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from label_selection import common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize one fixed T_lidar_camera from absolute selected pairs."
    )
    parser.add_argument("--selected-samples", type=Path, required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument(
        "--translation-residual-components",
        choices=("x", "y", "z", "xy", "xz", "yz", "xyz"),
        default="xyz",
    )
    parser.add_argument("--translation-sigma-mm", type=float, default=1000.0)
    parser.add_argument("--rotation-sigma-deg", type=float, default=10.0)
    parser.add_argument("--translation-prior-weight", type=float, default=1000.0)
    parser.add_argument("--rotation-prior-weight", type=float, default=20.0)
    parser.add_argument(
        "--robust-loss",
        choices=("linear", "soft_l1", "huber", "cauchy", "arctan"),
        default="soft_l1",
    )
    parser.add_argument("--f-scale", type=float, default=1.0)
    parser.add_argument("--max-nfev", type=int, default=500)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(3)
    theta = float(np.linalg.norm(vector))
    W = skew(vector)
    if theta < 1e-12:
        return np.eye(3) + W
    return (
        np.eye(3)
        + math.sin(theta) / theta * W
        + (1.0 - math.cos(theta)) / (theta * theta) * (W @ W)
    )


def se3_exp(update: np.ndarray) -> np.ndarray:
    """Small left update [rotation radians, translation millimetres]."""

    update = np.asarray(update, dtype=float).reshape(6)
    T = np.eye(4, dtype=float)
    T[:3, :3] = so3_exp(update[:3])
    T[:3, 3] = update[3:]
    return T


def robust_initial(priors: list[np.ndarray]) -> tuple[np.ndarray, dict[str, float]]:
    stacked = np.asarray(priors, dtype=float)
    initial = np.eye(4, dtype=float)
    initial[:3, 3] = np.median(stacked[:, :3, 3], axis=0)
    U, _, Vt = np.linalg.svd(np.mean(stacked[:, :3, :3], axis=0))
    initial[:3, :3] = U @ Vt
    if np.linalg.det(initial[:3, :3]) < 0:
        U[:, -1] *= -1
        initial[:3, :3] = U @ Vt
    translation_spread = np.asarray(
        [np.linalg.norm(T[:3, 3] - initial[:3, 3]) for T in priors]
    )
    rotation_spread = np.asarray(
        [
            common.rotation_error_deg(T[:3, :3], initial[:3, :3])
            for T in priors
        ]
    )
    return initial, {
        "translation_spread_mm_median": float(np.median(translation_spread)),
        "translation_spread_mm_max": float(np.max(translation_spread)),
        "rotation_spread_deg_median": float(np.median(rotation_spread)),
        "rotation_spread_deg_max": float(np.max(rotation_spread)),
    }


def load_samples(path: Path, camera_id: str) -> list[dict[str, Any]]:
    rows = common.read_csv(path)
    samples: list[dict[str, Any]] = []
    required = (
        "T_gigapose_known_center_aligned",
        "T_epnp_map_object_raw",
        "T_object_raw_object_centered",
        "T_map_lidar_metadata",
        "T_lidar_camera_prior_metadata",
    )
    for row_number, row in enumerate(rows, start=2):
        if row.get("comparison_mode") != common.COMPARISON_MODE:
            raise ValueError(
                f"{path}:{row_number} is not an uncalibrated absolute-selection row"
            )
        if row.get("camera_id") != camera_id:
            raise ValueError(
                f"{path}:{row_number} belongs to camera {row.get('camera_id')!r}; "
                f"expected {camera_id!r}"
            )
        missing = [field for field in required if not row.get(field)]
        if missing:
            raise ValueError(f"{path}:{row_number} is missing {missing}")
        sample = dict(row)
        sample["T_gigapose_centered"] = common.text_to_matrix(
            row["T_gigapose_known_center_aligned"]
        )
        sample["T_map_object_raw"] = common.text_to_matrix(
            row["T_epnp_map_object_raw"]
        )
        sample["C"] = common.text_to_matrix(
            row["T_object_raw_object_centered"]
        )
        sample["T_map_lidar"] = common.text_to_matrix(
            row["T_map_lidar_metadata"]
        )
        sample["T_lidar_camera_prior"] = common.text_to_matrix(
            row["T_lidar_camera_prior_metadata"]
        )
        for name in (
            "T_gigapose_centered",
            "T_map_object_raw",
            "C",
            "T_map_lidar",
            "T_lidar_camera_prior",
        ):
            common.rigid_transform_checks(sample[name], name)
        sample["T_lidar_object_centered_target"] = (
            np.linalg.inv(sample["T_map_lidar"])
            @ sample["T_map_object_raw"]
            @ sample["C"]
        )
        samples.append(sample)
    return samples


def error_row(sample: dict[str, Any], T_lidar_camera: np.ndarray) -> dict[str, Any]:
    predicted = T_lidar_camera @ sample["T_gigapose_centered"]
    target = sample["T_lidar_object_centered_target"]
    t_error, r_error, roll, pitch, yaw = common.pose_errors(predicted, target)
    corrected_camera_target = np.linalg.inv(T_lidar_camera) @ target
    camera_t, camera_r, _, _, _ = common.pose_errors(
        sample["T_gigapose_centered"], corrected_camera_target
    )
    return {
        "match_key": sample.get("match_key", ""),
        "scene_id": sample.get("scene_id", ""),
        "im_id": sample.get("im_id", ""),
        "epnp_label_path": sample.get("epnp_label_path", ""),
        "epnp_record_index": sample.get("epnp_record_index", ""),
        "translation_error_mm": t_error,
        "rotation_error_deg": r_error,
        "roll_error_deg": roll,
        "pitch_error_deg": pitch,
        "yaw_error_deg": yaw,
        "camera_frame_translation_error_mm": camera_t,
        "camera_frame_rotation_error_deg": camera_r,
        "T_epnp_camera_object_centered_from_calibration": common.matrix_to_text(
            corrected_camera_target
        ),
    }


def residual_function(
    update: np.ndarray,
    initial: np.ndarray,
    samples: list[dict[str, Any]],
    component_indices: list[int],
    translation_sigma_mm: float,
    rotation_sigma_rad: float,
    translation_prior_weight: float,
    rotation_prior_weight: float,
) -> np.ndarray:
    candidate = se3_exp(update) @ initial
    residuals: list[float] = []
    for sample in samples:
        predicted = candidate @ sample["T_gigapose_centered"]
        target = sample["T_lidar_object_centered_target"]
        translation = (
            predicted[:3, 3] - target[:3, 3]
        ) / translation_sigma_mm
        rotation = Rotation.from_matrix(
            target[:3, :3].T @ predicted[:3, :3]
        ).as_rotvec() / rotation_sigma_rad
        residuals.extend(translation[component_indices].tolist())
        residuals.extend(rotation.tolist())

    # These match the existing optimizer's coefficient semantics: weights are
    # coefficients in squared loss and therefore enter residuals as sqrt(weight).
    residuals.extend(
        (
            update[:3] * math.sqrt(max(rotation_prior_weight, 0.0))
        ).tolist()
    )
    residuals.extend(
        (
            update[3:]
            / translation_sigma_mm
            * math.sqrt(max(translation_prior_weight, 0.0))
        ).tolist()
    )
    return np.asarray(residuals, dtype=float)


def matrix_mm_to_m(T_mm: np.ndarray) -> list[list[float]]:
    T_m = np.asarray(T_mm, dtype=float).copy()
    T_m[:3, 3] *= 0.001
    return T_m.tolist()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{args.output_dir} is not empty; pass --overwrite to replace outputs"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.selected_samples, args.camera_id)
    if len(samples) < args.min_samples:
        raise ValueError(
            f"Only {len(samples)} valid {args.camera_id} samples; "
            f"--min-samples requires {args.min_samples}"
        )

    reference_C = samples[0]["C"]
    for index, sample in enumerate(samples[1:], start=1):
        if not np.allclose(sample["C"], reference_C, atol=1e-6):
            raise ValueError(f"Sample {index} uses a different raw-object center")

    initial, prior_spread = robust_initial(
        [sample["T_lidar_camera_prior"] for sample in samples]
    )
    components = {"x": 0, "y": 1, "z": 2}
    component_indices = [
        components[component]
        for component in args.translation_residual_components
    ]
    rotation_sigma_rad = math.radians(args.rotation_sigma_deg)
    if args.translation_sigma_mm <= 0 or rotation_sigma_rad <= 0:
        raise ValueError("Translation and rotation sigmas must be positive")

    before_rows = [error_row(sample, initial) for sample in samples]
    result = least_squares(
        residual_function,
        x0=np.zeros(6, dtype=float),
        args=(
            initial,
            samples,
            component_indices,
            args.translation_sigma_mm,
            rotation_sigma_rad,
            args.translation_prior_weight,
            args.rotation_prior_weight,
        ),
        loss=args.robust_loss,
        f_scale=args.f_scale,
        max_nfev=args.max_nfev,
    )
    optimized = se3_exp(result.x) @ initial
    common.rigid_transform_checks(optimized, "T_lidar_camera_optimized")
    after_rows = [error_row(sample, optimized) for sample in samples]

    for before, after in zip(before_rows, after_rows):
        before["stage"] = "before"
        after["stage"] = "after"
    common.write_csv(args.output_dir / "errors_before.csv", before_rows)
    common.write_csv(args.output_dir / "errors_after.csv", after_rows)
    paired_rows = []
    for before, after in zip(before_rows, after_rows):
        paired = {
            key: value
            for key, value in before.items()
            if key
            in (
                "match_key",
                "scene_id",
                "im_id",
                "epnp_label_path",
                "epnp_record_index",
            )
        }
        for key, value in before.items():
            if key not in paired and key != "stage":
                paired[f"before_{key}"] = value
        for key, value in after.items():
            if key not in paired and key != "stage":
                paired[f"after_{key}"] = value
        paired_rows.append(paired)
    common.write_csv(args.output_dir / "per_sample_calibration.csv", paired_rows)

    before_translation = common.summarize_numeric(before_rows, "translation_error_mm")
    after_translation = common.summarize_numeric(after_rows, "translation_error_mm")
    before_rotation = common.summarize_numeric(before_rows, "rotation_error_deg")
    after_rotation = common.summarize_numeric(after_rows, "rotation_error_deg")
    correction = optimized @ np.linalg.inv(initial)
    calibration_improves_mean = (
        float(after_translation.get("mean", float("inf")))
        <= float(before_translation.get("mean", float("inf")))
        and float(after_rotation.get("mean", float("inf")))
        <= float(before_rotation.get("mean", float("inf")))
    )
    calibration_data = {
        "format": "label_selection_camera_lidar_calibration_v1",
        "optimization_mode": common.CALIBRATION_MODE,
        "camera_id": args.camera_id,
        "translation_unit": "mm",
        "rotation_unit": "radians_in_optimizer",
        "selected_samples": str(args.selected_samples),
        "samples_used": len(samples),
        "comparison_mode_of_input": common.COMPARISON_MODE,
        "empirical_prediction_label_alignment_used": False,
        "T_lidar_camera_initial": initial.tolist(),
        "T_lidar_camera_initial_m": matrix_mm_to_m(initial),
        "T_lidar_camera_optimized": optimized.tolist(),
        "T_lidar_camera_optimized_m": matrix_mm_to_m(optimized),
        "T_camera_lidar_optimized": np.linalg.inv(optimized).tolist(),
        "T_camera_lidar_optimized_m": matrix_mm_to_m(np.linalg.inv(optimized)),
        "T_lidar_camera_correction": correction.tolist(),
        "T_object_raw_object_centered": reference_C.tolist(),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_cost": float(result.cost),
        "optimizer_nfev": int(result.nfev),
        "optimizer_optimality": float(result.optimality),
        "robust_loss": args.robust_loss,
        "f_scale": args.f_scale,
        "translation_residual_components": args.translation_residual_components,
        "translation_sigma_mm": args.translation_sigma_mm,
        "rotation_sigma_deg": args.rotation_sigma_deg,
        "translation_prior_weight": args.translation_prior_weight,
        "rotation_prior_weight": args.rotation_prior_weight,
        "loss_weight_semantics": (
            "prior weights are squared-loss coefficients; residuals are "
            "multiplied by sqrt(weight)"
        ),
        "prior_spread": prior_spread,
        "correction_translation_norm_mm": float(
            np.linalg.norm(correction[:3, 3])
        ),
        "correction_rotation_norm_deg": common.rotation_error_deg(
            correction[:3, :3], np.eye(3)
        ),
        "before_translation_error_mm": before_translation,
        "after_translation_error_mm": after_translation,
        "before_rotation_error_deg": before_rotation,
        "after_rotation_error_deg": after_rotation,
        "recommended_for_reselection": bool(
            result.success and calibration_improves_mean
        ),
        "equations": {
            "epnp_target_lidar_centered": (
                "Q_i = inv(T_map_lidar_i) @ T_map_object_raw_i "
                "@ T_object_raw_object_centered"
            ),
            "gigapose_prediction_lidar_centered": (
                "P_i(X) = T_lidar_camera @ T_camera_object_centered_gigapose_i"
            ),
            "corrected_epnp_camera_target": (
                "T_camera_object_centered_corrected_i = "
                "inv(T_lidar_camera_optimized) @ Q_i"
            ),
        },
    }
    common.save_json(
        args.output_dir / "optimized_extrinsics.json", calibration_data
    )
    common.save_json(args.output_dir / "optimization_report.json", calibration_data)
    print(json.dumps(calibration_data, indent=2))
    if not calibration_data["recommended_for_reselection"]:
        print(
            "WARNING: optimization completed, but mean translation and rotation "
            "did not both improve. Inspect errors_before.csv/errors_after.csv."
        )
    print(f"Wrote calibration to {args.output_dir / 'optimized_extrinsics.json'}")


if __name__ == "__main__":
    main()
