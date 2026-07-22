"""Select real labels with the explicit centered camera-LiDAR calibration.

This selector is paired only with
``fine_tuning.optimize_camera_lidar_extrinsics_centered``.  It deliberately
does not estimate or apply the older empirical left/right frame transform.

For every EPnP label it reconstructs the centered camera-frame target as

    T_camera_object_centered
        = inv(T_lidar_camera_optimized)
          @ inv(T_map_lidar_target)
          @ T_map_object_raw
          @ T_object_raw_object_centered

where ``T_map_lidar_target`` replays the vertical convention recorded in the
optimizer JSON.  The selector also replays the GigaPose pose-source convention:
raw calibrations use raw predictions directly, while aligned calibrations
estimate or load the same explicit GigaPose-to-EPnP frame transform.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from fine_tuning import select_real_label_candidates as select_base
from fine_tuning import (
    select_real_label_candidates_with_camera_lidar_extrinsics as legacy,
)
from fine_tuning.optimize_camera_lidar_extrinsics import (
    prepare_target_map_lidar_transforms,
    resolve_target_map_lidar,
)


OPTIMIZATION_MODE = "direct_map_lidar_raw_object_to_centered_gigapose"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select raw GigaPose/EPnP pairs using a centered-object "
            "camera-LiDAR calibration."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/media/hdd2/ARCL_multicar_bags/camera_dataset"),
    )
    parser.add_argument("--date", action="append", default=None)
    parser.add_argument("--gigapose-predictions", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--frame-map", type=Path, default=None)
    parser.add_argument("--epnp-root", type=Path, default=None)
    parser.add_argument("--epnp-glob", default="**/*")
    parser.add_argument("--epnp-key-field", default=None)
    parser.add_argument("--epnp-key-prefix", default="")
    parser.add_argument("--epnp-strip-trailing-instance-id", action="store_true")
    parser.add_argument(
        "--match-key",
        choices=(
            "auto",
            "scene_im",
            "image_stem",
            "image_name",
            "frame_id",
            "sim_time_ms",
        ),
        default="auto",
    )
    parser.add_argument("--optimized-extrinsics", type=Path, required=True)
    parser.add_argument("--frame-transform-json", type=Path, default=None)
    parser.add_argument(
        "--frame-transform-side", choices=("left", "right"), default="right"
    )
    parser.add_argument(
        "--frame-transform-refine-iterations", type=int, default=5
    )
    parser.add_argument(
        "--frame-transform-inlier-translation-mm", type=float, default=None
    )
    parser.add_argument(
        "--frame-transform-inlier-rotation-deg", type=float, default=None
    )
    parser.add_argument(
        "--allow-unsafe-calibration",
        action="store_true",
        help=(
            "Override the physical-calibration safety check. This is intended "
            "only for diagnostics and should not be used to create labels."
        ),
    )
    parser.add_argument("--epnp-map-pose-key", default="T_map_object_raw")
    parser.add_argument(
        "--epnp-camera-pose-key", default="T_camera_object_centered"
    )
    parser.add_argument(
        "--epnp-map-pose-unit", choices=("auto", "m", "mm"), default="m"
    )
    parser.add_argument(
        "--epnp-camera-pose-unit", choices=("auto", "m", "mm"), default="m"
    )
    parser.add_argument(
        "--epnp-translation-unit",
        choices=("auto", "m", "mm"),
        default=None,
        help="Compatibility alias that sets both EPnP translation units.",
    )
    parser.add_argument("--min-score", type=float, default=0.05)
    parser.add_argument("--max-translation-error-mm", type=float, default=2000.0)
    parser.add_argument("--max-rotation-error-deg", type=float, default=30.0)
    parser.add_argument("--max-roll-error-deg", type=float, default=None)
    parser.add_argument("--max-pitch-error-deg", type=float, default=None)
    parser.add_argument("--max-yaw-error-deg", type=float, default=None)
    parser.add_argument("--max-candidates-per-key", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_centered_calibration(
    path: Path, allow_unsafe: bool
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], Path]:
    resolved = legacy.resolve_extrinsics_path(path)
    data = json.loads(resolved.read_text())
    mode = str(data.get("optimization_mode", ""))
    if mode != OPTIMIZATION_MODE:
        raise ValueError(
            f"{resolved} has optimization_mode={mode!r}; this selector requires "
            f"{OPTIMIZATION_MODE!r}."
        )

    T_lidar_camera = np.asarray(
        data.get("T_lidar_camera_optimized"), dtype=float
    ).reshape(4, 4)
    preprocessing = data.get("preprocessing", {})
    center_value = preprocessing.get("T_object_raw_object_centered")
    if center_value is None:
        raise ValueError(
            f"{resolved} does not contain preprocessing."
            "T_object_raw_object_centered."
        )
    T_object_raw_object_centered = np.asarray(
        center_value, dtype=float
    ).reshape(4, 4)
    legacy.rigid_transform_checks(T_lidar_camera, "T_lidar_camera_optimized")
    legacy.rigid_transform_checks(
        T_object_raw_object_centered, "T_object_raw_object_centered"
    )

    correction = np.asarray(
        data.get("correction_translation_mm", [np.nan, np.nan, np.nan]),
        dtype=float,
    ).reshape(3)
    correction_norm_mm = float(np.linalg.norm(correction))
    reusable_limit_mm = float(data.get("max_reusable_correction_mm", 2000.0))
    explicitly_valid = data.get("calibration_valid_for_reuse")
    valid = bool(
        (explicitly_valid is not False)
        and np.isfinite(correction_norm_mm)
        and correction_norm_mm <= reusable_limit_mm
    )
    if not valid and not allow_unsafe:
        raise ValueError(
            f"Refusing unsafe calibration {resolved}: correction translation is "
            f"{correction_norm_mm:.3f} mm (limit {reusable_limit_mm:.3f} mm), "
            f"calibration_valid_for_reuse={explicitly_valid!r}. Rerun the "
            "centered optimizer with --target-lidar-z-mode epnp_corrected. "
            "Use --allow-unsafe-calibration only for diagnostics."
        )
    return T_lidar_camera, T_object_raw_object_centered, data, resolved


def pose_errors(T_pred: np.ndarray, T_target: np.ndarray) -> tuple[float, float]:
    return (
        float(np.linalg.norm(T_pred[:3, 3] - T_target[:3, 3])),
        select_base.rotation_error_deg(T_pred[:3, :3], T_target[:3, :3]),
    )


def make_candidate_rows(
    preds_by_key: dict[str, list[dict[str, Any]]],
    labels_by_key: dict[str, list[dict[str, Any]]],
    T_lidar_camera: np.ndarray,
    T_object_raw_object_centered: np.ndarray,
    gigapose_frame_transform: np.ndarray | None,
    gigapose_frame_transform_side: str,
    max_candidates_per_key: int,
    extrinsics_path: Path,
) -> list[dict[str, Any]]:
    T_camera_lidar = np.linalg.inv(T_lidar_camera)
    rows: list[dict[str, Any]] = []
    for key in sorted(set(preds_by_key) & set(labels_by_key)):
        predictions = sorted(
            preds_by_key[key],
            key=lambda row: float(row.get("score", 0.0)),
            reverse=True,
        )[:max_candidates_per_key]
        labels = labels_by_key[key][:max_candidates_per_key]
        for label in labels:
            T_map_lidar_raw = np.asarray(label["T_map_lidar"], dtype=float)
            T_map_lidar_target = resolve_target_map_lidar(label)
            T_map_object_raw = np.asarray(label["T_map_object"], dtype=float)
            T_map_object_centered = (
                T_map_object_raw @ T_object_raw_object_centered
            )
            T_lidar_object_centered = (
                np.linalg.inv(T_map_lidar_target) @ T_map_object_centered
            )
            T_camera_object_centered = (
                T_camera_lidar @ T_lidar_object_centered
            )
            T_map_camera = T_map_lidar_target @ T_lidar_camera

            for prediction in predictions:
                T_gigapose_raw = np.asarray(
                    prediction["T_gigapose"], dtype=float
                ).reshape(4, 4)
                T_gigapose = (
                    select_base.apply_frame_transform(
                        T_gigapose_raw,
                        gigapose_frame_transform,
                        gigapose_frame_transform_side,
                    )
                    if gigapose_frame_transform is not None
                    else T_gigapose_raw
                )
                translation_error, rotation_error = pose_errors(
                    T_gigapose, T_camera_object_centered
                )
                roll_error, pitch_error, yaw_error = (
                    select_base.rotation_error_rpy_deg(
                        T_gigapose[:3, :3],
                        T_camera_object_centered[:3, :3],
                    )
                )
                map_translation_error, map_rotation_error = pose_errors(
                    T_map_camera @ T_gigapose, T_map_object_centered
                )
                raw_translation_error, raw_rotation_error = pose_errors(
                    T_gigapose_raw, label["T_camera_object_original"]
                )
                original_translation_error, original_rotation_error = pose_errors(
                    T_gigapose, label["T_camera_object_original"]
                )
                rows.append(
                    {
                        "match_key": key,
                        "scene_id": prediction["scene_id"],
                        "im_id": prediction["im_id"],
                        "obj_id": prediction["obj_id"],
                        "instance_id": prediction.get("instance_id", ""),
                        "prediction_row_index": prediction["row_index"],
                        "score": prediction["score"],
                        "epnp_label_path": label["epnp_label_path"],
                        "epnp_record_index": label["epnp_record_index"],
                        "translation_error_mm": translation_error,
                        "rotation_error_deg": rotation_error,
                        "roll_error_deg": roll_error,
                        "pitch_error_deg": pitch_error,
                        "yaw_error_deg": yaw_error,
                        "raw_translation_error_mm": raw_translation_error,
                        "raw_rotation_error_deg": raw_rotation_error,
                        "original_translation_error_mm": original_translation_error,
                        "original_rotation_error_deg": original_rotation_error,
                        "optimized_camera_translation_error_mm": translation_error,
                        "optimized_camera_rotation_error_deg": rotation_error,
                        "map_centered_translation_error_mm": map_translation_error,
                        "map_centered_rotation_error_deg": map_rotation_error,
                        "map_camera_translation_disagreement_mm": abs(
                            map_translation_error - translation_error
                        ),
                        "map_camera_rotation_disagreement_deg": abs(
                            map_rotation_error - rotation_error
                        ),
                        "T_gigapose_cam_obj": select_base.matrix_to_text(
                            T_gigapose_raw
                        ),
                        "T_gigapose_aligned_epnp_obj": select_base.matrix_to_text(
                            T_gigapose
                        ),
                        "T_epnp_obj": select_base.matrix_to_text(
                            T_camera_object_centered
                        ),
                        "T_epnp_camera_obj_original": select_base.matrix_to_text(
                            label["T_camera_object_original"]
                        ),
                        "T_epnp_camera_obj_optimized": select_base.matrix_to_text(
                            T_camera_object_centered
                        ),
                        "T_epnp_map_obj": select_base.matrix_to_text(
                            T_map_object_raw
                        ),
                        "T_map_object_raw": select_base.matrix_to_text(
                            T_map_object_raw
                        ),
                        "T_object_raw_object_centered": select_base.matrix_to_text(
                            T_object_raw_object_centered
                        ),
                        "T_map_object_centered": select_base.matrix_to_text(
                            T_map_object_centered
                        ),
                        "T_map_lidar": select_base.matrix_to_text(
                            T_map_lidar_raw
                        ),
                        "T_map_lidar_raw": select_base.matrix_to_text(
                            T_map_lidar_raw
                        ),
                        "T_map_lidar_target": select_base.matrix_to_text(
                            T_map_lidar_target
                        ),
                        "corrected_lidar_map_z_mm": label.get(
                            "corrected_lidar_map_z_mm", ""
                        ),
                        "target_lidar_z_replacement_mm": label.get(
                            "target_lidar_z_replacement_mm", ""
                        ),
                        "T_lidar_object_centered_target": select_base.matrix_to_text(
                            T_lidar_object_centered
                        ),
                        "T_lidar_camera_optimized": select_base.matrix_to_text(
                            T_lidar_camera
                        ),
                        "T_camera_lidar_optimized": select_base.matrix_to_text(
                            T_camera_lidar
                        ),
                        "T_map_cam_optimized": select_base.matrix_to_text(
                            T_map_camera
                        ),
                        "T_camera_map_optimized": select_base.matrix_to_text(
                            np.linalg.inv(T_map_camera)
                        ),
                        "T_gigapose_map_obj_optimized": select_base.matrix_to_text(
                            T_map_camera @ T_gigapose
                        ),
                        "sample_metadata_path": label["metadata_path"],
                        "optimized_extrinsics_path": str(extrinsics_path),
                        "frame_transform_applied": (
                            gigapose_frame_transform_side
                            if gigapose_frame_transform is not None
                            else "none"
                        ),
                    }
                )
    rows.sort(
        key=lambda row: (
            row["match_key"],
            float(row["translation_error_mm"]),
            float(row["rotation_error_deg"]),
            -float(row["score"]),
        )
    )
    return rows


def numeric_summary(
    rows: list[dict[str, Any]], source_key: str, prefix: str
) -> dict[str, float | int | None]:
    values = np.asarray([float(row[source_key]) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": None,
            f"{prefix}_median": None,
            f"{prefix}_p90": None,
        }
    return {
        f"{prefix}_count": int(values.size),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
    }


def main() -> None:
    args = parse_args()
    if args.max_candidates_per_key <= 0:
        raise ValueError("--max-candidates-per-key must be positive")
    for name in (
        "max_roll_error_deg",
        "max_pitch_error_deg",
        "max_yaw_error_deg",
    ):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative")
    if args.epnp_translation_unit is not None:
        args.epnp_map_pose_unit = args.epnp_translation_unit
        args.epnp_camera_pose_unit = args.epnp_translation_unit
    args.output_dir.mkdir(parents=True, exist_ok=True)

    (
        T_lidar_camera,
        T_object_raw_object_centered,
        calibration,
        calibration_path,
    ) = load_centered_calibration(
        args.optimized_extrinsics, args.allow_unsafe_calibration
    )

    frame_map = select_base.load_frame_map(args.frame_map, args.dataset_dir)
    predictions = select_base.load_gigapose_predictions(
        args.gigapose_predictions, frame_map, args.match_key
    )
    predictions = [
        row
        for row in predictions
        if float(row.get("score", 0.0)) >= args.min_score
    ]
    dates = tuple(args.date or select_base.DEFAULT_DATES)
    epnp_roots = select_base.discover_epnp_roots(
        args.source_root, dates, args.epnp_root
    )
    labels, label_skip_counts = legacy.load_epnp_extrinsic_labels(
        epnp_roots,
        args.epnp_glob,
        args.epnp_key_field,
        args.epnp_key_prefix,
        args.epnp_strip_trailing_instance_id,
        args.epnp_map_pose_key,
        args.epnp_camera_pose_key,
        args.epnp_map_pose_unit,
        args.epnp_camera_pose_unit,
    )
    if not labels:
        raise ValueError(
            "No usable EPnP labels were loaded. Check the EPnP root, pose "
            f"keys, units, and metadata paths. Skip counts: {label_skip_counts}"
        )

    calibrated_camera = str(calibration.get("calibrated_camera") or "")
    loaded_cameras = sorted({str(label["metadata_camera"]) for label in labels})
    if loaded_cameras != [calibrated_camera]:
        raise ValueError(
            f"Calibration is for {calibrated_camera!r}, but labels contain "
            f"cameras {loaded_cameras}."
        )

    target_lidar_z_mode = str(
        calibration.get(
            "target_lidar_z_mode",
            calibration.get("preprocessing", {}).get(
                "target_lidar_z_mode", "raw"
            ),
        )
    )
    allow_missing_z = bool(
        calibration.get(
            "allow_missing_corrected_lidar_z",
            calibration.get("preprocessing", {}).get(
                "allow_missing_corrected_lidar_z", False
            ),
        )
    )
    preprocessing_summary = prepare_target_map_lidar_transforms(
        labels,
        target_lidar_z_mode=target_lidar_z_mode,
        allow_missing_corrected_lidar_z=allow_missing_z,
        timestamp_alignment="raw",
        timestamp_max_bracket_gap_ms=1.0,
        timestamp_fallback="error",
        interpolate_missing_lidar_timestamps=False,
        timestamp_max_imputation_gap_ms=1.0,
        timestamp_max_offset_jump_ms=1.0,
    )

    predictions_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    labels_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        predictions_by_key[str(prediction["match_key"])].append(prediction)
    for label in labels:
        labels_by_key[str(label["match_key"])].append(label)

    pose_source = str(
        calibration.get("preprocessing", {}).get(
            "gigapose_pose_source", "T_gigapose_cam_obj"
        )
    )
    uses_aligned_pose = "aligned" in pose_source.lower()
    estimated_pairs = 0
    refinement_history: list[dict[str, Any]] = []
    gigapose_frame_transform: np.ndarray | None = None
    if uses_aligned_pose:
        if args.frame_transform_json is not None:
            gigapose_frame_transform = select_base.load_frame_transform(
                args.frame_transform_json
            )
            frame_transform_source = str(args.frame_transform_json)
        else:
            one_to_one = select_base.initial_one_to_one_pairs(
                predictions_by_key, labels_by_key
            )
            gigapose_frame_transform = select_base.estimate_frame_transform(
                one_to_one, args.frame_transform_side
            )
            estimated_pairs = len(one_to_one)
            inlier_translation = (
                args.frame_transform_inlier_translation_mm
                if args.frame_transform_inlier_translation_mm is not None
                else args.max_translation_error_mm
            )
            inlier_rotation = (
                args.frame_transform_inlier_rotation_deg
                if args.frame_transform_inlier_rotation_deg is not None
                else args.max_rotation_error_deg
            )
            (
                gigapose_frame_transform,
                refinement_history,
            ) = select_base.refine_frame_transform(
                one_to_one,
                args.frame_transform_side,
                gigapose_frame_transform,
                args.frame_transform_refine_iterations,
                inlier_translation,
                inlier_rotation,
            )
            frame_transform_source = "estimated_from_current_one_to_one_pairs"
    else:
        if args.frame_transform_json is not None:
            raise ValueError(
                "This calibration used raw GigaPose poses; refusing an external "
                "frame transform because it would not replay the calibration."
            )
        frame_transform_source = "none_raw_pose_calibration"

    all_candidates = make_candidate_rows(
        predictions_by_key,
        labels_by_key,
        T_lidar_camera,
        T_object_raw_object_centered,
        gigapose_frame_transform,
        args.frame_transform_side,
        args.max_candidates_per_key,
        calibration_path,
    )
    best_candidates = select_base.best_rows_per_label(all_candidates)
    selected = [
        row
        for row in best_candidates
        if float(row["translation_error_mm"])
        <= args.max_translation_error_mm
        and float(row["rotation_error_deg"]) <= args.max_rotation_error_deg
        and (
            args.max_roll_error_deg is None
            or float(row["roll_error_deg"]) <= args.max_roll_error_deg
        )
        and (
            args.max_pitch_error_deg is None
            or float(row["pitch_error_deg"]) <= args.max_pitch_error_deg
        )
        and (
            args.max_yaw_error_deg is None
            or float(row["yaw_error_deg"]) <= args.max_yaw_error_deg
        )
    ]

    select_base.write_csv(args.output_dir / "all_candidate_pairs.csv", all_candidates)
    select_base.write_csv(
        args.output_dir / "best_candidate_per_epnp_label.csv", best_candidates
    )
    select_base.write_csv(args.output_dir / "selected_samples.csv", selected)
    frame_transform_path = (
        args.output_dir / "frame_transform_gigapose_to_epnp.json"
    )
    select_base.save_frame_transform(
        frame_transform_path,
        (
            gigapose_frame_transform
            if gigapose_frame_transform is not None
            else np.eye(4)
        ),
        estimated_pairs,
        args.frame_transform_side,
    )

    report: dict[str, Any] = {
        "description": (
            "Centered-object label selection using the exact vertical and CAD "
            "origin conventions saved by the centered camera-LiDAR optimizer."
        ),
        "gigapose_predictions": str(args.gigapose_predictions),
        "optimized_extrinsics": str(calibration_path),
        "calibrated_camera": calibrated_camera,
        "optimization_mode": calibration.get("optimization_mode"),
        "calibration_valid_for_reuse": calibration.get(
            "calibration_valid_for_reuse"
        ),
        "allow_unsafe_calibration": args.allow_unsafe_calibration,
        "gigapose_pose_source": pose_source,
        "frame_transform_applied": (
            args.frame_transform_side if uses_aligned_pose else "none"
        ),
        "frame_transform_source": frame_transform_source,
        "frame_transform_estimated_from_one_to_one_pairs": estimated_pairs,
        "frame_transform_refine_iterations": (
            args.frame_transform_refine_iterations if uses_aligned_pose else 0
        ),
        "frame_transform_refinement_history": refinement_history,
        "target_lidar_z_mode": target_lidar_z_mode,
        "target_map_lidar_preprocessing": preprocessing_summary,
        "raw_object_center_m": calibration.get("preprocessing", {}).get(
            "raw_object_center_m"
        ),
        "epnp_roots": [str(path) for path in epnp_roots],
        "loaded_predictions_after_score_filter": len(predictions),
        "loaded_epnp_labels": len(labels),
        "epnp_label_skip_counts": label_skip_counts,
        "matched_keys": len(set(predictions_by_key) & set(labels_by_key)),
        "candidate_pairs": len(all_candidates),
        "best_candidate_pairs": len(best_candidates),
        "selected_samples": len(selected),
        "min_score": args.min_score,
        "max_translation_error_mm": args.max_translation_error_mm,
        "max_rotation_error_deg": args.max_rotation_error_deg,
        "max_roll_error_deg": args.max_roll_error_deg,
        "max_pitch_error_deg": args.max_pitch_error_deg,
        "max_yaw_error_deg": args.max_yaw_error_deg,
    }
    report.update(
        numeric_summary(
            best_candidates, "translation_error_mm", "translation_error_mm"
        )
    )
    report.update(
        numeric_summary(
            best_candidates, "rotation_error_deg", "rotation_error_deg"
        )
    )
    (args.output_dir / "selection_report.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2))
    print(f"Wrote {len(selected)} selected rows to {args.output_dir / 'selected_samples.csv'}")


if __name__ == "__main__":
    main()
