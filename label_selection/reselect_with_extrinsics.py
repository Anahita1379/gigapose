"""Reselect pairs using one saved, fixed camera-LiDAR calibration.

For each EPnP label, the calibrated centered camera target is reconstructed as

    inv(T_lidar_camera_optimized)
    @ inv(T_map_lidar)
    @ T_map_object_raw
    @ T_object_raw_object_centered.

That reconstructed target is compared directly with the known-center-converted
GigaPose prediction.  No prediction-label alignment is estimated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from label_selection import common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reselect absolute pairs after camera-LiDAR calibration."
    )
    parser.add_argument("--gigapose-predictions", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--frame-map", type=Path, default=None)
    parser.add_argument("--epnp-root", type=Path, required=True)
    parser.add_argument("--epnp-glob", default="*.json")
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
        default="image_stem",
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
    parser.add_argument("--optimized-extrinsics", type=Path, required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument(
        "--prediction-translation-unit",
        choices=("mm", "m", "auto"),
        default="mm",
        help="Unit of the GigaPose CSV t column. BOP GigaPose output is normally mm.",
    )
    parser.add_argument("--min-score", type=float, default=0.05)
    parser.add_argument("--max-candidates-per-key", type=int, default=20)
    parser.add_argument("--pairing-translation-scale-mm", type=float, default=1000.0)
    parser.add_argument("--pairing-rotation-scale-deg", type=float, default=10.0)
    parser.add_argument("--max-translation-error-mm", type=float, default=3000.0)
    parser.add_argument("--max-rotation-error-deg", type=float, default=30.0)
    parser.add_argument("--max-roll-error-deg", type=float, default=None)
    parser.add_argument("--max-pitch-error-deg", type=float, default=None)
    parser.add_argument("--max-yaw-error-deg", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_calibration(
    path: Path, camera_id: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], Path]:
    resolved = path / "optimized_extrinsics.json" if path.is_dir() else path
    if not resolved.is_file():
        raise FileNotFoundError(f"Could not find calibration JSON: {resolved}")
    data = json.loads(resolved.read_text())
    if data.get("optimization_mode") != common.CALIBRATION_MODE:
        raise ValueError(
            f"{resolved} has optimization_mode={data.get('optimization_mode')!r}; "
            f"expected {common.CALIBRATION_MODE!r}"
        )
    if data.get("camera_id") != camera_id:
        raise ValueError(
            f"{resolved} is for camera {data.get('camera_id')!r}, not {camera_id!r}"
        )
    if data.get("translation_unit") != "mm":
        raise ValueError(
            f"{resolved} must store T_lidar_camera_optimized in millimetres"
        )
    T_lidar_camera = np.asarray(
        data["T_lidar_camera_optimized"], dtype=float
    ).reshape(4, 4)
    C = np.asarray(
        data["T_object_raw_object_centered"], dtype=float
    ).reshape(4, 4)
    common.rigid_transform_checks(T_lidar_camera, "T_lidar_camera_optimized")
    common.rigid_transform_checks(C, "T_object_raw_object_centered")
    return T_lidar_camera, C, data, resolved


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{args.output_dir} is not empty; pass --overwrite to replace outputs"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    T_lidar_camera, C, calibration, calibration_path = load_calibration(
        args.optimized_extrinsics, args.camera_id
    )

    predictions, _ = common.load_predictions(
        args.gigapose_predictions,
        args.dataset_dir,
        args.frame_map,
        args.match_key,
        args.min_score,
        args.prediction_translation_unit,
    )
    labels, skipped_labels = common.load_labels(
        args.epnp_root,
        args.epnp_glob,
        args.epnp_key_field,
        args.epnp_key_prefix,
        args.epnp_strip_trailing_instance_id,
        args.epnp_map_pose_key,
        args.epnp_camera_pose_key,
        args.epnp_map_pose_unit,
        args.epnp_camera_pose_unit,
    )
    wrong_cameras = sorted(
        {
            str(label.get("metadata_camera") or "unknown")
            for label in labels
            if str(label.get("metadata_camera") or "unknown") != args.camera_id
        }
    )
    if wrong_cameras:
        raise ValueError(
            f"EPnP root contains camera groups {wrong_cameras}; expected only "
            f"{args.camera_id!r}"
        )

    target_by_identity: dict[tuple[str, int], np.ndarray] = {}
    extra_by_identity: dict[tuple[str, int], dict[str, Any]] = {}
    for label in labels:
        identity = (
            str(label["epnp_label_path"]),
            int(label.get("epnp_record_index", 0)),
        )
        target_lidar = (
            np.linalg.inv(label["T_map_lidar"])
            @ label["T_map_object"]
            @ C
        )
        corrected_camera = np.linalg.inv(T_lidar_camera) @ target_lidar
        common.rigid_transform_checks(
            corrected_camera, "T_epnp_camera_object_centered_corrected"
        )
        target_by_identity[identity] = corrected_camera
        extra_by_identity[identity] = {
            "optimized_extrinsics_path": str(calibration_path),
            "T_lidar_camera_optimized": common.matrix_to_text(T_lidar_camera),
            "T_camera_lidar_optimized": common.matrix_to_text(
                np.linalg.inv(T_lidar_camera)
            ),
            "T_epnp_camera_object_centered_corrected": common.matrix_to_text(
                corrected_camera
            ),
        }

    all_pairs, assigned, pair_counts = common.build_and_assign_pairs(
        predictions,
        labels,
        C,
        args.pairing_translation_scale_mm,
        args.pairing_rotation_scale_deg,
        args.max_candidates_per_key,
        comparison_mode=common.POST_CALIBRATION_MODE,
        epnp_target_by_identity=target_by_identity,
        extra_by_identity=extra_by_identity,
    )
    for row in all_pairs:
        T_gigapose = common.text_to_matrix(
            row["T_gigapose_known_center_aligned"]
        )
        T_original = common.text_to_matrix(
            row["T_epnp_camera_object_centered_original"]
        )
        t_error, r_error, roll, pitch, yaw = common.pose_errors(
            T_gigapose, T_original
        )
        row.update(
            {
                "precalibration_absolute_translation_error_mm": t_error,
                "precalibration_absolute_rotation_error_deg": r_error,
                "precalibration_absolute_roll_error_deg": roll,
                "precalibration_absolute_pitch_error_deg": pitch,
                "precalibration_absolute_yaw_error_deg": yaw,
            }
        )
    # assigned rows are copies created before adding the diagnostics above.
    by_pair = {
        (
            row["match_key"],
            str(row["prediction_row_index"]),
            row["epnp_label_path"],
            str(row["epnp_record_index"]),
        ): row
        for row in all_pairs
    }
    for index, assigned_row in enumerate(assigned):
        key = (
            assigned_row["match_key"],
            str(assigned_row["prediction_row_index"]),
            assigned_row["epnp_label_path"],
            str(assigned_row["epnp_record_index"]),
        )
        source = by_pair[key]
        assigned[index] = dict(source)
        assigned[index]["assignment_method"] = "hungarian_one_to_one"

    selected = [
        row
        for row in assigned
        if common.passes_thresholds(
            row,
            args.max_translation_error_mm,
            args.max_rotation_error_deg,
            args.max_roll_error_deg,
            args.max_pitch_error_deg,
            args.max_yaw_error_deg,
        )
    ]
    selected.sort(
        key=lambda row: (
            int(row["scene_id"]),
            int(row["im_id"]),
            float(row["pairing_cost"]),
        )
    )
    common.write_csv(args.output_dir / "all_candidate_pairs.csv", all_pairs)
    common.write_csv(args.output_dir / "assigned_pairs.csv", assigned)
    common.write_csv(args.output_dir / "selected_samples.csv", selected)
    thresholds = {
        "min_score": args.min_score,
        "max_translation_error_mm": args.max_translation_error_mm,
        "max_rotation_error_deg": args.max_rotation_error_deg,
        "max_roll_error_deg": args.max_roll_error_deg,
        "max_pitch_error_deg": args.max_pitch_error_deg,
        "max_yaw_error_deg": args.max_yaw_error_deg,
    }
    report = common.selection_report(
        predictions_path=args.gigapose_predictions,
        dataset_dir=args.dataset_dir,
        epnp_root=args.epnp_root,
        output_dir=args.output_dir,
        comparison_mode=common.POST_CALIBRATION_MODE,
        predictions=predictions,
        labels=labels,
        pair_counts=pair_counts,
        assigned=assigned,
        selected=selected,
        skipped_labels=skipped_labels,
        C=C,
        thresholds=thresholds,
        extra={
            "optimized_extrinsics": str(calibration_path),
            "calibration_camera_id": calibration["camera_id"],
            "calibration_recommended_for_reselection": calibration.get(
                "recommended_for_reselection"
            ),
            "prediction_translation_unit": args.prediction_translation_unit,
            "corrected_target_equation": (
                "inv(T_lidar_camera_optimized) @ inv(T_map_lidar) "
                "@ T_map_object_raw @ T_object_raw_object_centered"
            ),
        },
    )
    common.save_json(args.output_dir / "selection_report.json", report)
    if not calibration.get("recommended_for_reselection", False):
        print(
            "WARNING: calibration JSON is not marked recommended_for_reselection; "
            "inspect its before/after error CSVs."
        )
    print(
        f"Selected {len(selected)}/{len(assigned)} calibrated one-to-one pairs. "
        "No empirical prediction-label alignment was used."
    )


if __name__ == "__main__":
    main()
