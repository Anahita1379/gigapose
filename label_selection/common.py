"""Shared geometry and I/O for the absolute label-selection pipeline.

All transforms in this package use millimetres for translation.  The EPnP
labels use the centered object frame, while the GigaPose CSV uses the native
raw-CAD frame.  The only object-frame conversion permitted by this package is
the known CAD-center transform::

    T_cam_object_centered = T_cam_object_raw @ T_object_raw_object_centered

No transform is estimated from GigaPose/EPnP pairs before their errors are
measured.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from fine_tuning import select_real_label_candidates as legacy_predictions
from fine_tuning.select_real_label_candidates_with_camera_lidar_extrinsics import (
    load_epnp_extrinsic_labels,
)


COMPARISON_MODE = "absolute_known_center_v1"
POST_CALIBRATION_MODE = "absolute_optimized_camera_lidar_v1"
CALIBRATION_MODE = "label_selection_fixed_camera_lidar_v1"
DEFAULT_RAW_OBJECT_CENTER_M = (-0.2411941141, 0.0009010172, 0.3329219520)


def matrix_to_text(T: np.ndarray) -> str:
    return " ".join(f"{value:.12g}" for value in np.asarray(T, dtype=float).reshape(-1))


def text_to_matrix(value: str) -> np.ndarray:
    values = np.fromstring(
        str(value).strip().strip("[]").replace(",", " ").replace(";", " "),
        sep=" ",
        dtype=float,
    )
    if values.size != 16:
        raise ValueError(f"Expected 16 values for a 4x4 matrix, got {values.size}")
    return values.reshape(4, 4)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def rigid_transform_checks(T: np.ndarray, name: str) -> None:
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError(f"{name} is not a finite 4x4 transform")
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError(f"{name} has invalid homogeneous row {T[3].tolist()}")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=2e-3):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(R), 1.0, atol=2e-3):
        raise ValueError(f"{name} rotation determinant is not +1")


def object_center_transform(center_raw_m: Iterable[float]) -> np.ndarray:
    """Return T_object_raw_object_centered in millimetres.

    If ``p_raw = p_centered + center_raw``, the homogeneous transform from the
    centered coordinates to raw coordinates is ``[I, center_raw]``.
    """

    center_mm = np.asarray(tuple(center_raw_m), dtype=float).reshape(3) * 1000.0
    T = np.eye(4, dtype=float)
    T[:3, 3] = center_mm
    return T


def raw_pose_to_centered(T_cam_object_raw: np.ndarray, C: np.ndarray) -> np.ndarray:
    T = np.asarray(T_cam_object_raw, dtype=float).reshape(4, 4) @ np.asarray(
        C, dtype=float
    ).reshape(4, 4)
    rigid_transform_checks(T, "T_cam_object_centered")
    return T


def rotation_error_deg(R_pred: np.ndarray, R_target: np.ndarray) -> float:
    """Geodesic SO(3) angle using atan2(||vee(R-R^T)||/2, trace)."""

    relative = np.asarray(R_target, dtype=float).reshape(3, 3).T @ np.asarray(
        R_pred, dtype=float
    ).reshape(3, 3)
    sin_theta = 0.5 * np.linalg.norm(
        np.array(
            [
                relative[2, 1] - relative[1, 2],
                relative[0, 2] - relative[2, 0],
                relative[1, 0] - relative[0, 1],
            ]
        )
    )
    cos_theta = 0.5 * (float(np.trace(relative)) - 1.0)
    return float(np.degrees(np.arctan2(sin_theta, cos_theta)))


def rotation_error_rpy_deg(
    R_pred: np.ndarray, R_target: np.ndarray
) -> tuple[float, float, float]:
    relative = np.asarray(R_target, dtype=float).reshape(3, 3).T @ np.asarray(
        R_pred, dtype=float
    ).reshape(3, 3)
    # scipy returns x/y/z intrinsic rotations.  These are diagnostic component
    # bounds; the invariant SO(3) angle above remains the primary rotation error.
    roll, pitch, yaw = Rotation.from_matrix(relative).as_euler("xyz", degrees=True)
    return abs(float(roll)), abs(float(pitch)), abs(float(yaw))


def pose_errors(
    T_pred: np.ndarray, T_target: np.ndarray
) -> tuple[float, float, float, float, float]:
    t_error = float(
        np.linalg.norm(
            np.asarray(T_pred, dtype=float)[:3, 3]
            - np.asarray(T_target, dtype=float)[:3, 3]
        )
    )
    r_error = rotation_error_deg(T_pred[:3, :3], T_target[:3, :3])
    roll, pitch, yaw = rotation_error_rpy_deg(
        T_pred[:3, :3], T_target[:3, :3]
    )
    return t_error, r_error, roll, pitch, yaw


def load_predictions(
    predictions_path: Path,
    dataset_dir: Path,
    frame_map_path: Path | None,
    match_key: str,
    min_score: float,
    translation_unit: str,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], dict[str, Any]]]:
    frame_map = legacy_predictions.load_frame_map(frame_map_path, dataset_dir)
    raw_predictions = legacy_predictions.collapse_top_predictions_per_detection(
        legacy_predictions.load_prediction_rows(predictions_path)
    )
    if translation_unit == "m":
        translation_scale = 1000.0
    elif translation_unit == "mm":
        translation_scale = 1.0
    elif translation_unit == "auto":
        translation_scale = legacy_predictions.infer_prediction_translation_scale(
            raw_predictions
        )
    else:
        raise ValueError(f"Unknown prediction translation unit: {translation_unit}")
    predictions: list[dict[str, Any]] = []
    for raw in raw_predictions:
        prediction = dict(raw)
        prediction["t_mm"] = (
            np.asarray(raw["t"], dtype=float).reshape(3) * translation_scale
        )
        prediction["T_gigapose"] = legacy_predictions.make_transform(
            raw["R"], prediction["t_mm"]
        )
        prediction["match_key"] = legacy_predictions.prediction_key(
            raw, frame_map, match_key
        )
        predictions.append(prediction)
    predictions = [
        prediction
        for prediction in predictions
        if float(prediction.get("score", 0.0)) >= min_score
    ]
    return predictions, frame_map


def load_labels(
    epnp_root: Path,
    epnp_glob: str,
    epnp_key_field: str | None,
    epnp_key_prefix: str,
    strip_trailing_instance_id: bool,
    epnp_map_pose_key: str,
    epnp_camera_pose_key: str,
    epnp_map_pose_unit: str,
    epnp_camera_pose_unit: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    return load_epnp_extrinsic_labels(
        roots=[epnp_root],
        glob_pattern=epnp_glob,
        key_field=epnp_key_field,
        key_prefix=epnp_key_prefix,
        strip_trailing_instance_id=strip_trailing_instance_id,
        map_pose_key=epnp_map_pose_key,
        camera_pose_key=epnp_camera_pose_key,
        map_pose_unit=epnp_map_pose_unit,
        camera_pose_unit=epnp_camera_pose_unit,
    )


def group_by_match_key(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["match_key"])].append(row)
    return dict(grouped)


def _candidate_row(
    pred: dict[str, Any],
    label: dict[str, Any],
    C: np.ndarray,
    pairing_translation_scale_mm: float,
    pairing_rotation_scale_deg: float,
    comparison_mode: str,
    epnp_target: np.ndarray | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    T_gigapose_native = np.asarray(pred["T_gigapose"], dtype=float)
    T_gigapose_centered = raw_pose_to_centered(T_gigapose_native, C)
    T_epnp_original = np.asarray(label["T_camera_object_original"], dtype=float)
    T_epnp_target = (
        np.asarray(epnp_target, dtype=float)
        if epnp_target is not None
        else T_epnp_original
    )
    t_error, r_error, roll, pitch, yaw = pose_errors(
        T_gigapose_centered, T_epnp_target
    )
    cost = (
        t_error / max(pairing_translation_scale_mm, 1e-9)
        + r_error / max(pairing_rotation_scale_deg, 1e-9)
    )
    camera_id = str(label.get("metadata_camera") or "unknown")
    row: dict[str, Any] = {
        "comparison_mode": comparison_mode,
        "selection_stage": (
            "post_extrinsics"
            if comparison_mode == POST_CALIBRATION_MODE
            else "pre_extrinsics"
        ),
        "match_key": str(pred["match_key"]),
        "camera_id": camera_id,
        "scene_id": int(pred["scene_id"]),
        "im_id": int(pred["im_id"]),
        "obj_id": int(pred.get("obj_id", 1)),
        "instance_id": pred.get("instance_id", ""),
        "prediction_row_index": int(pred.get("row_index", -1)),
        "score": float(pred.get("score", 0.0)),
        "epnp_label_path": str(label["epnp_label_path"]),
        "epnp_record_index": int(label.get("epnp_record_index", 0)),
        "sample_metadata_path": str(label.get("metadata_path", "")),
        "pairing_cost": float(cost),
        "pairing_translation_scale_mm": float(pairing_translation_scale_mm),
        "pairing_rotation_scale_deg": float(pairing_rotation_scale_deg),
        "absolute_translation_error_mm": t_error,
        "absolute_rotation_error_deg": r_error,
        "absolute_roll_error_deg": roll,
        "absolute_pitch_error_deg": pitch,
        "absolute_yaw_error_deg": yaw,
        # Compatibility aliases are deliberately equal to the absolute values.
        "translation_error_mm": t_error,
        "rotation_error_deg": r_error,
        "roll_error_deg": roll,
        "pitch_error_deg": pitch,
        "yaw_error_deg": yaw,
        "T_object_raw_object_centered": matrix_to_text(C),
        "T_gigapose_native_raw_cad": matrix_to_text(T_gigapose_native),
        "T_gigapose_known_center_aligned": matrix_to_text(T_gigapose_centered),
        "T_epnp_camera_object_centered_original": matrix_to_text(T_epnp_original),
        "T_epnp_camera_object_centered_target": matrix_to_text(T_epnp_target),
        "T_epnp_map_object_raw": matrix_to_text(label["T_map_object"]),
        "T_map_lidar_metadata": matrix_to_text(label["T_map_lidar"]),
        "T_lidar_camera_prior_metadata": matrix_to_text(
            label["T_lidar_camera_prior"]
        ),
        # Explicit compatibility columns for generic downstream readers.
        "T_gigapose_cam_obj": matrix_to_text(T_gigapose_centered),
        "T_epnp_obj": matrix_to_text(T_epnp_target),
    }
    if extra:
        row.update(extra)
    return row


def build_and_assign_pairs(
    predictions: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    C: np.ndarray,
    pairing_translation_scale_mm: float,
    pairing_rotation_scale_deg: float,
    max_candidates_per_key: int,
    comparison_mode: str = COMPARISON_MODE,
    epnp_target_by_identity: dict[tuple[str, int], np.ndarray] | None = None,
    extra_by_identity: dict[tuple[str, int], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Create all pair costs and a one-to-one assignment for each image key."""

    preds_by_key = group_by_match_key(predictions)
    labels_by_key = group_by_match_key(labels)
    shared_keys = sorted(set(preds_by_key) & set(labels_by_key))
    all_rows: list[dict[str, Any]] = []
    assigned_rows: list[dict[str, Any]] = []
    counts = {
        "prediction_keys": len(preds_by_key),
        "label_keys": len(labels_by_key),
        "shared_keys": len(shared_keys),
        "unmatched_prediction_keys": len(set(preds_by_key) - set(labels_by_key)),
        "unmatched_label_keys": len(set(labels_by_key) - set(preds_by_key)),
    }

    for key in shared_keys:
        key_predictions = sorted(
            preds_by_key[key],
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )[:max_candidates_per_key]
        key_labels = labels_by_key[key][:max_candidates_per_key]
        if not key_predictions or not key_labels:
            continue

        cost = np.empty((len(key_predictions), len(key_labels)), dtype=float)
        row_grid: list[list[dict[str, Any]]] = []
        for pred_index, pred in enumerate(key_predictions):
            row_grid.append([])
            for label_index, label in enumerate(key_labels):
                identity = (
                    str(label["epnp_label_path"]),
                    int(label.get("epnp_record_index", 0)),
                )
                target = (
                    epnp_target_by_identity.get(identity)
                    if epnp_target_by_identity is not None
                    else None
                )
                if epnp_target_by_identity is not None and target is None:
                    cost[pred_index, label_index] = 1e12
                    row_grid[pred_index].append({})
                    continue
                row = _candidate_row(
                    pred,
                    label,
                    C,
                    pairing_translation_scale_mm,
                    pairing_rotation_scale_deg,
                    comparison_mode,
                    epnp_target=target,
                    extra=(
                        extra_by_identity.get(identity, {})
                        if extra_by_identity is not None
                        else None
                    ),
                )
                row["pair_prediction_index_within_key"] = pred_index
                row["pair_label_index_within_key"] = label_index
                cost[pred_index, label_index] = float(row["pairing_cost"])
                row_grid[pred_index].append(row)
                all_rows.append(row)

        pred_indices, label_indices = linear_sum_assignment(cost)
        for pred_index, label_index in zip(pred_indices.tolist(), label_indices.tolist()):
            if cost[pred_index, label_index] >= 1e11:
                continue
            assigned = dict(row_grid[pred_index][label_index])
            assigned["assignment_method"] = "hungarian_one_to_one"
            assigned_rows.append(assigned)

    counts["all_candidate_pairs"] = len(all_rows)
    counts["assigned_pairs"] = len(assigned_rows)
    return all_rows, assigned_rows, counts


def passes_thresholds(
    row: dict[str, Any],
    max_translation_error_mm: float,
    max_rotation_error_deg: float,
    max_roll_error_deg: float | None,
    max_pitch_error_deg: float | None,
    max_yaw_error_deg: float | None,
) -> bool:
    checks = [
        float(row["absolute_translation_error_mm"]) <= max_translation_error_mm,
        float(row["absolute_rotation_error_deg"]) <= max_rotation_error_deg,
    ]
    if max_roll_error_deg is not None:
        checks.append(float(row["absolute_roll_error_deg"]) <= max_roll_error_deg)
    if max_pitch_error_deg is not None:
        checks.append(float(row["absolute_pitch_error_deg"]) <= max_pitch_error_deg)
    if max_yaw_error_deg is not None:
        checks.append(float(row["absolute_yaw_error_deg"]) <= max_yaw_error_deg)
    return all(checks)


def summarize_numeric(rows: list[dict[str, Any]], field: str) -> dict[str, float | int]:
    values = np.asarray(
        [
            float(row[field])
            for row in rows
            if row.get(field) not in ("", None)
            and np.isfinite(float(row[field]))
        ],
        dtype=float,
    )
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def selection_report(
    *,
    predictions_path: Path,
    dataset_dir: Path,
    epnp_root: Path,
    output_dir: Path,
    comparison_mode: str,
    predictions: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    pair_counts: dict[str, int],
    assigned: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    skipped_labels: dict[str, int],
    C: np.ndarray,
    thresholds: dict[str, float | None],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "format": "label_selection_report_v1",
        "comparison_mode": comparison_mode,
        "empirical_prediction_label_alignment_used": False,
        "predictions": str(predictions_path),
        "dataset_dir": str(dataset_dir),
        "epnp_root": str(epnp_root),
        "output_dir": str(output_dir),
        "predictions_after_score_filter": len(predictions),
        "prediction_translation_unit": extra.get("prediction_translation_unit")
        if extra
        else None,
        "epnp_labels_loaded": len(labels),
        "epnp_labels_skipped": skipped_labels,
        **pair_counts,
        "selected_pairs": len(selected),
        "thresholds": thresholds,
        "T_object_raw_object_centered_mm": np.asarray(C).tolist(),
        "assigned_absolute_translation_error_mm": summarize_numeric(
            assigned, "absolute_translation_error_mm"
        ),
        "assigned_absolute_rotation_error_deg": summarize_numeric(
            assigned, "absolute_rotation_error_deg"
        ),
        "selected_absolute_translation_error_mm": summarize_numeric(
            selected, "absolute_translation_error_mm"
        ),
        "selected_absolute_rotation_error_deg": summarize_numeric(
            selected, "absolute_rotation_error_deg"
        ),
        "equations": {
            "known_center_conversion": (
                "T_cam_object_centered = "
                "T_cam_object_raw @ T_object_raw_object_centered"
            ),
            "translation_error": (
                "||t_gigapose_centered - t_epnp_centered||_2"
            ),
            "rotation_error": (
                "angle(R_epnp_centered^T @ R_gigapose_centered)"
            ),
        },
    }
    if extra:
        report.update(extra)
    return report


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
