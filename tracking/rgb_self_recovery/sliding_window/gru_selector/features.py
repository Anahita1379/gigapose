"""Candidate and frame features used by the causal GRU selector."""

from __future__ import annotations

import math

import numpy as np

from tracking.geometry import rotation_error_deg, translation_error_m


CANDIDATE_FEATURE_NAMES = (
    "confidence",
    "quality",
    "silhouette_iou",
    "total_error",
    "delta_center_norm_px",
    "delta_log_depth",
    "delta_rotation_deg",
    "translation_from_gigapose_m",
    "rotation_from_gigapose_deg",
    "tx_over_z",
    "ty_over_z",
    "log_depth_m",
    "rotation_6d_0",
    "rotation_6d_1",
    "rotation_6d_2",
    "rotation_6d_3",
    "rotation_6d_4",
    "rotation_6d_5",
    "is_gigapose_baseline",
    "is_flip_hypothesis",
    "is_depth_hypothesis",
    "is_center_hypothesis",
    "is_rotation_hypothesis",
)

FRAME_FEATURE_NAMES = (
    "bbox_center_x_fraction",
    "bbox_center_y_fraction",
    "bbox_width_fraction",
    "bbox_height_fraction",
    "bbox_area_fraction",
    "detection_score",
    "delta_time_s",
    "gigapose_depth_m",
)


def candidate_features(result, baseline_pose: np.ndarray, *, baseline: bool) -> np.ndarray:
    pose = np.asarray(result.pose, dtype=float)
    z = max(float(pose[2, 3]), 1e-6)
    rotation_6d = pose[:3, :2].T.reshape(-1)
    source = str(result.source).lower()
    values = [
        float(result.confidence),
        float(result.quality),
        float(result.silhouette_iou),
        float(result.total_error),
        float(np.linalg.norm(result.delta_center_crop_px)),
        float(result.delta_log_depth),
        math.degrees(float(np.linalg.norm(result.delta_rotation_rad))),
        translation_error_m(pose, baseline_pose),
        rotation_error_deg(pose, baseline_pose),
        float(pose[0, 3] / z),
        float(pose[1, 3] / z),
        math.log(z),
        *rotation_6d.tolist(),
        float(baseline),
        float("flip180" in source),
        float("logz" in source),
        float("align_detection" in source or "|uv" in source),
        float("|r0" in source or "|r1" in source or "yaw" in source),
    ]
    output = np.asarray(values, dtype=np.float32)
    if output.shape != (len(CANDIDATE_FEATURE_NAMES),):
        raise RuntimeError("Candidate feature schema mismatch")
    return output


def frame_features(frame, detection, baseline_pose: np.ndarray, delta_time_s: float) -> np.ndarray:
    height, width = frame.image.shape[:2]
    x, y, box_width, box_height = np.asarray(detection.bbox_xywh, dtype=float)
    values = [
        (x + 0.5 * box_width) / max(width, 1),
        (y + 0.5 * box_height) / max(height, 1),
        box_width / max(width, 1),
        box_height / max(height, 1),
        box_width * box_height / max(width * height, 1),
        float(detection.score),
        max(float(delta_time_s), 0.0),
        float(baseline_pose[2, 3]),
    ]
    return np.asarray(values, dtype=np.float32)


def oracle_costs(
    poses: np.ndarray,
    ground_truth: np.ndarray,
    valid: np.ndarray,
    *,
    translation_scale_m: float,
    rotation_scale_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(poses)
    translation = np.full(count, np.inf, dtype=np.float32)
    rotation = np.full(count, np.inf, dtype=np.float32)
    for index in np.flatnonzero(valid):
        translation[index] = translation_error_m(poses[index], ground_truth)
        rotation[index] = rotation_error_deg(poses[index], ground_truth)
    cost = translation / translation_scale_m + rotation / rotation_scale_deg
    return cost, translation, rotation
