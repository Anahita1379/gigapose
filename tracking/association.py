"""Hungarian association for detections, tracks, and fresh pose groups."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracking.config import AssociationConfig
from tracking.geometry import project_center
from tracking.types import Detection, PoseHypothesis, Track


def bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    ax, ay, aw, ah = np.asarray(first, dtype=float)
    bx, by, bw, bh = np.asarray(second, dtype=float)
    intersection_w = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    intersection_h = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    intersection = intersection_w * intersection_h
    union = aw * ah + bw * bh - intersection
    return float(intersection / union) if union > 0 else 0.0


def normalized_center_distance(
    first: np.ndarray, second: np.ndarray, image_shape: tuple[int, int]
) -> float:
    height, width = image_shape
    diagonal = max(float(np.hypot(width, height)), 1.0)
    return float(np.linalg.norm(first - second) / diagonal)


def associate_tracks(
    tracks: Sequence[Track],
    detections: Sequence[Detection],
    image_shape: tuple[int, int],
    config: AssociationConfig,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))
    cost = np.full((len(tracks), len(detections)), 1e3, dtype=float)
    for track_index, track in enumerate(tracks):
        track_bbox = np.asarray(track.bbox_xywh, dtype=float)
        track_center = np.asarray(
            [
                track_bbox[0] + 0.5 * track_bbox[2],
                track_bbox[1] + 0.5 * track_bbox[3],
            ]
        )
        for detection_index, detection in enumerate(detections):
            if track.obj_id != detection.obj_id:
                continue
            iou = bbox_iou(track_bbox, detection.bbox_xywh)
            distance = normalized_center_distance(
                track_center, detection.center, image_shape
            )
            if iou < config.min_iou and distance > config.max_center_distance_frac:
                continue
            cost[track_index, detection_index] = (
                config.iou_weight * (1.0 - iou)
                + config.center_weight
                * min(distance / max(config.max_center_distance_frac, 1e-6), 1.0)
            )
    row_indices, column_indices = linear_sum_assignment(cost)
    matches = [
        (int(row), int(column))
        for row, column in zip(row_indices, column_indices)
        if cost[row, column] <= config.max_cost
    ]
    matched_tracks = {row for row, _ in matches}
    matched_detections = {column for _, column in matches}
    return (
        matches,
        [index for index in range(len(tracks)) if index not in matched_tracks],
        [
            index
            for index in range(len(detections))
            if index not in matched_detections
        ],
    )


def assign_prediction_groups(
    groups: Sequence[Sequence[PoseHypothesis]],
    detections: Sequence[Detection],
    K: np.ndarray,
    image_shape: tuple[int, int],
) -> dict[int, list[PoseHypothesis]]:
    """Assign each GigaPose instance group to one current detection."""

    if not groups or not detections:
        return {}
    height, width = image_shape
    diagonal = max(float(np.hypot(width, height)), 1.0)
    cost = np.full((len(groups), len(detections)), 1e3, dtype=float)
    for group_index, group in enumerate(groups):
        if not group:
            continue
        center, valid = project_center(group[0].pose, K)
        if not valid:
            continue
        for detection_index, detection in enumerate(detections):
            if group[0].obj_id != detection.obj_id:
                continue
            cost[group_index, detection_index] = float(
                np.linalg.norm(center - detection.center) / diagonal
            )
    rows, columns = linear_sum_assignment(cost)
    assignments: dict[int, list[PoseHypothesis]] = {}
    for row, column in zip(rows, columns):
        if cost[row, column] <= 0.5:
            assignments[int(column)] = [item.copy() for item in groups[int(row)]]
    return assignments
