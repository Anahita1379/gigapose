"""Hungarian association for detections, tracks, and fresh pose groups."""

from __future__ import annotations

from typing import Sequence

import cv2
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


def mask_iou(first: np.ndarray | None, second: np.ndarray | None) -> float | None:
    """Return mask IoU, or ``None`` when the masks cannot be compared."""

    if first is None or second is None:
        return None
    first_bool = np.asarray(first, dtype=bool)
    second_bool = np.asarray(second, dtype=bool)
    if first_bool.shape != second_bool.shape:
        return None
    union = int(np.logical_or(first_bool, second_bool).sum())
    if union == 0:
        return None
    intersection = int(np.logical_and(first_bool, second_bool).sum())
    return float(intersection / union)


def detection_appearance_descriptor(
    image: np.ndarray | None, detection: Detection
) -> np.ndarray | None:
    """Compute a compact color histogram inside one detected instance."""

    if image is None or image.ndim != 3 or image.shape[2] < 3:
        return None
    height, width = image.shape[:2]
    x, y, box_width, box_height = np.asarray(
        detection.bbox_xywh, dtype=float
    )
    x0 = int(np.clip(np.floor(x), 0, width))
    y0 = int(np.clip(np.floor(y), 0, height))
    x1 = int(np.clip(np.ceil(x + box_width), 0, width))
    y1 = int(np.clip(np.ceil(y + box_height), 0, height))
    if x1 <= x0 or y1 <= y0:
        return None
    crop = np.asarray(image[y0:y1, x0:x1, :3], dtype=np.uint8)
    mask = np.asarray(detection.mask, dtype=bool)
    if mask.shape == (height, width):
        crop_mask = mask[y0:y1, x0:x1].astype(np.uint8) * 255
    else:
        crop_mask = np.ones(crop.shape[:2], dtype=np.uint8) * 255
    if int(np.count_nonzero(crop_mask)) < 8:
        return None
    # FrameData images are RGB. Hue provides identity discrimination while the
    # saturation/value axes retain useful evidence for low-saturation liveries.
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    histogram = cv2.calcHist(
        [hsv], [0, 1], crop_mask, [24, 16], [0, 180, 0, 256]
    ).astype(np.float32)
    total = float(histogram.sum())
    if total <= 0:
        return None
    return (histogram / total).reshape(-1)


def appearance_distance(
    first: np.ndarray | None, second: np.ndarray | None
) -> float | None:
    """Bhattacharyya distance between normalized appearance histograms."""

    if first is None or second is None:
        return None
    first_array = np.asarray(first, dtype=np.float32).reshape(-1)
    second_array = np.asarray(second, dtype=np.float32).reshape(-1)
    if first_array.shape != second_array.shape or first_array.size == 0:
        return None
    return float(
        np.clip(
            cv2.compareHist(
                first_array,
                second_array,
                cv2.HISTCMP_BHATTACHARYYA,
            ),
            0.0,
            1.0,
        )
    )


def associate_tracks(
    tracks: Sequence[Track],
    detections: Sequence[Detection],
    image_shape: tuple[int, int],
    config: AssociationConfig,
    image: np.ndarray | None = None,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))
    detection_descriptors = (
        [detection_appearance_descriptor(image, item) for item in detections]
        if config.identity_enabled and config.appearance_weight > 0
        else [None] * len(detections)
    )
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
            external_ids_available = (
                track.external_id is not None and detection.external_id is not None
            )
            external_id_match = (
                external_ids_available
                and int(track.external_id) == int(detection.external_id)
            )
            if (
                config.identity_enabled
                and config.use_external_id
                and config.external_id_strict
                and external_ids_available
                and not external_id_match
            ):
                continue
            iou = bbox_iou(track_bbox, detection.bbox_xywh)
            distance = normalized_center_distance(
                track_center, detection.center, image_shape
            )
            may_bypass_spatial_gate = (
                config.identity_enabled
                and config.use_external_id
                and config.external_id_strict
                and external_id_match
            )
            if (
                not may_bypass_spatial_gate
                and iou < config.min_iou
                and distance > config.max_center_distance_frac
            ):
                continue
            weighted_cost = config.iou_weight * (1.0 - iou) + (
                config.center_weight
                * min(
                    distance / max(config.max_center_distance_frac, 1e-6),
                    1.0,
                )
            )
            if not config.identity_enabled:
                cost[track_index, detection_index] = weighted_cost
                continue
            total_weight = config.iou_weight + config.center_weight
            if (
                config.use_external_id
                and external_ids_available
                and config.external_id_weight > 0
            ):
                weighted_cost += config.external_id_weight * (
                    0.0 if external_id_match else 1.0
                )
                total_weight += config.external_id_weight
            previous_mask_iou = mask_iou(track.previous_mask, detection.mask)
            if previous_mask_iou is not None and config.mask_iou_weight > 0:
                weighted_cost += config.mask_iou_weight * (
                    1.0 - previous_mask_iou
                )
                total_weight += config.mask_iou_weight
            descriptor_distance = appearance_distance(
                track.appearance_descriptor,
                detection_descriptors[detection_index],
            )
            if (
                descriptor_distance is not None
                and config.appearance_weight > 0
            ):
                weighted_cost += config.appearance_weight * descriptor_distance
                total_weight += config.appearance_weight
            cost[track_index, detection_index] = weighted_cost / max(
                total_weight, 1e-9
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
