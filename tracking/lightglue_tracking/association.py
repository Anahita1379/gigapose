"""LightGlue-aware Hungarian association without changing the base tracker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracking.association import (
    appearance_distance,
    bbox_iou,
    detection_appearance_descriptor,
    mask_iou,
    normalized_center_distance,
)
from tracking.config import AssociationConfig
from tracking.lightglue_tracking.config import LightGlueTrackingConfig
from tracking.lightglue_tracking.matcher import (
    ALIKEDLightGlue,
    MatchResult,
    RegionFeatures,
)
from tracking.types import Detection, Track, TrackMode


@dataclass
class AssociationResult:
    matches: list[tuple[int, int]]
    unmatched_tracks: list[int]
    unmatched_detections: list[int]
    pair_matches: dict[tuple[int, int], MatchResult]
    attempted_lightglue_pairs: int


def _should_match_pair(
    policy: str,
    track: Track,
    track_index: int,
    detection_index: int,
    valid_pairs: np.ndarray,
) -> bool:
    if policy == "off":
        return False
    if policy == "always":
        return True
    if policy == "uncertain_lost":
        return track.mode != TrackMode.NORMAL
    if policy == "ambiguous":
        return bool(
            track.mode != TrackMode.NORMAL
            or int(valid_pairs[track_index].sum()) > 1
            or int(valid_pairs[:, detection_index].sum()) > 1
        )
    raise ValueError(f"Unknown LightGlue association policy: {policy}")


def associate_tracks(
    tracks: Sequence[Track],
    detections: Sequence[Detection],
    image_shape: tuple[int, int],
    base_config: AssociationConfig,
    lightglue_config: LightGlueTrackingConfig,
    matcher: ALIKEDLightGlue,
    track_features: Mapping[int, RegionFeatures],
    detection_features: Sequence[RegionFeatures | None],
    image: np.ndarray | None = None,
) -> AssociationResult:
    """Associate tracks while adding verified LightGlue quality to ambiguous pairs."""

    if not tracks or not detections:
        return AssociationResult(
            matches=[],
            unmatched_tracks=list(range(len(tracks))),
            unmatched_detections=list(range(len(detections))),
            pair_matches={},
            attempted_lightglue_pairs=0,
        )
    detection_descriptors = (
        [detection_appearance_descriptor(image, item) for item in detections]
        if base_config.identity_enabled and base_config.appearance_weight > 0
        else [None] * len(detections)
    )
    valid_pairs = np.zeros((len(tracks), len(detections)), dtype=bool)
    base_weighted = np.full((len(tracks), len(detections)), np.nan, dtype=float)
    base_weights = np.zeros_like(base_weighted)

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
                track.external_id is not None
                and detection.external_id is not None
            )
            external_id_match = bool(
                external_ids_available
                and int(track.external_id) == int(detection.external_id)
            )
            if (
                base_config.identity_enabled
                and base_config.use_external_id
                and base_config.external_id_strict
                and external_ids_available
                and not external_id_match
            ):
                continue
            iou = bbox_iou(track_bbox, detection.bbox_xywh)
            distance = normalized_center_distance(
                track_center, detection.center, image_shape
            )
            bypass_spatial_gate = bool(
                base_config.identity_enabled
                and base_config.use_external_id
                and base_config.external_id_strict
                and external_id_match
            )
            if (
                not bypass_spatial_gate
                and iou < base_config.min_iou
                and distance > base_config.max_center_distance_frac
            ):
                continue
            weighted_cost = base_config.iou_weight * (1.0 - iou)
            weighted_cost += base_config.center_weight * min(
                distance
                / max(base_config.max_center_distance_frac, 1e-6),
                1.0,
            )
            total_weight = base_config.iou_weight + base_config.center_weight
            if base_config.identity_enabled:
                if (
                    base_config.use_external_id
                    and external_ids_available
                    and base_config.external_id_weight > 0
                ):
                    weighted_cost += base_config.external_id_weight * (
                        0.0 if external_id_match else 1.0
                    )
                    total_weight += base_config.external_id_weight
                previous_mask_iou = mask_iou(
                    track.previous_mask, detection.mask
                )
                if (
                    previous_mask_iou is not None
                    and base_config.mask_iou_weight > 0
                ):
                    weighted_cost += base_config.mask_iou_weight * (
                        1.0 - previous_mask_iou
                    )
                    total_weight += base_config.mask_iou_weight
                descriptor_distance = appearance_distance(
                    track.appearance_descriptor,
                    detection_descriptors[detection_index],
                )
                if (
                    descriptor_distance is not None
                    and base_config.appearance_weight > 0
                ):
                    weighted_cost += (
                        base_config.appearance_weight * descriptor_distance
                    )
                    total_weight += base_config.appearance_weight
            valid_pairs[track_index, detection_index] = True
            base_weighted[track_index, detection_index] = weighted_cost
            base_weights[track_index, detection_index] = total_weight

    cost = np.full((len(tracks), len(detections)), 1e3, dtype=float)
    pair_matches: dict[tuple[int, int], MatchResult] = {}
    attempted = 0
    for track_index, track in enumerate(tracks):
        for detection_index, _detection in enumerate(detections):
            if not valid_pairs[track_index, detection_index]:
                continue
            weighted_cost = float(base_weighted[track_index, detection_index])
            total_weight = float(base_weights[track_index, detection_index])
            should_use = bool(
                lightglue_config.enabled
                and lightglue_config.association_enabled
                and lightglue_config.association_weight > 0
                and _should_match_pair(
                    lightglue_config.association_policy,
                    track,
                    track_index,
                    detection_index,
                    valid_pairs,
                )
            )
            previous = track_features.get(track.track_id)
            current = detection_features[detection_index]
            if should_use and previous is not None and current is not None:
                attempted += 1
                result = matcher.match(previous, current)
                pair_matches[(track_index, detection_index)] = result
                quality = (
                    result.similarity.quality
                    if result.similarity is not None
                    else 0.0
                )
                weighted_cost += lightglue_config.association_weight * (
                    1.0 - quality
                )
                total_weight += lightglue_config.association_weight
            cost[track_index, detection_index] = weighted_cost / max(
                total_weight, 1e-9
            )

    rows, columns = linear_sum_assignment(cost)
    matches = [
        (int(row), int(column))
        for row, column in zip(rows, columns)
        if cost[row, column] <= base_config.max_cost
    ]
    matched_tracks = {row for row, _ in matches}
    matched_detections = {column for _, column in matches}
    return AssociationResult(
        matches=matches,
        unmatched_tracks=[
            index for index in range(len(tracks)) if index not in matched_tracks
        ],
        unmatched_detections=[
            index
            for index in range(len(detections))
            if index not in matched_detections
        ],
        pair_matches=pair_matches,
        attempted_lightglue_pairs=attempted,
    )
