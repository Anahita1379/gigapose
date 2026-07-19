"""Cheap sparse optical-flow pose propagation for normal tracking frames."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tracking.config import FlowConfig
from tracking.geometry import rotate_pose, shift_projected_center


@dataclass
class FlowMeasurement:
    affine: np.ndarray
    point_count: int
    inlier_count: int
    inlier_ratio: float
    confidence: float

    @property
    def scale(self) -> float:
        return float(np.linalg.norm(self.affine[:2, 0]))

    @property
    def angle_rad(self) -> float:
        return float(np.arctan2(self.affine[1, 0], self.affine[0, 0]))


class SparseFlowPoseUpdater:
    def __init__(self, config: FlowConfig):
        self.config = config

    def estimate(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
        previous_mask: np.ndarray,
    ) -> FlowMeasurement | None:
        if not self.config.enabled or previous_mask is None:
            return None
        mask_u8 = np.asarray(previous_mask, dtype=np.uint8) * 255
        points = cv2.goodFeaturesToTrack(
            np.asarray(previous_gray, dtype=np.uint8),
            maxCorners=self.config.max_corners,
            qualityLevel=self.config.quality_level,
            minDistance=self.config.min_distance_px,
            mask=mask_u8,
        )
        if points is None or len(points) < self.config.min_points:
            return None
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(
            np.asarray(previous_gray, dtype=np.uint8),
            np.asarray(current_gray, dtype=np.uint8),
            points,
            None,
            winSize=(self.config.window_size, self.config.window_size),
            maxLevel=self.config.pyramid_levels,
        )
        if tracked is None or status is None:
            return None
        keep = status.reshape(-1).astype(bool)
        source = points.reshape(-1, 2)[keep]
        target = tracked.reshape(-1, 2)[keep]
        if len(source) < self.config.min_points:
            return None
        affine, inliers = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.config.ransac_threshold_px,
        )
        if affine is None or inliers is None:
            return None
        inlier_count = int(inliers.sum())
        ratio = inlier_count / max(len(source), 1)
        scale = float(np.linalg.norm(affine[:2, 0]))
        if (
            ratio < self.config.min_inlier_ratio
            or not 0.5 <= scale <= 2.0
            or not np.isfinite(affine).all()
        ):
            return None
        confidence = float(
            np.clip(ratio * min(inlier_count / 30.0, 1.0), 0.0, 1.0)
        )
        return FlowMeasurement(
            affine=np.vstack([affine, [0.0, 0.0, 1.0]]),
            point_count=len(source),
            inlier_count=inlier_count,
            inlier_ratio=ratio,
            confidence=confidence,
        )

    def update_pose(
        self,
        pose: np.ndarray,
        K: np.ndarray,
        measurement: FlowMeasurement,
    ) -> np.ndarray:
        translation = np.asarray(pose[:3, 3], dtype=float)
        if translation[2] <= 1e-6:
            return pose.copy()
        center_h = np.asarray(K, dtype=float) @ translation
        center_h /= center_h[2]
        transformed = measurement.affine @ center_h
        transformed /= max(float(transformed[2]), 1e-9)
        output = shift_projected_center(
            pose,
            K,
            transformed[:2] - center_h[:2],
            -np.log(max(measurement.scale, 1e-6)),
        )
        output = rotate_pose(
            output,
            np.asarray([0.0, 0.0, measurement.angle_rad]),
            side="left",
        )
        return output
