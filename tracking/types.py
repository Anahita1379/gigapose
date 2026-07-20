"""Typed records shared by the tracking modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class TrackMode(str, Enum):
    NORMAL = "normal"
    UNCERTAIN = "uncertain"
    LOST = "lost"


@dataclass
class Detection:
    detection_id: int
    bbox_xywh: np.ndarray
    mask: np.ndarray
    obj_id: int = 1
    external_id: int | None = None
    score: float = 1.0

    @property
    def center(self) -> np.ndarray:
        x, y, width, height = self.bbox_xywh.astype(float)
        return np.asarray([x + width * 0.5, y + height * 0.5])


@dataclass
class FrameData:
    scene_id: int
    im_id: int
    image: np.ndarray
    K: np.ndarray
    detections: list[Detection]
    depth_m: np.ndarray | None = None
    image_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def gray(self) -> np.ndarray:
        if self.image.ndim == 2:
            return self.image
        # Avoid importing OpenCV in the data model.
        return np.round(
            0.299 * self.image[..., 0]
            + 0.587 * self.image[..., 1]
            + 0.114 * self.image[..., 2]
        ).astype(np.uint8)


@dataclass
class PoseHypothesis:
    pose: np.ndarray
    source: str
    measurement_score: float = 0.5
    obj_id: int = 1
    prediction_instance_id: int | None = None
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy(self, *, source: str | None = None) -> "PoseHypothesis":
        return PoseHypothesis(
            pose=self.pose.copy(),
            source=source or self.source,
            measurement_score=float(self.measurement_score),
            obj_id=int(self.obj_id),
            prediction_instance_id=self.prediction_instance_id,
            rank=int(self.rank),
            metadata=dict(self.metadata),
        )


@dataclass
class ScoreBreakdown:
    total_error: float
    total_score: float
    confidence: float
    feature_error: float
    silhouette_error: float
    silhouette_iou: float
    edge_error: float
    depth_error: float
    bbox_error: float
    bbox_iou: float
    motion_error: float
    motion_translation_m: float
    motion_rotation_deg: float
    center_dx_frac: float
    center_dy_frac: float
    log_area_ratio: float
    log_depth_ratio: float
    rendered_pixels: int
    valid_depth_pixels: int


@dataclass
class EvaluatedHypothesis:
    hypothesis: PoseHypothesis
    score: ScoreBreakdown
    rendered_mask: np.ndarray | None = None
    rendered_depth_m: np.ndarray | None = None


@dataclass
class Track:
    track_id: int
    obj_id: int
    pose: np.ndarray
    previous_pose: np.ndarray | None
    bbox_xywh: np.ndarray
    confidence: float
    mode: TrackMode
    source: str
    scene_id: int
    im_id: int
    age: int = 1
    missed: int = 0
    frames_since_global: int = 0
    previous_gray: np.ndarray | None = None
    previous_mask: np.ndarray | None = None
    last_score: ScoreBreakdown | None = None
    history: list[np.ndarray] = field(default_factory=list)
    appearance_descriptor: np.ndarray | None = None
    external_id: int | None = None

    def propagated_pose(self) -> np.ndarray:
        from tracking.geometry import propagate_constant_velocity

        return propagate_constant_velocity(self.previous_pose, self.pose)
