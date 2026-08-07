"""Static measurement/context features for learned filtering."""

from __future__ import annotations

import numpy as np


CONTEXT_FEATURE_NAMES = (
    "measurement_score",
    "bbox_center_x_fraction",
    "bbox_center_y_fraction",
    "bbox_width_fraction",
    "bbox_height_fraction",
    "bbox_area_fraction",
    "measurement_depth_m",
)


def context_features(frame, detection, measurement) -> np.ndarray:
    height, width = frame.image.shape[:2]
    x, y, box_width, box_height = np.asarray(detection.bbox_xywh, dtype=float)
    return np.asarray(
        [
            float(measurement.measurement_score),
            (x + 0.5 * box_width) / max(width, 1),
            (y + 0.5 * box_height) / max(height, 1),
            box_width / max(width, 1),
            box_height / max(height, 1),
            box_width * box_height / max(width * height, 1),
            float(measurement.pose[2, 3]),
        ],
        dtype=np.float32,
    )
