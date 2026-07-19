"""Diagnostic overlays for the adaptive tracking run."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tracking.tracker import TrackingFrameResult
from tracking.types import FrameData


COLORS = (
    (30, 220, 80),
    (255, 170, 30),
    (60, 160, 255),
    (220, 80, 220),
    (255, 220, 40),
)


def save_tracking_overlay(
    frame: FrameData, result: TrackingFrameResult, path: Path
) -> None:
    canvas = np.asarray(frame.image, dtype=np.uint8).copy()
    for instance in result.instances:
        track_id = instance.track.track_id
        color = COLORS[track_id % len(COLORS)]
        rendered_mask = instance.chosen.rendered_mask
        if rendered_mask is not None:
            mask = np.asarray(rendered_mask, dtype=bool)
            blend = canvas[mask].astype(np.float32) * 0.55 + np.asarray(
                color, dtype=np.float32
            ) * 0.45
            canvas[mask] = np.clip(blend, 0, 255).astype(np.uint8)
            edge = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            canvas[edge] = color
        x, y, width, height = np.round(instance.detection.bbox_xywh).astype(int)
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (255, 220, 0), 2)
        label = (
            f"T{track_id} {instance.track.mode.value} "
            f"{instance.track.confidence:.2f} {instance.chosen.hypothesis.source}"
        )
        cv2.putText(
            canvas,
            label,
            (max(0, x), max(18, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    # frame.image is RGB while OpenCV output is BGR.
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
