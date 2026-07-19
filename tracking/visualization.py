"""Diagnostic overlays for the adaptive tracking run."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from tracking.tracker import TrackingFrameResult
from tracking.types import FrameData, PoseHypothesis


TRACKED_COLOR = (30, 220, 80)
ORIGINAL_COLOR = (255, 70, 70)
DETECTION_COLOR = (255, 220, 40)


def _bbox_from_mask(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if xs.size == 0:
        return None
    return np.asarray(
        [
            int(xs.min()),
            int(ys.min()),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        ]
    )


def _draw_mask(
    canvas: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != canvas.shape[:2] or not mask.any():
        return
    blend = canvas[mask].astype(np.float32) * (1.0 - alpha) + np.asarray(
        color, dtype=np.float32
    ) * alpha
    canvas[mask] = np.clip(blend, 0, 255).astype(np.uint8)
    edge = mask & ~cv2.erode(
        mask.astype(np.uint8), np.ones((3, 3), np.uint8)
    ).astype(bool)
    canvas[edge] = color


def _draw_bbox(
    canvas: np.ndarray,
    bbox_xywh: np.ndarray | None,
    color: tuple[int, int, int],
    width: int,
) -> None:
    if bbox_xywh is None:
        return
    x, y, box_width, box_height = np.round(bbox_xywh).astype(int)
    if box_width <= 0 or box_height <= 0:
        return
    cv2.rectangle(
        canvas,
        (x, y),
        (x + box_width - 1, y + box_height - 1),
        color,
        width,
    )


def save_tracking_overlay(
    frame: FrameData,
    result: TrackingFrameResult,
    path: Path,
    *,
    original_gigapose: Mapping[
        int, tuple[PoseHypothesis, np.ndarray]
    ] | None = None,
) -> None:
    canvas = np.asarray(frame.image, dtype=np.uint8).copy()
    original_gigapose = original_gigapose or {}
    for instance in result.instances:
        track_id = instance.track.track_id
        detection_id = int(instance.detection.detection_id)
        original = original_gigapose.get(detection_id)
        if original is not None:
            original_hypothesis, original_mask = original
            _draw_mask(canvas, original_mask, ORIGINAL_COLOR, alpha=0.22)
            original_bbox = _bbox_from_mask(original_mask)
            _draw_bbox(canvas, original_bbox, ORIGINAL_COLOR, width=2)
            if original_bbox is not None:
                original_x, original_y, _, original_height = original_bbox
                cv2.putText(
                    canvas,
                    f"GP raw top1 {original_hypothesis.measurement_score:.2f}",
                    (
                        max(0, int(original_x)),
                        min(
                            canvas.shape[0] - 5,
                            int(original_y + original_height + 16),
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    ORIGINAL_COLOR,
                    1,
                    cv2.LINE_AA,
                )

        rendered_mask = instance.chosen.rendered_mask
        if rendered_mask is not None:
            _draw_mask(canvas, rendered_mask, TRACKED_COLOR, alpha=0.38)
            _draw_bbox(
                canvas,
                _bbox_from_mask(rendered_mask),
                TRACKED_COLOR,
                width=2,
            )
        x, y, _, _ = np.round(instance.detection.bbox_xywh).astype(int)
        _draw_bbox(
            canvas,
            instance.detection.bbox_xywh,
            DETECTION_COLOR,
            width=2,
        )
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
            TRACKED_COLOR,
            1,
            cv2.LINE_AA,
        )

    legend = (
        "RED: original GigaPose CAD/bbox  "
        "GREEN: tracked CAD/bbox  "
        "YELLOW: observed detection bbox"
    )
    cv2.rectangle(canvas, (4, 4), (min(canvas.shape[1] - 1, 760), 27), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        legend,
        (10, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # frame.image is RGB while OpenCV output is BGR.
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
