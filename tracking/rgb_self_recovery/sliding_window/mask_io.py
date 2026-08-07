"""Optional direct-mask loading and rendered-mask fallback for isolated runners."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _bbox(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if not len(xs):
        return np.zeros(4, dtype=float)
    return np.asarray(
        [xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1],
        dtype=float,
    )


class MaskResolver:
    """Override dataset masks from a directory, with explicit fallback policy.

    A pattern containing ``detection_id`` or ``external_id`` is interpreted as
    one binary file per instance. Otherwise one frame-level integer mask is
    loaded and split using the detection's external ID (or detection index+1).
    """

    def __init__(
        self,
        mask_dir: Path | None,
        pattern: str,
        fallback: str,
    ):
        self.mask_dir = None if mask_dir is None else Path(mask_dir)
        self.pattern = str(pattern)
        self.fallback = str(fallback)
        if self.fallback not in {"dataset", "bbox", "render"}:
            raise ValueError("mask fallback must be dataset, bbox, or render")
        if self.mask_dir is not None and not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory does not exist: {self.mask_dir}")

    def _path(self, frame, detection=None) -> Path | None:
        if self.mask_dir is None:
            return None
        values = {
            "scene_id": int(frame.scene_id),
            "im_id": int(frame.im_id),
            "detection_id": 0 if detection is None else int(detection.detection_id),
            "external_id": 0
            if detection is None or detection.external_id is None
            else int(detection.external_id),
        }
        return self.mask_dir / self.pattern.format(**values)

    @staticmethod
    def _read(path: Path, shape: tuple[int, int]) -> np.ndarray:
        value = np.asarray(Image.open(path))
        if value.ndim == 3:
            if value.shape[2] >= 3:
                value = (
                    value[..., 0].astype(np.int64)
                    + (value[..., 1].astype(np.int64) << 8)
                    + (value[..., 2].astype(np.int64) << 16)
                )
            else:
                value = value[..., 0]
        if value.shape != shape:
            value = cv2.resize(
                value.astype(np.int32),
                (shape[1], shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return value

    def apply_directory(self, frame) -> dict[int, str]:
        statuses: dict[int, str] = {}
        instance_pattern = (
            "{detection_id" in self.pattern or "{external_id" in self.pattern
        )
        shared = None
        shared_path = self._path(frame)
        if (
            self.mask_dir is not None
            and not instance_pattern
            and shared_path is not None
            and shared_path.is_file()
        ):
            shared = self._read(shared_path, frame.image.shape[:2])
        for detection in frame.detections:
            mask = None
            path = self._path(frame, detection)
            if instance_pattern and path is not None and path.is_file():
                mask = self._read(path, frame.image.shape[:2]) > 0
            elif shared is not None:
                unique = np.unique(shared)
                if len(frame.detections) == 1 and len(unique) <= 2:
                    mask = shared > 0
                else:
                    value = (
                        detection.external_id
                        if detection.external_id is not None
                        else detection.detection_id + 1
                    )
                    mask = shared == int(value)
            if mask is not None and int(mask.sum()) > 0:
                detection.mask = np.asarray(mask, dtype=bool)
                detection.bbox_xywh = _bbox(mask)
                statuses[detection.detection_id] = "mask_directory"
            else:
                statuses[detection.detection_id] = "missing_override"
                if self.fallback == "bbox":
                    canvas = np.zeros(frame.image.shape[:2], dtype=bool)
                    x, y, w, h = np.round(detection.bbox_xywh).astype(int)
                    x0, y0 = max(x, 0), max(y, 0)
                    x1 = min(x + w, canvas.shape[1])
                    y1 = min(y + h, canvas.shape[0])
                    canvas[y0:y1, x0:x1] = True
                    detection.mask = canvas
                    statuses[detection.detection_id] = "bbox_fallback"
                elif self.fallback == "dataset":
                    statuses[detection.detection_id] = "dataset_fallback"
        return statuses

    def maybe_render_fallback(
        self, frame, detection, fresh_hypotheses, renderer, statuses
    ) -> str:
        status = statuses.get(detection.detection_id, "dataset_fallback")
        if self.fallback != "render" or status == "mask_directory":
            return status
        if fresh_hypotheses:
            rendered, _ = renderer.render(
                fresh_hypotheses[0].pose, frame.K, frame.image.shape[:2]
            )
            if int(rendered.sum()) > 0:
                detection.mask = np.asarray(rendered, dtype=bool)
                detection.bbox_xywh = _bbox(rendered)
                return "rank0_render_fallback"
        return "dataset_fallback_no_render"
