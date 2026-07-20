"""CAD alignment evidence and confidence scoring for one pose hypothesis."""

from __future__ import annotations

import math
from typing import Protocol, Sequence

import cv2
import numpy as np

from tracking.association import bbox_iou
from tracking.config import TrackerConfig
from tracking.geometry import rotation_error_deg, translation_error_m
from tracking.types import (
    Detection,
    EvaluatedHypothesis,
    FrameData,
    PoseHypothesis,
    ScoreBreakdown,
)


class RenderProtocol(Protocol):
    def render(
        self,
        pose_m: np.ndarray,
        K: np.ndarray,
        image_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]: ...


def bbox_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return np.zeros(4, dtype=float)
    return np.asarray(
        [xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1],
        dtype=float,
    )


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.logical_and(first, second).sum())
    union = int(np.logical_or(first, second).sum())
    return intersection / union if union else 0.0


def _boundary(mask: np.ndarray) -> np.ndarray:
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    eroded = cv2.erode(mask_u8, np.ones((3, 3), np.uint8))
    return np.logical_and(mask_u8 > 0, eroded == 0)


def symmetric_edge_error(
    first: np.ndarray, second: np.ndarray, truncation_px: float
) -> float:
    support = np.logical_or(first, second)
    ys, xs = np.where(support)
    if xs.size == 0:
        return 1.0
    padding = max(2, int(math.ceil(truncation_px)))
    x0, x1 = max(0, int(xs.min()) - padding), min(
        first.shape[1], int(xs.max()) + padding + 1
    )
    y0, y1 = max(0, int(ys.min()) - padding), min(
        first.shape[0], int(ys.max()) + padding + 1
    )
    first = first[y0:y1, x0:x1]
    second = second[y0:y1, x0:x1]
    first_edge, second_edge = _boundary(first), _boundary(second)
    if not first_edge.any() or not second_edge.any():
        return 1.0
    first_distance = cv2.distanceTransform(
        (~first_edge).astype(np.uint8), cv2.DIST_L2, 3
    )
    second_distance = cv2.distanceTransform(
        (~second_edge).astype(np.uint8), cv2.DIST_L2, 3
    )
    distance = 0.5 * (
        float(second_distance[first_edge].mean())
        + float(first_distance[second_edge].mean())
    )
    return float(np.clip(distance / max(truncation_px, 1e-6), 0.0, 1.0))


class CandidateScorer:
    """Combine independent image evidence with a deliberately weak motion prior."""

    def __init__(self, renderer: RenderProtocol, config: TrackerConfig):
        self.renderer = renderer
        self.config = config

    def _render_for_detection(
        self,
        hypothesis: PoseHypothesis,
        frame: FrameData,
        detection: Detection,
        occluder_poses: Sequence[np.ndarray] = (),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render a low-resolution padded ROI and restore full-image coordinates."""

        image_height, image_width = frame.image.shape[:2]
        x, y, width, height = np.asarray(detection.bbox_xywh, dtype=float)
        padding = self.config.render_crop_padding_frac * max(width, height, 1.0)
        crop_width = max(width + 2.0 * padding, self.config.render_crop_min_side_px)
        crop_height = max(height + 2.0 * padding, self.config.render_crop_min_side_px)
        center_x, center_y = x + 0.5 * width, y + 0.5 * height
        x0 = max(0, int(math.floor(center_x - 0.5 * crop_width)))
        y0 = max(0, int(math.floor(center_y - 0.5 * crop_height)))
        x1 = min(image_width, int(math.ceil(center_x + 0.5 * crop_width)))
        y1 = min(image_height, int(math.ceil(center_y + 0.5 * crop_height)))
        if x1 <= x0 or y1 <= y0:
            x0, y0, x1, y1 = 0, 0, image_width, image_height
        crop_height_int, crop_width_int = y1 - y0, x1 - x0
        scale = self.config.render_scale
        render_width = max(8, int(round(crop_width_int * scale)))
        render_height = max(8, int(round(crop_height_int * scale)))
        scale_x, scale_y = (
            render_width / crop_width_int,
            render_height / crop_height_int,
        )
        K_crop = np.asarray(frame.K, dtype=float).copy()
        K_crop[0, 2] -= x0
        K_crop[1, 2] -= y0
        K_render = np.diag([scale_x, scale_y, 1.0]) @ K_crop
        render_instances = getattr(self.renderer, "render_instances", None)
        if (
            self.config.occlusion.enabled
            and occluder_poses
            and callable(render_instances)
        ):
            segmentation_small, depth_small = render_instances(
                [hypothesis.pose, *occluder_poses],
                K_render,
                (render_height, render_width),
            )
            mask_small = np.asarray(segmentation_small) == 1
        else:
            mask_small, depth_small = self.renderer.render(
                hypothesis.pose, K_render, (render_height, render_width)
            )
        mask_crop = cv2.resize(
            np.asarray(mask_small, dtype=np.uint8),
            (crop_width_int, crop_height_int),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        depth_crop = cv2.resize(
            np.asarray(depth_small, dtype=np.float32),
            (crop_width_int, crop_height_int),
            interpolation=cv2.INTER_NEAREST,
        )
        mask = np.zeros((image_height, image_width), dtype=bool)
        depth = np.zeros((image_height, image_width), dtype=np.float32)
        mask[y0:y1, x0:x1] = mask_crop
        depth[y0:y1, x0:x1] = depth_crop
        return mask, depth

    def evaluate(
        self,
        hypothesis: PoseHypothesis,
        frame: FrameData,
        detection: Detection,
        motion_reference_pose: np.ndarray | None = None,
        *,
        keep_render: bool = False,
        occluder_poses: Sequence[np.ndarray] = (),
    ) -> EvaluatedHypothesis:
        mask, rendered_depth = self._render_for_detection(
            hypothesis, frame, detection, occluder_poses
        )
        rendered_pixels = int(mask.sum())
        observed_mask = np.asarray(detection.mask, dtype=bool)
        silhouette_iou = mask_iou(mask, observed_mask)
        silhouette_error = 1.0 - silhouette_iou
        rendered_bbox = bbox_from_mask(mask)
        bbox_overlap = bbox_iou(rendered_bbox, detection.bbox_xywh)
        bbox_error = 1.0 - bbox_overlap
        edge_error = symmetric_edge_error(
            mask, observed_mask, self.config.max_edge_distance_px
        )

        valid_depth = np.zeros_like(mask, dtype=bool)
        depth_error = 0.0
        log_depth_ratio = 0.0
        if frame.depth_m is not None:
            valid_depth = (
                mask
                & observed_mask
                & np.isfinite(frame.depth_m)
                & (frame.depth_m > 0)
                & np.isfinite(rendered_depth)
                & (rendered_depth > 0)
            )
            if valid_depth.any():
                residual = np.abs(
                    np.log(rendered_depth[valid_depth] + 1e-6)
                    - np.log(frame.depth_m[valid_depth] + 1e-6)
                )
                depth_error = float(
                    np.clip(
                        np.median(residual)
                        / max(math.log1p(self.config.depth_truncation_m), 1e-6),
                        0.0,
                        1.0,
                    )
                )
                log_depth_ratio = float(
                    np.log(
                        (np.median(rendered_depth[valid_depth]) + 1e-6)
                        / (np.median(frame.depth_m[valid_depth]) + 1e-6)
                    )
                )

        feature_error = 1.0 - float(
            np.clip(hypothesis.measurement_score, 0.0, 1.0)
        )
        motion_translation, motion_rotation, motion_error = 0.0, 0.0, 0.0
        if motion_reference_pose is not None:
            motion_translation = translation_error_m(
                hypothesis.pose, motion_reference_pose
            )
            motion_rotation = rotation_error_deg(hypothesis.pose, motion_reference_pose)
            motion_error = 0.5 * (
                min(
                    motion_translation
                    / max(self.config.motion_translation_scale_m, 1e-6),
                    1.0,
                )
                + min(
                    motion_rotation
                    / max(self.config.motion_rotation_scale_deg, 1e-6),
                    1.0,
                )
            )

        rendered_center = np.asarray(
            [
                rendered_bbox[0] + 0.5 * rendered_bbox[2],
                rendered_bbox[1] + 0.5 * rendered_bbox[3],
            ]
        )
        height, width = frame.image.shape[:2]
        center_delta = rendered_center - detection.center
        log_area_ratio = float(
            np.log(
                (max(float(rendered_pixels), 1.0))
                / max(float(observed_mask.sum()), 1.0)
            )
        )

        weights = self.config.score
        # Missing depth is missing evidence, not a perfect zero-error match.
        depth_weight = weights.depth if valid_depth.any() else 0.0
        evidence_error = (
            weights.feature * feature_error
            + weights.silhouette * silhouette_error
            + weights.edge * edge_error
            + depth_weight * depth_error
            + weights.bbox * bbox_error
        )
        evidence_weight = (
            weights.feature
            + weights.silhouette
            + weights.edge
            + depth_weight
            + weights.bbox
        )
        total_error = evidence_error + weights.motion * motion_error
        total_weight = evidence_weight + weights.motion
        normalized_error = total_error / max(total_weight, 1e-9)
        total_score = float(np.clip(1.0 - normalized_error, 0.0, 1.0))
        # Confidence excludes motion so a self-consistent but visually wrong
        # track cannot become permanently overconfident.
        confidence = float(
            np.clip(1.0 - evidence_error / max(evidence_weight, 1e-9), 0.0, 1.0)
        )
        if rendered_pixels < self.config.min_render_pixels:
            total_error += total_weight
            total_score = confidence = 0.0

        breakdown = ScoreBreakdown(
            total_error=float(total_error),
            total_score=total_score,
            confidence=confidence,
            feature_error=feature_error,
            silhouette_error=silhouette_error,
            silhouette_iou=silhouette_iou,
            edge_error=edge_error,
            depth_error=depth_error,
            bbox_error=bbox_error,
            bbox_iou=bbox_overlap,
            motion_error=float(motion_error),
            motion_translation_m=float(motion_translation),
            motion_rotation_deg=float(motion_rotation),
            center_dx_frac=float(center_delta[0] / max(width, 1)),
            center_dy_frac=float(center_delta[1] / max(height, 1)),
            log_area_ratio=log_area_ratio,
            log_depth_ratio=log_depth_ratio,
            rendered_pixels=rendered_pixels,
            valid_depth_pixels=int(valid_depth.sum()),
        )
        return EvaluatedHypothesis(
            hypothesis=hypothesis,
            score=breakdown,
            rendered_mask=mask if keep_render else None,
            rendered_depth_m=rendered_depth if keep_render else None,
        )
