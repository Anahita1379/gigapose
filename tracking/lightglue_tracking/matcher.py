"""Frozen ALIKED + LightGlue extraction and geometric match verification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch

from tracking.lightglue_tracking.config import LightGlueTrackingConfig


@dataclass
class RegionFeatures:
    """LightGlue feature dictionary plus original-image coordinates."""

    data: dict[str, torch.Tensor]
    keypoints_xy: np.ndarray
    local_keypoints_xy: np.ndarray
    crop_bounds_xyxy: tuple[int, int, int, int]
    extraction_time_s: float = 0.0

    @property
    def count(self) -> int:
        return int(self.keypoints_xy.shape[0])


@dataclass
class SimilarityMeasurement:
    affine: np.ndarray
    point_count: int
    inlier_count: int
    inlier_ratio: float
    median_reprojection_px: float
    mean_match_score: float
    quality: float

    @property
    def scale(self) -> float:
        return float(np.linalg.norm(self.affine[:2, 0]))

    @property
    def angle_rad(self) -> float:
        return float(np.arctan2(self.affine[1, 0], self.affine[0, 0]))


@dataclass
class MatchResult:
    indices0: np.ndarray
    indices1: np.ndarray
    points0_xy: np.ndarray
    points1_xy: np.ndarray
    scores: np.ndarray
    similarity: SimilarityMeasurement | None
    matching_time_s: float

    @property
    def count(self) -> int:
        return int(self.indices0.size)


def estimate_similarity(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    scores: np.ndarray,
    config: LightGlueTrackingConfig,
) -> SimilarityMeasurement | None:
    """Fit and validate a partial affine (2D similarity) with RANSAC."""

    source = np.asarray(source_xy, dtype=np.float32).reshape(-1, 2)
    target = np.asarray(target_xy, dtype=np.float32).reshape(-1, 2)
    match_scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if (
        len(source) < config.min_matches
        or source.shape != target.shape
        or match_scores.size != len(source)
    ):
        return None
    affine_2x3, inlier_mask = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(config.ransac_threshold_px),
        maxIters=3000,
        confidence=0.995,
        refineIters=10,
    )
    if affine_2x3 is None or inlier_mask is None:
        return None
    affine = np.vstack([affine_2x3, [0.0, 0.0, 1.0]]).astype(np.float64)
    if not np.isfinite(affine).all():
        return None
    inliers = inlier_mask.reshape(-1).astype(bool)
    inlier_count = int(inliers.sum())
    ratio = inlier_count / max(len(source), 1)
    scale = float(np.linalg.norm(affine[:2, 0]))
    projected_h = (
        affine
        @ np.concatenate(
            [source.astype(np.float64), np.ones((len(source), 1))],
            axis=1,
        ).T
    ).T
    errors = np.linalg.norm(projected_h[:, :2] - target, axis=1)
    median_error = (
        float(np.median(errors[inliers])) if inlier_count else float("inf")
    )
    mean_score = (
        float(np.mean(match_scores[inliers])) if inlier_count else 0.0
    )
    if (
        inlier_count < config.min_inliers
        or ratio < config.min_inlier_ratio
        or median_error > config.max_median_reprojection_px
        or not config.minimum_scale <= scale <= config.maximum_scale
    ):
        return None
    count_quality = min(inlier_count / 30.0, 1.0)
    reprojection_quality = float(
        np.exp(
            -median_error
            / max(float(config.max_median_reprojection_px), 1e-6)
        )
    )
    quality = float(
        np.clip(
            0.30 * ratio
            + 0.25 * count_quality
            + 0.25 * mean_score
            + 0.20 * reprojection_quality,
            0.0,
            1.0,
        )
    )
    return SimilarityMeasurement(
        affine=affine,
        point_count=len(source),
        inlier_count=inlier_count,
        inlier_ratio=float(ratio),
        median_reprojection_px=median_error,
        mean_match_score=mean_score,
        quality=quality,
    )


def subset_region_features(
    features: RegionFeatures, indices: np.ndarray
) -> RegionFeatures:
    """Keep a descriptor-aligned subset of region features."""

    selected = np.asarray(indices, dtype=np.int64).reshape(-1)
    tensor_indices = torch.as_tensor(
        selected, dtype=torch.long, device=features.data["keypoints"].device
    )
    data: dict[str, torch.Tensor] = {}
    feature_count = features.count
    for key, value in features.data.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim >= 2
            and value.shape[0] == 1
            and value.shape[1] == feature_count
        ):
            data[key] = value.index_select(1, tensor_indices)
        else:
            data[key] = value
    return RegionFeatures(
        data=data,
        keypoints_xy=features.keypoints_xy[selected].copy(),
        local_keypoints_xy=features.local_keypoints_xy[selected].copy(),
        crop_bounds_xyxy=features.crop_bounds_xyxy,
        extraction_time_s=features.extraction_time_s,
    )


class ALIKEDLightGlue:
    """One frozen ALIKED extractor and LightGlue matcher shared by a run."""

    def __init__(
        self,
        config: LightGlueTrackingConfig,
        *,
        device: str = "cuda",
    ):
        config.validate()
        self.config = config
        self.device = torch.device(device)
        root = Path(config.lightglue_root).expanduser().resolve()
        if not (root / "lightglue" / "__init__.py").is_file():
            raise FileNotFoundError(
                f"LightGlue checkout not found at {root}. "
                "Pass --lightglue-root."
            )
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        try:
            from lightglue import ALIKED, LightGlue
        except ImportError as exc:
            raise ImportError(
                "Could not import the local LightGlue checkout. Install its "
                "dependencies in the GigaPose environment."
            ) from exc
        self.extractor = ALIKED(
            max_num_keypoints=int(config.max_num_keypoints),
            detection_threshold=-1.0,
        ).eval().to(self.device)
        self.matcher = LightGlue(
            features="aliked",
            filter_threshold=float(config.filter_threshold),
            depth_confidence=float(config.depth_confidence),
            width_confidence=float(config.width_confidence),
            mp=bool(config.mixed_precision),
        ).eval().to(self.device)
        for module in (self.extractor, self.matcher):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.extractions = 0
        self.matches = 0
        self.extraction_time_s = 0.0
        self.matching_time_s = 0.0

    @staticmethod
    def _crop_bounds(
        bbox_xywh: np.ndarray,
        image_shape: tuple[int, int],
        padding_frac: float,
    ) -> tuple[int, int, int, int]:
        height, width = image_shape
        x, y, box_width, box_height = np.asarray(
            bbox_xywh, dtype=float
        ).reshape(4)
        padding = float(padding_frac) * max(box_width, box_height, 1.0)
        x0 = int(np.clip(np.floor(x - padding), 0, width - 1))
        y0 = int(np.clip(np.floor(y - padding), 0, height - 1))
        x1 = int(np.clip(np.ceil(x + box_width + padding), x0 + 1, width))
        y1 = int(np.clip(np.ceil(y + box_height + padding), y0 + 1, height))
        return x0, y0, x1, y1

    def extract_region(
        self,
        image_rgb: np.ndarray,
        bbox_xywh: np.ndarray,
        mask: np.ndarray | None,
    ) -> RegionFeatures | None:
        image = np.asarray(image_rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("ALIKED expects an RGB image")
        x0, y0, x1, y1 = self._crop_bounds(
            bbox_xywh,
            image.shape[:2],
            self.config.crop_padding_frac,
        )
        crop = image[y0:y1, x0:x1, :3].copy()
        crop_mask: np.ndarray | None = None
        if mask is not None:
            full_mask = np.asarray(mask, dtype=bool)
            if full_mask.shape == image.shape[:2]:
                crop_mask = full_mask[y0:y1, x0:x1]
                crop[~crop_mask] = 0
        tensor = (
            torch.from_numpy(np.moveaxis(crop, -1, 0).copy())
            .to(self.device, dtype=torch.float32)
            / 255.0
        )
        started = time.perf_counter()
        with torch.inference_mode():
            data = self.extractor.extract(
                tensor,
                resize=int(self.config.resize),
            )
        elapsed = time.perf_counter() - started
        local = data["keypoints"][0].detach().cpu().numpy()
        keep = np.ones(len(local), dtype=bool)
        if crop_mask is not None and len(local):
            rounded = np.rint(local).astype(int)
            valid = (
                (rounded[:, 0] >= 0)
                & (rounded[:, 0] < crop_mask.shape[1])
                & (rounded[:, 1] >= 0)
                & (rounded[:, 1] < crop_mask.shape[0])
            )
            keep &= valid
            valid_indices = np.where(valid)[0]
            keep[valid_indices] &= crop_mask[
                rounded[valid_indices, 1],
                rounded[valid_indices, 0],
            ]
        selected = np.where(keep)[0]
        if selected.size < 4:
            return None
        tensor_indices = torch.as_tensor(
            selected, dtype=torch.long, device=self.device
        )
        for key in ("keypoints", "descriptors", "keypoint_scores"):
            if key in data:
                data[key] = data[key].index_select(1, tensor_indices)
        local = local[selected]
        global_xy = local + np.asarray([x0, y0], dtype=np.float32)
        data["keypoints"] = torch.from_numpy(global_xy).to(
            self.device, dtype=torch.float32
        )[None]
        data["image_size"] = torch.tensor(
            [[image.shape[1], image.shape[0]]],
            dtype=torch.float32,
            device=self.device,
        )
        result = RegionFeatures(
            data=data,
            keypoints_xy=global_xy,
            local_keypoints_xy=local,
            crop_bounds_xyxy=(x0, y0, x1, y1),
            extraction_time_s=elapsed,
        )
        self.extractions += 1
        self.extraction_time_s += elapsed
        return result

    def match(
        self,
        first: RegionFeatures,
        second: RegionFeatures,
    ) -> MatchResult:
        started = time.perf_counter()
        with torch.inference_mode():
            output: dict[str, Any] = self.matcher(
                {"image0": first.data, "image1": second.data}
            )
        elapsed = time.perf_counter() - started
        raw_matches = output["matches"][0]
        raw_scores = output["scores"][0]
        indices = raw_matches.detach().cpu().numpy().astype(np.int64)
        scores = raw_scores.detach().cpu().numpy().astype(np.float32)
        if indices.size == 0:
            indices = np.empty((0, 2), dtype=np.int64)
        indices0, indices1 = indices[:, 0], indices[:, 1]
        points0 = first.keypoints_xy[indices0]
        points1 = second.keypoints_xy[indices1]
        similarity = estimate_similarity(
            points0,
            points1,
            scores,
            self.config,
        )
        self.matches += 1
        self.matching_time_s += elapsed
        return MatchResult(
            indices0=indices0,
            indices1=indices1,
            points0_xy=points0,
            points1_xy=points1,
            scores=scores,
            similarity=similarity,
            matching_time_s=elapsed,
        )

    def runtime_report(self) -> dict[str, float | int]:
        return {
            "feature_extractions": self.extractions,
            "feature_matches": self.matches,
            "feature_extraction_time_s": self.extraction_time_s,
            "feature_matching_time_s": self.matching_time_s,
        }
