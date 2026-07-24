"""Render-backed temporal 2D-to-3D LightGlue PnP recovery.

This module does not try to match real RGB directly to untextured CAD
geometry.  Instead, an accepted real frame becomes the visual template.  CAD
depth rendered at that pose converts its ALIKED keypoints into object-frame
3D points.  LightGlue then matches those real template keypoints to the next
real frame and PnP estimates a complete metric pose without observed depth.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tracking.geometry import (
    closest_rotation,
    rotation_error_deg,
    translation_error_m,
)
from tracking.lightglue_tracking.config import LightGlueTrackingConfig
from tracking.lightglue_tracking.matcher import (
    ALIKEDLightGlue,
    MatchResult,
    RegionFeatures,
    subset_region_features,
)
from tracking.rgb_self_recovery.inference import RecoveryProposal


@dataclass
class PnPTemplate:
    features: RegionFeatures
    object_points_m: np.ndarray
    source_pose: np.ndarray
    scene_id: int
    im_id: int


@dataclass
class PnPResult:
    proposal: RecoveryProposal
    match: MatchResult
    inlier_count: int
    inlier_ratio: float
    median_reprojection_px: float
    quality: float


def build_template(
    *,
    renderer,
    features: RegionFeatures,
    pose_m: np.ndarray,
    K: np.ndarray,
    scene_id: int,
    im_id: int,
    minimum_points: int,
) -> PnPTemplate | None:
    """Attach CAD object coordinates to real-image ALIKED descriptors."""

    x0, y0, x1, y1 = features.crop_bounds_xyxy
    width, height = x1 - x0, y1 - y0
    crop_homography = np.asarray(
        [[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    K_crop = crop_homography @ np.asarray(K, dtype=np.float64)
    _mask, depth = renderer.render(
        np.asarray(pose_m, dtype=float),
        K_crop,
        (height, width),
    )
    local = np.asarray(features.local_keypoints_xy, dtype=float)
    rounded = np.rint(local).astype(int)
    inside = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    depths = np.zeros(len(local), dtype=np.float64)
    valid_indices = np.where(inside)[0]
    depths[valid_indices] = depth[
        rounded[valid_indices, 1], rounded[valid_indices, 0]
    ]
    valid = inside & np.isfinite(depths) & (depths > 1e-5)
    selected = np.where(valid)[0]
    if selected.size < int(minimum_points):
        return None
    pixels_h = np.concatenate(
        [local[selected], np.ones((len(selected), 1))], axis=1
    )
    rays = np.linalg.solve(K_crop, pixels_h.T).T
    camera_points = rays * depths[selected, None]
    pose = np.asarray(pose_m, dtype=np.float64)
    object_points = (
        pose[:3, :3].T @ (camera_points - pose[:3, 3]).T
    ).T
    finite = np.isfinite(object_points).all(axis=1)
    selected = selected[finite]
    object_points = object_points[finite]
    if len(selected) < int(minimum_points):
        return None
    return PnPTemplate(
        features=subset_region_features(features, selected),
        object_points_m=object_points.astype(np.float32),
        source_pose=pose.copy(),
        scene_id=int(scene_id),
        im_id=int(im_id),
    )


def recover_pose(
    *,
    template: PnPTemplate,
    current_features: RegionFeatures,
    matcher: ALIKEDLightGlue,
    K: np.ndarray,
    previous_pose: np.ndarray,
    config: LightGlueTrackingConfig,
) -> PnPResult | None:
    """Match a real template to the current frame and solve metric PnP."""

    matched = matcher.match(template.features, current_features)
    if matched.count < config.pnp_min_points:
        return None
    object_points = template.object_points_m[matched.indices0]
    image_points = current_features.keypoints_xy[matched.indices1].astype(
        np.float32
    )
    success, rotation_vector, translation, inlier_indices = cv2.solvePnPRansac(
        object_points,
        image_points,
        np.asarray(K, dtype=np.float64),
        None,
        iterationsCount=int(config.pnp_iterations),
        reprojectionError=float(config.pnp_reprojection_px),
        confidence=0.995,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inlier_indices is None:
        return None
    inliers = inlier_indices.reshape(-1).astype(np.int64)
    inlier_count = int(inliers.size)
    ratio = inlier_count / max(matched.count, 1)
    if (
        inlier_count < config.pnp_min_inliers
        or ratio < config.pnp_min_inlier_ratio
    ):
        return None
    if hasattr(cv2, "solvePnPRefineLM") and inlier_count >= 6:
        rotation_vector, translation = cv2.solvePnPRefineLM(
            object_points[inliers],
            image_points[inliers],
            np.asarray(K, dtype=np.float64),
            None,
            rotation_vector,
            translation,
        )
    rotation, _ = cv2.Rodrigues(rotation_vector)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = closest_rotation(rotation)
    pose[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    if not np.isfinite(pose).all() or pose[2, 3] <= 1e-5:
        return None
    if (
        rotation_error_deg(pose, previous_pose)
        > config.pnp_max_rotation_step_deg
        or translation_error_m(pose, previous_pose)
        > config.pnp_max_translation_step_m
    ):
        return None
    projected, _ = cv2.projectPoints(
        object_points[inliers],
        rotation_vector,
        translation,
        np.asarray(K, dtype=np.float64),
        None,
    )
    errors = np.linalg.norm(
        projected.reshape(-1, 2) - image_points[inliers], axis=1
    )
    median_error = float(np.median(errors))
    mean_score = float(np.mean(matched.scores[inliers]))
    count_quality = min(inlier_count / 30.0, 1.0)
    reprojection_quality = float(
        np.exp(-median_error / max(config.pnp_reprojection_px, 1e-6))
    )
    quality = float(
        np.clip(
            0.35 * ratio
            + 0.25 * count_quality
            + 0.20 * mean_score
            + 0.20 * reprojection_quality,
            0.0,
            1.0,
        )
    )
    proposal = RecoveryProposal(
        pose=pose,
        source="lightglue_temporal_pnp",
        measurement_score=quality,
        prior_error=0.0,
    )
    return PnPResult(
        proposal=proposal,
        match=matched,
        inlier_count=inlier_count,
        inlier_ratio=float(ratio),
        median_reprojection_px=median_error,
        quality=quality,
    )
