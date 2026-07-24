from __future__ import annotations

import numpy as np
import torch
import unittest

from tracking.config import AssociationConfig
from tracking.lightglue_tracking.association import associate_tracks
from tracking.lightglue_tracking.config import LightGlueTrackingConfig
from tracking.lightglue_tracking.matcher import (
    MatchResult,
    RegionFeatures,
    SimilarityMeasurement,
    estimate_similarity,
)
from tracking.lightglue_tracking.pnp import PnPTemplate, recover_pose
from tracking.types import Detection, Track, TrackMode


def _features(points: np.ndarray, identity: int = 0) -> RegionFeatures:
    points = np.asarray(points, dtype=np.float32)
    return RegionFeatures(
        data={
            "keypoints": torch.from_numpy(points)[None],
            "descriptors": torch.zeros(1, len(points), 128),
            "image_size": torch.tensor([[100.0, 100.0]]),
            "identity": identity,
        },
        keypoints_xy=points.copy(),
        local_keypoints_xy=points.copy(),
        crop_bounds_xyxy=(0, 0, 100, 100),
    )


def _measurement(quality: float) -> SimilarityMeasurement:
    return SimilarityMeasurement(
        affine=np.eye(3),
        point_count=20,
        inlier_count=20,
        inlier_ratio=1.0,
        median_reprojection_px=0.0,
        mean_match_score=1.0,
        quality=quality,
    )


class IdentityMatcher:
    def match(self, first: RegionFeatures, second: RegionFeatures) -> MatchResult:
        same = first.data["identity"] == second.data["identity"]
        points0 = first.keypoints_xy[:4]
        points1 = second.keypoints_xy[:4]
        return MatchResult(
            indices0=np.arange(4),
            indices1=np.arange(4),
            points0_xy=points0,
            points1_xy=points1,
            scores=np.ones(4, dtype=np.float32),
            similarity=_measurement(1.0 if same else 0.0),
            matching_time_s=0.0,
        )


class FixedMatcher:
    def match(self, first: RegionFeatures, second: RegionFeatures) -> MatchResult:
        count = min(first.count, second.count)
        indices = np.arange(count, dtype=np.int64)
        return MatchResult(
            indices0=indices,
            indices1=indices,
            points0_xy=first.keypoints_xy[:count],
            points1_xy=second.keypoints_xy[:count],
            scores=np.ones(count, dtype=np.float32),
            similarity=None,
            matching_time_s=0.0,
        )


def _track(track_id: int) -> Track:
    pose = np.eye(4)
    pose[2, 3] = 5.0
    return Track(
        track_id=track_id,
        obj_id=1,
        pose=pose,
        previous_pose=None,
        bbox_xywh=np.asarray([25.0, 25.0, 50.0, 50.0]),
        confidence=0.9,
        mode=TrackMode.NORMAL,
        source="test",
        scene_id=1,
        im_id=1,
    )


def _detection(detection_id: int) -> Detection:
    return Detection(
        detection_id=detection_id,
        bbox_xywh=np.asarray([25.0, 25.0, 50.0, 50.0]),
        mask=np.ones((100, 100), dtype=bool),
        obj_id=1,
    )


def test_similarity_ransac_recovers_scale_rotation_and_translation():
    rng = np.random.default_rng(7)
    source = rng.uniform(10.0, 90.0, size=(40, 2))
    angle = np.deg2rad(13.0)
    scale = 1.17
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    target = scale * (source @ rotation.T) + np.asarray([8.0, -4.0])
    target += rng.normal(0.0, 0.15, size=target.shape)
    target[:5] = rng.uniform(0.0, 100.0, size=(5, 2))
    config = LightGlueTrackingConfig(
        min_matches=8,
        min_inliers=10,
        min_inlier_ratio=0.5,
        ransac_threshold_px=2.0,
        max_median_reprojection_px=1.0,
    )
    result = estimate_similarity(source, target, np.ones(40), config)
    assert result is not None
    assert abs(result.scale - scale) < 0.02
    assert abs(result.angle_rad - angle) < np.deg2rad(1.0)
    assert result.inlier_count >= 34
    assert result.median_reprojection_px < 0.5


def test_lightglue_quality_resolves_geometrically_ambiguous_assignment():
    points = np.asarray([[20, 20], [40, 20], [20, 40], [40, 40]])
    tracks = [_track(0), _track(1)]
    detections = [_detection(0), _detection(1)]
    config = LightGlueTrackingConfig(
        association_policy="always",
        association_weight=2.0,
    )
    result = associate_tracks(
        tracks,
        detections,
        (100, 100),
        AssociationConfig(
            identity_enabled=True,
            appearance_weight=0.0,
            mask_iou_weight=0.0,
        ),
        config,
        IdentityMatcher(),
        {0: _features(points, 0), 1: _features(points, 1)},
        [_features(points, 1), _features(points, 0)],
        np.zeros((100, 100, 3), dtype=np.uint8),
    )
    assert sorted(result.matches) == [(0, 1), (1, 0)]
    assert result.attempted_lightglue_pairs == 4


def test_temporal_pnp_recovers_full_metric_pose_without_observed_depth():
    object_points = np.asarray(
        [
            [-1.0, -0.5, -0.3],
            [1.0, -0.5, -0.3],
            [-1.0, 0.5, -0.3],
            [1.0, 0.5, -0.3],
            [-1.0, -0.5, 0.3],
            [1.0, -0.5, 0.3],
            [-1.0, 0.5, 0.3],
            [1.0, 0.5, 0.3],
            [0.0, 0.0, 0.7],
            [0.3, -0.2, -0.7],
        ],
        dtype=np.float32,
    )
    rotation_vector = np.asarray([0.08, -0.12, 0.18], dtype=np.float64)
    import cv2

    rotation, _ = cv2.Rodrigues(rotation_vector)
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = [0.2, -0.1, 8.0]
    K = np.asarray(
        [[800.0, 0.0, 320.0], [0.0, 810.0, 180.0], [0.0, 0.0, 1.0]]
    )
    image_points, _ = cv2.projectPoints(
        object_points,
        rotation_vector,
        pose[:3, 3],
        K,
        None,
    )
    image_points = image_points.reshape(-1, 2).astype(np.float32)
    template_features = _features(
        np.column_stack(
            [
                np.linspace(50, 150, len(object_points)),
                np.linspace(40, 120, len(object_points)),
            ]
        )
    )
    current_features = _features(image_points)
    template = PnPTemplate(
        features=template_features,
        object_points_m=object_points,
        source_pose=pose.copy(),
        scene_id=1,
        im_id=1,
    )
    config = LightGlueTrackingConfig(
        pnp_recovery_enabled=True,
        pnp_min_points=8,
        pnp_min_inliers=8,
        pnp_min_inlier_ratio=0.5,
        pnp_max_rotation_step_deg=90.0,
        pnp_max_translation_step_m=10.0,
    )
    result = recover_pose(
        template=template,
        current_features=current_features,
        matcher=FixedMatcher(),
        K=K,
        previous_pose=pose,
        config=config,
    )
    assert result is not None
    np.testing.assert_allclose(
        result.proposal.pose[:3, 3], pose[:3, 3], atol=1e-3
    )
    np.testing.assert_allclose(
        result.proposal.pose[:3, :3], pose[:3, :3], atol=1e-3
    )
    assert result.inlier_count == len(object_points)


class LightGlueCoreTests(unittest.TestCase):
    """Make the dependency-free checks runnable without pytest."""

    def test_similarity_geometry(self):
        test_similarity_ransac_recovers_scale_rotation_and_translation()

    def test_association_quality(self):
        test_lightglue_quality_resolves_geometrically_ambiguous_assignment()

    def test_metric_pnp(self):
        test_temporal_pnp_recovers_full_metric_pose_without_observed_depth()
