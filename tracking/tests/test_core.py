from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tracking.association import associate_tracks
from tracking.config import TrackerConfig
from tracking.flow import SparseFlowPoseUpdater
from tracking.geometry import (
    pose_from_rt,
    rotation_error_deg,
    so3_exp,
    so3_log,
)
from tracking.io import (
    PredictionCSVProvider,
    decode_uncompressed_rle,
    merge_prediction_group_sets,
)
from tracking.refinement import CandidateRefiner
from tracking.scoring import CandidateScorer
from tracking.tracker import AdaptivePoseTracker
from tracking.types import Detection, FrameData, PoseHypothesis, Track, TrackMode


class SquareRenderer:
    def render(self, pose_m, K, image_shape):
        height, width = image_shape
        output = np.zeros((height, width), dtype=bool)
        translation = pose_m[:3, 3]
        if translation[2] <= 0:
            return output, np.zeros_like(output, dtype=np.float32)
        center = K @ translation
        center = center[:2] / center[2]
        radius = max(2, int(round(8.0 * K[0, 0] / 100.0 / translation[2])))
        x, y = np.round(center).astype(int)
        output[max(0, y - radius) : min(height, y + radius + 1),
               max(0, x - radius) : min(width, x + radius + 1)] = True
        depth = np.zeros_like(output, dtype=np.float32)
        depth[output] = translation[2]
        return output, depth


def square_detection(center=(50, 50), radius=8):
    mask = np.zeros((100, 100), dtype=bool)
    x, y = center
    mask[y - radius : y + radius + 1, x - radius : x + radius + 1] = True
    return Detection(
        0,
        np.asarray([x - radius, y - radius, 2 * radius + 1, 2 * radius + 1]),
        mask,
    )


class GeometryTests(unittest.TestCase):
    def test_so3_roundtrip_and_atan2_rotation_error(self):
        vector = np.asarray([0.3, -0.2, 0.1])
        reconstructed = so3_exp(so3_log(so3_exp(vector)))
        self.assertTrue(np.allclose(reconstructed, so3_exp(vector), atol=1e-7))
        first = pose_from_rt(np.eye(3), [0, 0, 1])
        second = pose_from_rt(so3_exp([0, 0, np.pi]), [0, 0, 1])
        self.assertAlmostEqual(rotation_error_deg(first, second), 180.0, places=5)

    def test_uncompressed_coco_rle(self):
        # Fortran-flattened [0, 1; 0, 1].
        decoded = decode_uncompressed_rle({"size": [2, 2], "counts": [2, 2]})
        expected = np.asarray([[False, True], [False, True]])
        self.assertTrue(np.array_equal(decoded, expected))


class PredictionIOTests(unittest.TestCase):
    def test_multi_hypothesis_grouping_and_mm_conversion(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "scene_id", "im_id", "obj_id", "score",
                        "R", "t", "time", "instance_id",
                    ],
                )
                writer.writeheader()
                for instance, score in ((7, 0.4), (7, 0.8), (8, 0.5)):
                    writer.writerow(
                        {
                            "scene_id": 1,
                            "im_id": 2,
                            "obj_id": 1,
                            "score": score,
                            "R": "1 0 0 0 1 0 0 0 1",
                            "t": "1000 0 2000",
                            "time": 0,
                            "instance_id": instance,
                        }
                    )
            groups = PredictionCSVProvider(path).groups_for_frame(1, 2)
            self.assertEqual([len(group) for group in groups], [2, 1])
            self.assertAlmostEqual(groups[0][0].measurement_score, 0.8)
            self.assertTrue(np.allclose(groups[0][0].pose[:3, 3], [1, 0, 2]))

    def test_auxiliary_candidates_merge_by_instance(self):
        pose = pose_from_rt(np.eye(3), [0, 0, 1])
        first = PoseHypothesis(
            pose, "gigapose", 0.7, prediction_instance_id=4
        )
        second = PoseHypothesis(
            pose, "epnp", 0.9, prediction_instance_id=4
        )
        merged = merge_prediction_group_sets([[[first]], [[second]]])
        self.assertEqual(len(merged), 1)
        self.assertEqual([item.source for item in merged[0]], ["epnp", "gigapose"])


class FlowAndAssociationTests(unittest.TestCase):
    def test_sparse_flow_recovers_translation(self):
        previous = np.zeros((100, 100), dtype=np.uint8)
        for y in range(25, 76, 10):
            for x in range(25, 76, 10):
                cv2.circle(previous, (x, y), 2, 255, -1)
        affine = np.float32([[1, 0, 4], [0, 1, -3]])
        current = cv2.warpAffine(previous, affine, (100, 100))
        mask = np.zeros_like(previous, dtype=bool)
        mask[15:85, 15:85] = True
        config = TrackerConfig().flow
        measurement = SparseFlowPoseUpdater(config).estimate(previous, current, mask)
        self.assertIsNotNone(measurement)
        self.assertAlmostEqual(measurement.affine[0, 2], 4.0, delta=0.5)
        self.assertAlmostEqual(measurement.affine[1, 2], -3.0, delta=0.5)

    def test_track_detection_association(self):
        pose = pose_from_rt(np.eye(3), [0, 0, 1])
        tracks = [
            Track(3, 1, pose, None, np.asarray([10, 10, 20, 20]), 1, TrackMode.NORMAL, "", 1, 0)
        ]
        detections = [Detection(5, np.asarray([11, 10, 20, 20]), np.ones((4, 4), bool))]
        matches, unmatched_tracks, unmatched_detections = associate_tracks(
            tracks, detections, (100, 100), TrackerConfig().association
        )
        self.assertEqual(matches, [(0, 0)])
        self.assertFalse(unmatched_tracks)
        self.assertFalse(unmatched_detections)


class ScoringAndTrackerTests(unittest.TestCase):
    def setUp(self):
        self.K = np.asarray([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=float)
        self.frame = FrameData(
            1,
            0,
            np.zeros((100, 100, 3), dtype=np.uint8),
            self.K,
            [square_detection()],
        )

    def test_cad_evidence_prefers_aligned_pose(self):
        config = TrackerConfig()
        scorer = CandidateScorer(SquareRenderer(), config)
        good = PoseHypothesis(pose_from_rt(np.eye(3), [0, 0, 1]), "good", 0.5)
        bad = PoseHypothesis(pose_from_rt(np.eye(3), [0.3, 0, 1]), "bad", 0.5)
        good_score = scorer.evaluate(good, self.frame, self.frame.detections[0]).score
        bad_score = scorer.evaluate(bad, self.frame, self.frame.detections[0]).score
        self.assertLess(good_score.total_error, bad_score.total_error)
        self.assertGreater(good_score.confidence, bad_score.confidence)

    def test_tracker_propagates_same_identity_without_fresh_pose(self):
        config = TrackerConfig()
        config.refinement.normal_iterations = 0
        config.refinement.uncertain_iterations = 0
        config.refinement.lost_iterations = 0
        config.hypotheses.add_flip_when_uncertain = False
        config.hypotheses.center_offsets_px = ()
        config.hypotheses.log_depth_offsets = ()
        config.hypotheses.translation_offsets_m = ()
        config.hypotheses.yaw_offsets_deg = ()
        scorer = CandidateScorer(SquareRenderer(), config)
        tracker = AdaptivePoseTracker(config, CandidateRefiner(scorer, config))
        fresh = [[PoseHypothesis(pose_from_rt(np.eye(3), [0, 0, 1]), "gigapose", 0.9)]]
        first = tracker.process_frame(self.frame, fresh)
        self.assertEqual(len(first.instances), 1)
        second_frame = FrameData(
            1, 1, self.frame.image.copy(), self.K, [square_detection()]
        )
        second = tracker.process_frame(second_frame, [])
        self.assertEqual(len(second.instances), 1)
        self.assertEqual(first.instances[0].track.track_id, second.instances[0].track.track_id)


if __name__ == "__main__":
    unittest.main()
