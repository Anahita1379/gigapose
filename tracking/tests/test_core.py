from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch

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
    WebDatasetSequence,
    decode_uncompressed_rle,
    merge_prediction_group_sets,
)
from tracking.refinement import CandidateRefiner
from tracking.recovery import FEATURE_NAMES, RecoveryPredictor
from tracking.run_tracking import _apply_config_overrides
from tracking.scoring import CandidateScorer
from tracking.tracker import AdaptivePoseTracker
from tracking.train_recovery import _prepare_recovery_data
from tracking.train_recovery import main as train_recovery_main
from tracking.types import Detection, FrameData, PoseHypothesis, Track, TrackMode
from tracking.visualization import save_tracking_overlay


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

    def render_instances(self, poses_m, K, image_shape):
        height, width = image_shape
        segmentation = np.zeros((height, width), dtype=np.uint16)
        nearest_depth = np.zeros((height, width), dtype=np.float32)
        for instance_id, pose in enumerate(poses_m, start=1):
            mask, depth = self.render(pose, K, image_shape)
            visible = mask & (
                (nearest_depth <= 0)
                | ((depth > 0) & (depth < nearest_depth))
            )
            segmentation[visible] = instance_id
            nearest_depth[visible] = depth[visible]
        return segmentation, nearest_depth


class OcclusionRenderer:
    def render(self, pose_m, K, image_shape):
        height, width = image_shape
        mask = np.zeros((height, width), dtype=bool)
        mask[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = True
        depth = np.zeros((height, width), dtype=np.float32)
        depth[mask] = 2.0
        return mask, depth

    def render_instances(self, poses_m, K, image_shape):
        height, width = image_shape
        segmentation = np.zeros((height, width), dtype=np.uint16)
        y0, y1 = height // 4, 3 * height // 4
        x0, x1 = width // 4, 3 * width // 4
        middle = (x0 + x1) // 2
        segmentation[y0:y1, x0:middle] = 1
        segmentation[y0:y1, middle:x1] = 2
        depth = np.zeros((height, width), dtype=np.float32)
        depth[segmentation > 0] = 2.0
        return segmentation, depth


class FixedRecoveryModel(torch.nn.Module):
    def forward(self, features):
        batch_shape = features.shape[:-1]
        dtype, device = features.dtype, features.device
        rotation = torch.tensor(
            [0.0, 0.0, 0.2], dtype=dtype, device=device
        ).expand(*batch_shape, 3)
        translation = torch.tensor(
            [0.2, 0.0, 0.0], dtype=dtype, device=device
        ).expand(*batch_shape, 3)
        scalar = torch.zeros(batch_shape, dtype=dtype, device=device)
        return {
            "rotation_raw": rotation,
            "translation_raw": translation,
            "confidence_logit": scalar,
            "quality": scalar,
        }


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
    def test_webdataset_sequence_excludes_keys_removed_from_split(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset_dir = Path(folder)
            split_dir = dataset_dir / "test"
            split_dir.mkdir()
            (dataset_dir / "frame_map.json").write_text(
                json.dumps(
                    [
                        {"scene_id": 1, "im_id": 1},
                        {"scene_id": 1, "im_id": 2},
                    ]
                )
            )
            (split_dir / "key_to_shard.json").write_text(
                json.dumps({"000001_000001": 0})
            )

            sequence = WebDatasetSequence(dataset_dir, "test", load_depth=False)

            self.assertEqual(len(sequence), 1)
            self.assertEqual(int(sequence.rows[0]["im_id"]), 1)

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


class ConfigurationTests(unittest.TestCase):
    @staticmethod
    def override_args(**updates):
        values = {
            "same_frame_recovery": None,
            "identity_aware_association": None,
            "use_external_ids": None,
            "occlusion_aware_scoring": None,
            "global_safety_interval": None,
            "max_track_age_without_global": None,
        }
        values.update(updates)
        return SimpleNamespace(**values)

    def test_no_cli_overrides_preserve_legacy_defaults(self):
        config = _apply_config_overrides(
            TrackerConfig(), self.override_args()
        )
        self.assertFalse(config.same_frame_recovery.enabled)
        self.assertFalse(config.association.identity_enabled)
        self.assertFalse(config.association.use_external_id)
        self.assertFalse(config.occlusion.enabled)
        self.assertEqual(config.state.safety_interval, 15)
        self.assertEqual(config.state.max_track_age_without_global, 30)

    def test_external_ids_enable_identity_association(self):
        config = _apply_config_overrides(
            TrackerConfig(),
            self.override_args(use_external_ids=True),
        )
        self.assertTrue(config.association.identity_enabled)
        self.assertTrue(config.association.use_external_id)

    def test_conflicting_identity_cli_switches_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            _apply_config_overrides(
                TrackerConfig(),
                self.override_args(
                    identity_aware_association=False,
                    use_external_ids=True,
                ),
            )


class RecoveryDataTests(unittest.TestCase):
    @staticmethod
    def write_dataset(path, group_ids, feature_value):
        group_ids = np.asarray(group_ids, dtype=np.int64)
        sample_count = len(group_ids)
        np.savez_compressed(
            path,
            features=np.full(
                (sample_count, len(FEATURE_NAMES)),
                feature_value,
                dtype=np.float32,
            ),
            rotation_targets=np.zeros(
                (sample_count, 3), dtype=np.float32
            ),
            translation_targets=np.zeros(
                (sample_count, 3), dtype=np.float32
            ),
            confidence_targets=np.ones(sample_count, dtype=np.float32),
            quality_targets=np.zeros(sample_count, dtype=np.float32),
            group_ids=group_ids,
            feature_names=np.asarray(FEATURE_NAMES),
        )

    def test_external_validation_data_remains_disjoint_with_overlapping_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            training_path = Path(folder) / "train.npz"
            validation_path = Path(folder) / "val.npz"
            self.write_dataset(training_path, [0, 0, 1, 1], 1.0)
            # A separately generated file starts its local IDs at zero again.
            self.write_dataset(validation_path, [0, 0], 9.0)
            arrays, training_groups, validation_groups, metadata = (
                _prepare_recovery_data(
                    training_path,
                    validation_path,
                    0.15,
                    np.random.default_rng(7),
                )
            )
            self.assertTrue(
                set(training_groups).isdisjoint(set(validation_groups))
            )
            training_mask = np.isin(
                arrays["group_ids"], training_groups
            )
            validation_mask = np.isin(
                arrays["group_ids"], validation_groups
            )
            self.assertEqual(int(training_mask.sum()), 4)
            self.assertEqual(int(validation_mask.sum()), 2)
            self.assertTrue(
                np.all(arrays["features"][training_mask] == 1.0)
            )
            self.assertTrue(
                np.all(arrays["features"][validation_mask] == 9.0)
            )
            self.assertEqual(
                metadata["validation_strategy"], "external_dataset"
            )

    def test_external_validation_training_writes_checkpoints_and_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            training_path = root / "train.npz"
            validation_path = root / "val.npz"
            output_dir = root / "output"
            self.write_dataset(training_path, [0, 0, 1, 1], 1.0)
            self.write_dataset(validation_path, [0, 0], 2.0)
            argv = [
                "tracking.train_recovery",
                "--data",
                str(training_path),
                "--validation-data",
                str(validation_path),
                "--output-dir",
                str(output_dir),
                "--epochs",
                "1",
                "--groups-per-batch",
                "1",
                "--hidden-dim",
                "8",
                "--device",
                "cpu",
            ]
            with patch("sys.argv", argv):
                train_recovery_main()
            self.assertTrue((output_dir / "best.ckpt").is_file())
            self.assertTrue((output_dir / "last.ckpt").is_file())
            report = json.loads(
                (output_dir / "run_report.json").read_text()
            )
            self.assertEqual(
                report["validation_strategy"], "external_dataset"
            )
            self.assertEqual(report["training_candidates"], 4)
            self.assertEqual(report["validation_candidates"], 2)


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

    def test_optional_external_identity_prevents_crossing_swap(self):
        pose = pose_from_rt(np.eye(3), [0, 0, 1])
        left_mask = np.zeros((100, 100), dtype=bool)
        right_mask = np.zeros((100, 100), dtype=bool)
        left_mask[40:60, 10:30] = True
        right_mask[40:60, 70:90] = True
        tracks = [
            Track(
                1,
                1,
                pose,
                None,
                np.asarray([10, 40, 20, 20]),
                1,
                TrackMode.NORMAL,
                "",
                1,
                0,
                external_id=11,
                previous_mask=left_mask,
            ),
            Track(
                2,
                1,
                pose,
                None,
                np.asarray([70, 40, 20, 20]),
                1,
                TrackMode.NORMAL,
                "",
                1,
                0,
                external_id=22,
                previous_mask=right_mask,
            ),
        ]
        # The identities crossed, while the detection list remains spatially
        # ordered from left to right.
        detections = [
            Detection(
                0,
                np.asarray([10, 40, 20, 20]),
                left_mask,
                external_id=22,
            ),
            Detection(
                1,
                np.asarray([70, 40, 20, 20]),
                right_mask,
                external_id=11,
            ),
        ]
        legacy_matches, _, _ = associate_tracks(
            tracks,
            detections,
            (100, 100),
            TrackerConfig().association,
        )
        self.assertEqual(legacy_matches, [(0, 0), (1, 1)])

        identity_config = TrackerConfig().association
        identity_config.identity_enabled = True
        identity_config.use_external_id = True
        identity_config.external_id_strict = True
        identity_config.appearance_weight = 0.0
        identity_config.mask_iou_weight = 0.0
        identity_matches, _, _ = associate_tracks(
            tracks,
            detections,
            (100, 100),
            identity_config,
        )
        self.assertEqual(identity_matches, [(0, 1), (1, 0)])


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

    def test_recovery_head_applies_rotation_and_translation_residuals(self):
        config = TrackerConfig()
        scorer = CandidateScorer(SquareRenderer(), config)
        hypothesis = PoseHypothesis(
            pose_from_rt(np.eye(3), [0, 0, 1]), "candidate", 0.5
        )
        evaluated = scorer.evaluate(
            hypothesis, self.frame, self.frame.detections[0]
        )
        predictor = RecoveryPredictor(
            FixedRecoveryModel(),
            feature_mean=np.zeros(len(FEATURE_NAMES), dtype=np.float32),
            feature_std=np.ones(len(FEATURE_NAMES), dtype=np.float32),
            max_rotation_rad=np.pi,
            max_translation_m=1.0,
            device="cpu",
        )
        corrected, _, _ = predictor.correct(evaluated)
        self.assertGreater(corrected.pose[0, 3], hypothesis.pose[0, 3])
        self.assertGreater(
            rotation_error_deg(corrected.pose, hypothesis.pose), 1.0
        )

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

    def test_optional_same_frame_recovery_uses_fresh_global_pose(self):
        def make_tracker(enabled):
            config = TrackerConfig()
            config.same_frame_recovery.enabled = enabled
            config.refinement.normal_iterations = 0
            config.refinement.uncertain_iterations = 0
            config.refinement.lost_iterations = 0
            config.hypotheses.add_flip_when_uncertain = False
            config.hypotheses.center_offsets_px = ()
            config.hypotheses.log_depth_offsets = ()
            config.hypotheses.translation_offsets_m = ()
            config.hypotheses.yaw_offsets_deg = ()
            scorer = CandidateScorer(SquareRenderer(), config)
            return AdaptivePoseTracker(
                config, CandidateRefiner(scorer, config)
            )

        initial = PoseHypothesis(
            pose_from_rt(np.eye(3), [0, 0, 1]), "gigapose", 0.9
        )
        moved = PoseHypothesis(
            pose_from_rt(np.eye(3), [0.3, 0, 1]), "gigapose", 0.9
        )
        second_frame = FrameData(
            1,
            1,
            self.frame.image.copy(),
            self.K,
            [square_detection(center=(80, 50))],
        )

        legacy = make_tracker(False)
        legacy.process_frame(self.frame, [[initial]])
        legacy_result = legacy.process_frame(second_frame, [[moved]])
        self.assertEqual(legacy_result.instances[0].track.mode, TrackMode.LOST)
        self.assertFalse(legacy_result.instances[0].same_frame_recovery)

        improved = make_tracker(True)
        improved.process_frame(self.frame, [[initial]])
        improved_result = improved.process_frame(second_frame, [[moved]])
        self.assertEqual(
            improved_result.instances[0].track.mode, TrackMode.NORMAL
        )
        self.assertTrue(improved_result.instances[0].same_frame_recovery)
        self.assertIn(
            "global_rank", improved_result.instances[0].track.source
        )

    def test_optional_occlusion_scoring_uses_visible_target_silhouette(self):
        observed = np.zeros((100, 100), dtype=bool)
        observed[25:75, 25:50] = True
        detection = Detection(
            0, np.asarray([0, 0, 100, 100]), observed
        )
        frame = FrameData(
            1,
            0,
            np.zeros((100, 100, 3), dtype=np.uint8),
            self.K,
            [detection],
        )
        hypothesis = PoseHypothesis(
            pose_from_rt(np.eye(3), [0, 0, 2]), "test", 0.5
        )
        config = TrackerConfig()
        config.render_scale = 1.0
        config.occlusion.enabled = False
        scorer = CandidateScorer(OcclusionRenderer(), config)
        unoccluded = scorer.evaluate(
            hypothesis,
            frame,
            detection,
            occluder_poses=(hypothesis.pose,),
        )
        config.occlusion.enabled = True
        occlusion_aware = scorer.evaluate(
            hypothesis,
            frame,
            detection,
            occluder_poses=(hypothesis.pose,),
        )
        self.assertAlmostEqual(unoccluded.score.silhouette_iou, 0.5)
        self.assertAlmostEqual(
            occlusion_aware.score.silhouette_iou, 1.0
        )

    def test_occlusion_aware_tracker_scores_each_car_with_the_other_car(self):
        config = TrackerConfig()
        config.occlusion.enabled = True
        config.refinement.lost_iterations = 0
        config.hypotheses.add_flip_when_uncertain = False
        config.hypotheses.center_offsets_px = ()
        config.hypotheses.log_depth_offsets = ()
        config.hypotheses.translation_offsets_m = ()
        config.hypotheses.yaw_offsets_deg = ()
        scorer = CandidateScorer(SquareRenderer(), config)
        tracker = AdaptivePoseTracker(
            config, CandidateRefiner(scorer, config)
        )
        frame = FrameData(
            1,
            0,
            np.zeros((100, 100, 3), dtype=np.uint8),
            self.K,
            [
                square_detection(center=(35, 50)),
                square_detection(center=(65, 50)),
            ],
        )
        groups = [
            [
                PoseHypothesis(
                    pose_from_rt(np.eye(3), [-0.15, 0, 1]),
                    "left",
                    0.9,
                )
            ],
            [
                PoseHypothesis(
                    pose_from_rt(np.eye(3), [0.15, 0, 1]),
                    "right",
                    0.9,
                )
            ],
        ]
        result = tracker.process_frame(frame, groups)
        self.assertEqual(len(result.instances), 2)
        self.assertEqual(
            [item.occluders_used for item in result.instances], [1, 1]
        )

    def test_improvements_are_off_by_default_and_selectable_by_config(self):
        default = TrackerConfig()
        self.assertFalse(default.same_frame_recovery.enabled)
        self.assertFalse(default.association.identity_enabled)
        self.assertFalse(default.occlusion.enabled)
        improved = TrackerConfig.load(
            Path(__file__).parents[1] / "configs" / "improved.json"
        )
        self.assertTrue(improved.same_frame_recovery.enabled)
        self.assertTrue(improved.association.identity_enabled)
        self.assertTrue(improved.occlusion.enabled)

    def test_overlay_compares_original_and_tracked_cad(self):
        config = TrackerConfig()
        config.refinement.lost_iterations = 0
        config.hypotheses.add_flip_when_uncertain = False
        scorer = CandidateScorer(SquareRenderer(), config)
        tracker = AdaptivePoseTracker(config, CandidateRefiner(scorer, config))
        original = PoseHypothesis(
            pose_from_rt(np.eye(3), [0, 0, 1]),
            "gigapose",
            0.9,
        )
        result = tracker.process_frame(self.frame, [[original]])
        original_mask, _ = SquareRenderer().render(
            original.pose, self.K, self.frame.image.shape[:2]
        )

        with tempfile.TemporaryDirectory() as folder:
            output_path = Path(folder) / "overlay.png"
            save_tracking_overlay(
                self.frame,
                result,
                output_path,
                original_gigapose={0: (original, original_mask)},
            )

            output = cv2.imread(str(output_path))
            self.assertIsNotNone(output)
            self.assertEqual(output.shape[:2], self.frame.image.shape[:2])


if __name__ == "__main__":
    unittest.main()
