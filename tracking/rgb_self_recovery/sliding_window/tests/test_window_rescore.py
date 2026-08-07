"""Regression tests for post-smoothing visual verification."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from tracking.rgb_self_recovery.inference import RecoveryResult
from tracking.rgb_self_recovery.sliding_window.runner import (
    _apply_recovery_abstention,
    _rescore_window_endpoint,
    _select_with_soft_orientation,
    _state_confidence,
)
from tracking.types import Track, TrackMode


def result(z: float, *, iou: float, quality: float, source: str) -> RecoveryResult:
    pose = np.eye(4)
    pose[2, 3] = z
    return RecoveryResult(
        pose=pose,
        source=source,
        confidence=0.9,
        quality=quality,
        silhouette_iou=iou,
        total_error=quality + 1.5 * (1.0 - iou),
        delta_center_crop_px=np.zeros(2),
        delta_log_depth=0.0,
        delta_rotation_rad=np.zeros(3),
        rendered_mask_crop=np.zeros((4, 4), dtype=bool),
    )


class Predictor:
    def __init__(self, rescored: RecoveryResult):
        self.rescored = rescored

    def refine_and_score(self, *_args, **_kwargs):
        return [self.rescored]


class WindowRescoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.args = SimpleNamespace(
            quality_weight=1.0,
            confidence_weight=0.5,
            silhouette_weight=1.5,
            window_max_visual_cost_increase=0.02,
            window_max_iou_drop=0.02,
        )
        self.anchor = result(10.0, iou=0.8, quality=0.2, source="verified")
        self.smoothed = result(12.0, iou=0.8, quality=0.2, source="verified|window5")

    def run_rescore(self, rescored: RecoveryResult):
        diagnostics = {
            "window_size_used": 5,
            "window_continuous_refinement": 1,
        }
        selected = _rescore_window_endpoint(
            Predictor(rescored),
            None,
            None,
            None,
            None,
            self.smoothed,
            self.anchor,
            self.args,
            diagnostics,
        )
        return selected, diagnostics

    def test_rejects_smoothed_pose_when_iou_drops(self) -> None:
        selected, diagnostics = self.run_rescore(
            result(12.0, iou=0.5, quality=0.2, source="rescored")
        )
        self.assertTrue(np.array_equal(selected.pose, self.anchor.pose))
        self.assertEqual(diagnostics["window_continuous_accepted"], 0)
        self.assertGreater(diagnostics["window_iou_drop"], 0.02)

    def test_accepts_and_updates_scores_when_visual_evidence_holds(self) -> None:
        selected, diagnostics = self.run_rescore(
            result(12.0, iou=0.81, quality=0.19, source="rescored")
        )
        self.assertTrue(np.array_equal(selected.pose, self.smoothed.pose))
        self.assertEqual(selected.silhouette_iou, 0.81)
        self.assertEqual(diagnostics["window_continuous_accepted"], 1)
        self.assertIn("continuous_verified", selected.source)

    def test_soft_orientation_reranks_without_rejecting_candidates(self) -> None:
        anchor_pose = np.eye(4)
        flipped_pose = anchor_pose.copy()
        flipped_pose[:3, :3] = np.diag([-1.0, -1.0, 1.0])
        flipped = result(10.0, iou=0.8, quality=0.1, source="flipped")
        flipped.pose = flipped_pose
        flipped.total_error = 0.1
        correct = result(10.0, iou=0.8, quality=0.15, source="correct")
        correct.pose = anchor_pose.copy()
        correct.total_error = 0.15
        track = Track(
            track_id=0,
            obj_id=1,
            pose=anchor_pose,
            previous_pose=None,
            bbox_xywh=np.asarray([0.0, 0.0, 10.0, 10.0]),
            confidence=0.9,
            mode=TrackMode.NORMAL,
            source="previous",
            scene_id=1,
            im_id=1,
        )
        args = SimpleNamespace(
            normal_max_rotation_step_deg=30.0,
            uncertain_max_rotation_step_deg=60.0,
            max_rank0_rotation_disagreement_deg=90.0,
            soft_orientation_sigma_deg=30.0,
            soft_orientation_previous_weight=0.25,
            soft_orientation_rank0_weight=0.05,
        )
        selected, ranked, diagnostics = _select_with_soft_orientation(
            [flipped, correct], track=track, rank0_pose=flipped_pose, args=args
        )
        self.assertEqual(selected.source, "correct")
        self.assertEqual(len(ranked), 2)
        self.assertEqual(diagnostics["orientation_candidates_rejected"], 0)
        self.assertEqual(diagnostics["orientation_gate_triggered"], 1)

    def test_iou_floor_does_not_promote_near_zero_verifier_score(self) -> None:
        aligned = result(10.0, iou=0.6, quality=0.2, source="aligned")
        aligned.confidence = 0.0
        state, raw, floor = _state_confidence(
            aligned,
            SimpleNamespace(
                state_iou_confidence_weight=0.5,
                state_iou_min_verifier_confidence=0.05,
            ),
        )
        self.assertEqual(raw, 0.0)
        self.assertEqual(floor, 0.0)
        self.assertEqual(state, 0.0)

    def test_iou_floor_can_support_a_nonzero_verifier_score(self) -> None:
        aligned = result(10.0, iou=0.6, quality=0.2, source="aligned")
        aligned.confidence = 0.1
        state, raw, floor = _state_confidence(
            aligned,
            SimpleNamespace(
                state_iou_confidence_weight=0.5,
                state_iou_min_verifier_confidence=0.05,
            ),
        )
        self.assertAlmostEqual(floor, 0.3)
        self.assertAlmostEqual(state, 0.3)
        self.assertLess(raw, state)

    @staticmethod
    def abstention_args() -> SimpleNamespace:
        return SimpleNamespace(
            recovery_abstention=True,
            recovery_min_verifier_confidence=0.05,
            recovery_large_translation_m=1.0,
            recovery_large_rotation_deg=45.0,
            normal_max_rotation_step_deg=30.0,
            uncertain_max_rotation_step_deg=60.0,
            window_max_relative_speed_mps=80.0,
            window_fps=20.0,
        )

    def test_abstains_from_unsupported_low_confidence_large_jump(self) -> None:
        baseline = result(10.0, iou=0.4, quality=2.0, source="rank0")
        candidate = result(12.0, iou=0.7, quality=2.0, source="recovered")
        candidate.confidence = 0.01
        selected, ranked, diagnostics = _apply_recovery_abstention(
            candidate, [candidate], baseline, None, self.abstention_args()
        )
        self.assertTrue(np.array_equal(selected.pose, baseline.pose))
        self.assertEqual(len(ranked), 1)
        self.assertIn("recovery_abstention_rank0", selected.source)
        self.assertEqual(diagnostics["recovery_abstention_triggered"], 1)

    def test_keeps_temporally_supported_low_confidence_recovery(self) -> None:
        baseline = result(10.0, iou=0.4, quality=2.0, source="rank0")
        candidate = result(12.0, iou=0.7, quality=2.0, source="recovered")
        candidate.confidence = 0.01
        track = Track(
            track_id=0,
            obj_id=1,
            pose=candidate.pose.copy(),
            previous_pose=None,
            bbox_xywh=np.asarray([0.0, 0.0, 10.0, 10.0]),
            confidence=0.8,
            mode=TrackMode.NORMAL,
            source="previous",
            scene_id=1,
            im_id=1,
        )
        selected, _ranked, diagnostics = _apply_recovery_abstention(
            candidate, [candidate], baseline, track, self.abstention_args()
        )
        self.assertEqual(selected.source, "recovered")
        self.assertEqual(diagnostics["recovery_temporally_supported"], 1)
        self.assertEqual(diagnostics["recovery_abstention_triggered"], 0)


if __name__ == "__main__":
    unittest.main()
