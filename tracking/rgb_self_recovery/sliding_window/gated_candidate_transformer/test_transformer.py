import unittest
from argparse import Namespace

import numpy as np
import torch

from .infer import _decode_window, _validate_fixed_lag
from .model import GatedCandidateTransformer
from .orientation import AdaptiveFixedLagOrientationGuard
from .train import _step


def _pose(yaw_deg: float = 0.0) -> np.ndarray:
    angle = np.deg2rad(yaw_deg)
    pose = np.eye(4)
    pose[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    return pose


class ModelTest(unittest.TestCase):
    def test_shapes_and_mask(self):
        model = GatedCandidateTransformer(23, 8, window_length=8, candidate_count=16)
        candidate = torch.randn(2, 8, 16, 23)
        frame = torch.randn(2, 8, 8)
        candidate_valid = torch.ones(2, 8, 16, dtype=torch.bool)
        frame_valid = torch.ones(2, 8, dtype=torch.bool)
        candidate_valid[:, :, -2:] = False
        poses = torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(2, 8, 16, 1, 1)
        times = torch.arange(8, dtype=torch.float32)[None].repeat(2, 1) * 0.1
        output = model(
            candidate, frame, candidate_valid, frame_valid,
            candidate_poses=poses, time_s=times,
        )
        self.assertEqual(output["candidate_logits"].shape, (2, 8, 16))
        self.assertEqual(output["residual"].shape, (2, 8, 16, 6))
        self.assertEqual(output["log_sigma"].shape, (2, 8, 16, 2))
        self.assertEqual(output["abstention_logits"].shape, (2, 8))
        self.assertEqual(output["translation_trust_logits"].shape, (2, 8, 16))
        self.assertEqual(output["rotation_trust_logits"].shape, (2, 8, 16))
        self.assertEqual(output["translation_fixed_lag_delta"].shape, (2, 8, 3))
        self.assertEqual(output["translation_fixed_lag_gate"].shape, (2, 8))
        self.assertTrue(torch.isfinite(output["residual"]).all())

    def test_lag_four_mask_preserves_target_future_and_blocks_leakage(self):
        mask = GatedCandidateTransformer.temporal_attention_mask(8, 4, "cpu")
        expected_row_zero = torch.tensor(
            [False, False, False, False, False, True, True, True]
        )
        self.assertTrue(torch.equal(mask[0], expected_row_zero))
        # The fixed-lag target is 8 - 4 - 1 = position 3. It may use all
        # observed positions 0..7, including exactly four future frames.
        self.assertFalse(bool(mask[3].any()))

    def test_masked_attention_output_ignores_frames_beyond_lag(self):
        torch.manual_seed(7)
        model = GatedCandidateTransformer(
            5, 3, window_length=4, candidate_count=4,
            model_dim=16, heads=4, feedforward_dim=32,
            cross_layers=2, dropout=0.0, attention_future_lag=2,
        ).eval()
        candidate = torch.randn(1, 4, 4, 5)
        frame = torch.randn(1, 4, 3)
        candidate_valid = torch.ones(1, 4, 4, dtype=torch.bool)
        frame_valid = torch.ones(1, 4, dtype=torch.bool)
        poses = torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(1, 4, 4, 1, 1)
        times = torch.arange(4, dtype=torch.float32)[None] * 0.1
        first = model(
            candidate, frame, candidate_valid, frame_valid,
            candidate_poses=poses, time_s=times,
        )
        changed_candidate = candidate.clone()
        changed_frame = frame.clone()
        changed_candidate[:, 3] += 100.0
        changed_frame[:, 3] -= 100.0
        second = model(
            changed_candidate, changed_frame, candidate_valid, frame_valid,
            candidate_poses=poses, time_s=times,
        )
        # Query zero may see positions 0..2 with lag two, but never position 3.
        for key in (
            "candidate_logits",
            "residual",
            "translation_trust_logits",
            "rotation_trust_logits",
            "translation_fixed_lag_delta",
            "translation_fixed_lag_gate",
        ):
            self.assertTrue(
                torch.allclose(first[key][:, 0], second[key][:, 0], atol=1e-6),
                key,
            )

    def test_inference_rejects_attention_lag_mismatch(self):
        config = {"window_length": 8, "attention_future_lag": 4}
        self.assertEqual(_validate_fixed_lag(4, config), 4)
        with self.assertRaisesRegex(ValueError, "attention_future_lag=4"):
            _validate_fixed_lag(0, config)

    def test_translation_and_rotation_fallback_are_independent(self):
        poses = torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(1, 1, 2, 1, 1)
        poses[..., 0, :3, 3] = torch.tensor([1.0, 0.0, 0.0])
        poses[..., 1, :3, 3] = torch.tensor([5.0, 0.0, 0.0])
        poses[..., 1, :3, :3] = torch.as_tensor(_pose(90)[:3, :3]).float()
        batch = {
            "poses": poses,
            "candidate_valid": torch.ones(1, 1, 2, dtype=torch.bool),
        }
        output = {
            "candidate_logits": torch.tensor([[[0.0, 10.0]]]),
            "orientation_logits": torch.full((1, 1, 2), 10.0),
            "residual": torch.zeros(1, 1, 2, 6),
            "log_sigma": torch.full((1, 1, 2, 2), -5.0),
            "abstention_logits": torch.full((1, 1), 10.0),
            "translation_trust_logits": torch.tensor([[[0.0, 10.0]]]),
            "rotation_trust_logits": torch.tensor([[[0.0, -10.0]]]),
            "translation_fixed_lag_delta": torch.zeros(1, 1, 3),
            "translation_fixed_lag_gate": torch.zeros(1, 1),
        }
        args = Namespace(
            orientation_gate_weight=0.0,
            max_correction_translation_m=20.0,
            max_correction_rotation_deg=45.0,
            minimum_accept_probability=0.5,
            minimum_candidate_probability=0.05,
            max_translation_uncertainty_m=20.0,
            max_rotation_uncertainty_deg=90.0,
            minimum_translation_trust=0.5,
            minimum_rotation_trust=0.5,
        )
        decoded = _decode_window(batch, output, args, 1.0, 20.0)
        self.assertAlmostEqual(float(decoded["pose"][0, 0, 0, 3]), 5.0)
        self.assertTrue(torch.allclose(
            decoded["pose"][0, 0, :3, :3], poses[0, 0, 0, :3, :3]
        ))
        self.assertFalse(bool(decoded["translation_fallback"][0, 0]))
        self.assertTrue(bool(decoded["rotation_fallback"][0, 0]))

        output["translation_trust_logits"][..., 1] = -10.0
        output["rotation_trust_logits"][..., 1] = 10.0
        decoded = _decode_window(batch, output, args, 1.0, 20.0)
        self.assertAlmostEqual(float(decoded["pose"][0, 0, 0, 3]), 1.0)
        self.assertTrue(torch.allclose(
            decoded["pose"][0, 0, :3, :3], poses[0, 0, 1, :3, :3]
        ))
        self.assertTrue(bool(decoded["translation_fallback"][0, 0]))
        self.assertFalse(bool(decoded["rotation_fallback"][0, 0]))

    def test_training_step_supervises_both_trust_heads(self):
        model = GatedCandidateTransformer(
            5, 3, window_length=4, candidate_count=4,
            model_dim=16, heads=4, feedforward_dim=32,
            attention_future_lag=2,
        )
        poses = torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(1, 4, 4, 1, 1)
        poses[..., :3, 3] = torch.randn(1, 4, 4, 3) * 0.1
        target = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 4, 1, 1)
        batch = {
            "candidate_features": torch.randn(1, 4, 4, 5),
            "frame_features": torch.randn(1, 4, 3),
            "candidate_valid": torch.ones(1, 4, 4, dtype=torch.bool),
            "frame_valid": torch.ones(1, 4, dtype=torch.bool),
            "poses": poses,
            "ground_truth_pose": target,
            "labels": torch.zeros(1, 4, dtype=torch.long),
            "oracle_costs": torch.rand(1, 4, 4),
            "rotation_error_deg": torch.rand(1, 4, 4) * 30.0,
            "time_s": torch.arange(4, dtype=torch.float32)[None] * 0.1,
        }
        args = Namespace(
            max_oracle_cost=20.0,
            orientation_correct_deg=90.0,
            validation_orientation_gate_weight=1.0,
            translation_scale_m=1.0,
            rotation_scale_deg=20.0,
            abstention_improvement_margin=0.05,
            translation_fallback_margin_m=0.0,
            rotation_fallback_margin_deg=0.0,
            candidate_weight=1.0,
            expected_cost_weight=0.1,
            orientation_weight=0.5,
            translation_weight=1.0,
            rotation_weight=1.0,
            abstention_weight=0.25,
            translation_trust_weight=0.5,
            rotation_trust_weight=0.5,
            velocity_weight=0.1,
            acceleration_weight=0.05,
            jerk_weight=0.02,
            fixed_lag_regularization_weight=0.01,
            velocity_scale_mps=10.0,
            acceleration_scale_mps2=20.0,
            jerk_scale_mps3=100.0,
            maximum_translation_refinement_m=20.0,
            gradient_clip=5.0,
        )
        metrics = _step(model, batch, "cpu", args)
        self.assertTrue(np.isfinite(metrics["translation_trust_loss"]))
        self.assertTrue(np.isfinite(metrics["rotation_trust_loss"]))
        self.assertTrue(np.isfinite(metrics["rotation_trust_accuracy"]))
        self.assertTrue(np.isfinite(metrics["acceleration_loss"]))
        self.assertTrue(np.isfinite(metrics["jerk_loss"]))
        self.assertTrue(np.isfinite(metrics["fixed_lag_regularization"]))


class OrientationTest(unittest.TestCase):
    def _stable_guard(self):
        guard = AdaptiveFixedLagOrientationGuard()
        for index in range(6):
            guard.resolve(
                _pose(), time_s=float(index), future_poses=[_pose()],
                future_orientation_probabilities=[0.99],
            )
        return guard

    def test_isolated_flip_uses_temporal_orientation(self):
        guard = self._stable_guard()
        decision = guard.resolve(
            _pose(180), time_s=7.0,
            future_poses=[_pose(180), _pose()],
            future_orientation_probabilities=[0.99, 0.99],
        )
        self.assertTrue(decision.temporal_fallback)
        self.assertFalse(decision.flip_confirmed)
        self.assertLess(abs(decision.pose[0, 0] - 1.0), 1e-6)

    def test_consistent_future_confirms_real_flip(self):
        guard = self._stable_guard()
        future = [(_pose(180), 0.99) for _ in range(5)]
        decision = guard.resolve(
            _pose(180), time_s=7.0,
            future_poses=[item[0] for item in future],
            future_orientation_probabilities=[item[1] for item in future],
        )
        self.assertTrue(decision.flip_confirmed)
        self.assertFalse(decision.temporal_fallback)
        self.assertLess(abs(decision.pose[0, 0] + 1.0), 1e-6)


if __name__ == "__main__":
    unittest.main()
