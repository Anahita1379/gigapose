"""Tests for the standalone fixed-lag translation package."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.model import GRUKalmanFilter
from .inference import run_filter_segments
from .smoother import FixedLagConfig, smooth_segment


class FixedLagTranslationTests(unittest.TestCase):
    def test_isolated_causal_jump_selects_consistent_gigapose(self):
        time = np.arange(5, dtype=float)
        truth = np.column_stack((time, np.zeros(5), 20.0 + time))
        causal = truth.copy()
        causal[2] += np.asarray([12.0, 0.0, 0.0])
        measurement = truth.copy()
        measurement[2] += np.asarray([9.0, 0.0, 0.0])
        gigapose = truth.copy()
        result = smooth_segment(
            causal, measurement, gigapose, time,
            FixedLagConfig(
                minimum_gate_m=0.5,
                depth_gate_fraction=0.0,
                minimum_candidate_improvement_m=0.5,
                gigapose_penalty_m=0.1,
                motion_blend=0.0,
            ),
        )
        self.assertTrue(result.repair_mask[2])
        self.assertEqual(result.selected_source[2], "gigapose")
        np.testing.assert_allclose(result.positions[2], truth[2], atol=1e-5)
        self.assertEqual(result.replay_start_local_index[2], 0)

    def test_smooth_motion_is_not_modified(self):
        time = np.arange(9, dtype=float) * 0.1
        position = np.column_stack((4.0 * time, time**2, 30.0 + 2.0 * time))
        result = smooth_segment(
            position, position, position, time,
            FixedLagConfig(minimum_gate_m=0.25, depth_gate_fraction=0.0),
        )
        self.assertFalse(result.repair_mask.any())
        np.testing.assert_array_equal(result.positions, position)

    def test_short_segment_is_left_unchanged(self):
        time = np.arange(4, dtype=float)
        position = np.column_stack((time, time, time))
        result = smooth_segment(
            position, position + 1, position - 1, time, FixedLagConfig(),
        )
        self.assertFalse(result.repair_mask.any())
        np.testing.assert_array_equal(result.positions, position)

    def test_rejects_nonmonotonic_time(self):
        position = np.zeros((5, 3), dtype=float)
        with self.assertRaises(ValueError):
            smooth_segment(
                position, position, position,
                np.asarray([0.0, 1.0, 1.0, 2.0, 3.0]), FixedLagConfig(),
            )

    def test_segment_replay_propagates_corrected_measurement(self):
        count = 6
        measurement = np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0)
        measurement[:, 0, 3] = np.arange(count, dtype=np.float32)
        arrays = {
            "measurement_pose": measurement,
            "baseline_pose": measurement.copy(),
            "context_features": np.ones((count, 7), dtype=np.float32),
            "delta_time_s": np.asarray([0.0] + [0.1] * (count - 1), dtype=np.float32),
            "segment_id": np.zeros(count, dtype=np.int64),
        }
        torch.manual_seed(3)
        model = GRUKalmanFilter(7, hidden_dim=16, context_hidden=8)
        causal, _ = run_filter_segments(model, arrays, "cpu")
        corrected = measurement.copy()
        corrected[2, 0, 3] -= 2.0
        replayed, _ = run_filter_segments(
            model, arrays, "cpu", measurement_pose=corrected,
        )
        self.assertGreater(np.linalg.norm(replayed[2, :3, 3] - causal[2, :3, 3]), 0)
        self.assertGreater(np.linalg.norm(replayed[3, :3, 3] - causal[3, :3, 3]), 0)


if __name__ == "__main__":
    unittest.main()
