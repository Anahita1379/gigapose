"""Focused tests for sparse sequence factor-graph refinement."""

import unittest

import numpy as np

from tracking.geometry import so3_exp

from .optimizer import (
    FactorGraphConfig,
    normalized_visual_cost,
    optimize_sequence,
    visual_initialization,
)


def _poses(count: int) -> np.ndarray:
    values = np.repeat(np.eye(4)[None], count, axis=0)
    values[:, 0, 3] = np.arange(count, dtype=float)
    return values


class FactorGraphTest(unittest.TestCase):
    def test_candidate_only_graph_repairs_translation_and_rotation_outlier(self):
        count = 9
        truth = _poses(count)
        candidates = np.repeat(truth[:, None], 2, axis=1)
        candidates[4, 0, 0, 3] += 10.0
        candidates[4, 0, :3, :3] = so3_exp(np.asarray([0.0, 0.0, np.pi]))
        valid = np.ones((count, 2), dtype=bool)
        raw_visual = np.tile(np.asarray([0.0, 0.4]), (count, 1))
        visual = normalized_visual_cost(raw_visual, valid)
        initial, _ = visual_initialization(candidates, valid, visual)
        result = optimize_sequence(
            candidates,
            valid,
            visual,
            np.arange(count, dtype=float) * 0.1,
            initial_poses=initial,
            config=FactorGraphConfig(
                outer_iterations=4,
                maximum_nfev=100,
                acceleration_sigma_mps2=5.0,
                angular_acceleration_sigma_radps2=0.5,
            ),
        )
        self.assertLess(
            np.linalg.norm(result.poses[4, :3, 3] - truth[4, :3, 3]), 1.0
        )
        self.assertEqual(int(result.selected_candidate[4]), 1)
        self.assertGreater(result.assignment_changes, 0)

    def test_causal_mode_uses_soft_prior_without_copying_it_exactly(self):
        count = 7
        truth = _poses(count)
        candidates = np.repeat(truth[:, None], 2, axis=1)
        candidates[:, 0, 1, 3] = 1.0
        causal = truth.copy()
        causal[:, 1, 3] = 0.3
        valid = np.ones((count, 2), dtype=bool)
        visual = normalized_visual_cost(
            np.tile(np.asarray([0.0, 0.2]), (count, 1)), valid
        )
        result = optimize_sequence(
            candidates,
            valid,
            visual,
            np.arange(count, dtype=float) * 0.1,
            initial_poses=causal,
            prior_poses=causal,
            config=FactorGraphConfig(
                outer_iterations=2,
                maximum_nfev=60,
                visual_assignment_weight=0.1,
            ),
        )
        self.assertLess(np.mean(np.abs(result.poses[:, 1, 3])), 0.3)
        self.assertGreater(np.mean(np.abs(result.poses[:, 1, 3])), 0.0)

    def test_visual_cost_masks_padding(self):
        cost = normalized_visual_cost(
            np.asarray([[2.0, 1.0, 99.0]]),
            np.asarray([[True, True, False]]),
        )
        self.assertEqual(float(cost[0, 1]), 0.0)
        self.assertEqual(float(cost[0, 0]), 1.0)
        self.assertTrue(np.isinf(cost[0, 2]))


if __name__ == "__main__":
    unittest.main()
