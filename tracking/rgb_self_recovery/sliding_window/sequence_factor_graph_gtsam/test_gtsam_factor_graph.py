"""Focused tests for the isolated GTSAM sequence optimizer."""

import importlib.util
import unittest

import numpy as np

from tracking.geometry import so3_exp

from .optimizer import (
    GTSAMGraphConfig,
    normalized_visual_cost,
    optimize_sequence,
    visual_initialization,
)


GTSAM_AVAILABLE = importlib.util.find_spec("gtsam") is not None


def _poses(count: int) -> np.ndarray:
    values = np.repeat(np.eye(4)[None], count, axis=0)
    values[:, 0, 3] = np.arange(count, dtype=float)
    return values


@unittest.skipUnless(GTSAM_AVAILABLE, "gtsam==4.2 is not installed")
class GTSAMFactorGraphTest(unittest.TestCase):
    def test_candidate_graph_repairs_pose_outlier(self):
        count = 9
        truth = _poses(count)
        candidates = np.repeat(truth[:, None], 2, axis=1)
        candidates[4, 0, 0, 3] += 10.0
        candidates[4, 0, :3, :3] = so3_exp(np.asarray([0.0, 0.0, np.pi]))
        valid = np.ones((count, 2), dtype=bool)
        visual = normalized_visual_cost(
            np.tile(np.asarray([0.0, 0.4]), (count, 1)), valid
        )
        initial, _ = visual_initialization(candidates, valid, visual)
        result = optimize_sequence(
            candidates,
            valid,
            visual,
            np.arange(count, dtype=float) * 0.1,
            initial_poses=initial,
            config=GTSAMGraphConfig(
                outer_iterations=4,
                maximum_iterations=100,
                acceleration_sigma_mps2=5.0,
                angular_acceleration_sigma_radps2=0.5,
            ),
        )
        self.assertTrue(result.solution_accepted, result.optimizer_message)
        self.assertLess(
            np.linalg.norm(result.poses[4, :3, 3] - truth[4, :3, 3]), 1.0
        )
        self.assertEqual(int(result.selected_candidate[4]), 1)

    def test_soft_prior_is_not_copied_exactly(self):
        count = 7
        truth = _poses(count)
        candidates = np.repeat(truth[:, None], 2, axis=1)
        candidates[:, 0, 1, 3] = 1.0
        prior = truth.copy()
        prior[:, 1, 3] = 0.3
        valid = np.ones((count, 2), dtype=bool)
        visual = normalized_visual_cost(
            np.tile(np.asarray([0.0, 0.2]), (count, 1)), valid
        )
        result = optimize_sequence(
            candidates,
            valid,
            visual,
            np.arange(count, dtype=float) * 0.1,
            initial_poses=prior,
            prior_poses=prior,
            config=GTSAMGraphConfig(
                outer_iterations=2,
                maximum_iterations=60,
                visual_assignment_weight=0.1,
            ),
        )
        self.assertTrue(result.solution_accepted, result.optimizer_message)
        mean_offset = float(np.mean(np.abs(result.poses[:, 1, 3])))
        self.assertLess(mean_offset, 1.0)
        self.assertGreater(mean_offset, 0.0)
        self.assertNotAlmostEqual(mean_offset, 0.3, places=5)


class VisualCostTest(unittest.TestCase):
    def test_invalid_padding_is_ignored(self):
        cost = normalized_visual_cost(
            np.asarray([[2.0, 1.0, 99.0]]),
            np.asarray([[True, True, False]]),
        )
        self.assertEqual(float(cost[0, 1]), 0.0)
        self.assertEqual(float(cost[0, 0]), 1.0)
        self.assertTrue(np.isinf(cost[0, 2]))


if __name__ == "__main__":
    unittest.main()
