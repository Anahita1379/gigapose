import unittest

import numpy as np

from tracking.geometry import rotation_error_deg, so3_exp

from .optimizer import (
    BootstrapConfig,
    optimize_candidate_fixed_lag,
    optimize_polarity_fixed_lag,
    polarity_rotations,
)


class OrientationBootstrapTest(unittest.TestCase):
    def setUp(self):
        self.flip = so3_exp(np.asarray([0.0, 0.0, np.pi]))

    def test_polarity_alternatives_are_half_turn_apart(self):
        values = polarity_rotations(np.eye(3), self.flip)
        self.assertAlmostEqual(rotation_error_deg(values[0], values[1]), 180.0, places=6)

    def test_anchor_unary_corrects_flipped_sequence(self):
        frames = 8
        wrong = np.repeat(self.flip[None], frames, axis=0)
        alternatives = np.stack(
            [polarity_rotations(rotation, self.flip) for rotation in wrong]
        )
        unary = np.tile(np.asarray([25.0, 0.0]), (frames, 1))
        states, _ = optimize_polarity_fixed_lag(
            alternatives, unary, BootstrapConfig(window_length=5)
        )
        np.testing.assert_array_equal(states, np.ones(frames, dtype=np.int64))
        for index, state in enumerate(states):
            self.assertLess(rotation_error_deg(alternatives[index, state], np.eye(3)), 1e-5)

    def test_motion_rejects_single_frame_polarity_spike(self):
        frames = 7
        alternatives = np.stack(
            [polarity_rotations(np.eye(3), self.flip) for _ in range(frames)]
        )
        unary = np.zeros((frames, 2))
        unary[:, 1] = 2.0
        unary[3] = [1.0, 0.0]
        states, _ = optimize_polarity_fixed_lag(
            alternatives,
            unary,
            BootstrapConfig(
                window_length=5, motion_weight=1.0,
                motion_scale_deg=20.0, switch_penalty=2.0,
            ),
        )
        np.testing.assert_array_equal(states, np.zeros(frames, dtype=np.int64))

    def test_candidate_optimizer_uses_future_anchor_support(self):
        frames, states = 5, 2
        poses = np.tile(np.eye(4), (frames, states, 1, 1))
        poses[:, 1, :3, :3] = self.flip
        unary = np.tile(np.asarray([0.0, 8.0]), (frames, 1))
        unary[0] = [3.0, 0.0]
        selected, _ = optimize_candidate_fixed_lag(
            poses, unary, np.ones((frames, states), dtype=bool),
            np.asarray([0, 1]), BootstrapConfig(window_length=5),
        )
        self.assertEqual(int(selected[0]), 0)


if __name__ == "__main__":
    unittest.main()
