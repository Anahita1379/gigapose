"""Tests for causal temporal orientation candidates and flip hysteresis."""

from __future__ import annotations

import math
import unittest

import numpy as np

from tracking.geometry import rotate_pose, rotation_error_deg

from .temporal_orientation import FixedLagOrientationResolver, TemporalOrientationSelector


def pose(degrees: float) -> np.ndarray:
    return rotate_pose(
        np.eye(4), np.asarray([0.0, 0.0, math.radians(degrees)]), side="right"
    )


class TemporalOrientationTests(unittest.TestCase):
    def selector(self, confirmation: int = 3) -> TemporalOrientationSelector:
        return TemporalOrientationSelector(
            flip_confirmation_frames=confirmation,
            stable_orientation_frames=4,
            stable_flip_confirmation_frames=5,
            maximum_prediction_step_deg=20.0,
        )

    def run_frame(self, selector, candidates, probabilities, time_s):
        poses = np.stack(candidates)
        valid = np.ones(len(candidates), dtype=bool)
        selected, predicted, angles, reranked = selector.rerank(
            poses, np.asarray(probabilities), valid, time_s
        )
        decision = selector.enforce(
            poses[selected], selected, np.asarray(probabilities), valid,
            predicted, angles, time_s,
        )
        decision.temporal_reranked = reranked
        return decision

    def test_single_flip_is_rejected_to_temporal_orientation(self):
        selector = self.selector()
        first = self.run_frame(selector, [pose(0), pose(180)], [.9, .1], 0.0)
        # Model the observed failure: every available visual candidate is in
        # the wrong front/rear mode, so reranking cannot find a normal one.
        second = self.run_frame(selector, [pose(180), pose(170)], [.9, .1], 0.1)
        self.assertFalse(first.temporal_fallback)
        self.assertTrue(second.temporal_fallback)
        self.assertLess(rotation_error_deg(second.pose, pose(0)), 1.0)

    def test_persistent_flip_is_accepted_after_confirmation(self):
        selector = self.selector(confirmation=3)
        self.run_frame(selector, [pose(0), pose(180)], [.9, .1], 0.0)
        decisions = [
            self.run_frame(selector, [pose(180), pose(170)], [.99, .01], time)
            for time in (0.1, 0.2, 0.3)
        ]
        self.assertTrue(decisions[0].temporal_fallback)
        self.assertTrue(decisions[1].temporal_fallback)
        self.assertTrue(decisions[2].flip_confirmed)
        self.assertLess(rotation_error_deg(decisions[2].pose, pose(180)), 1.0)

    def test_soft_penalty_can_rerank_a_large_rotation_jump(self):
        selector = self.selector()
        self.run_frame(selector, [pose(0), pose(180)], [.9, .1], 0.0)
        poses = np.stack([pose(10), pose(80)])
        selected, _predicted, _angles, reranked = selector.rerank(
            poses, np.asarray([.45, .55]), np.ones(2, dtype=bool), 0.1
        )
        self.assertTrue(reranked)
        self.assertEqual(selected, 0)

    def test_flip_like_visual_winner_reaches_hysteresis(self):
        selector = self.selector()
        self.run_frame(selector, [pose(0), pose(180)], [.9, .1], 0.0)
        poses = np.stack([pose(0), pose(180)])
        selected, _predicted, angles, reranked = selector.rerank(
            poses, np.asarray([.01, .99]), np.ones(2, dtype=bool), 0.1
        )
        self.assertFalse(reranked)
        self.assertEqual(selected, 1)
        decision = selector.enforce(
            poses[selected], selected, np.asarray([.01, .99]),
            np.ones(2, dtype=bool), _predicted, angles, 0.1,
        )
        self.assertTrue(decision.temporal_fallback)
        self.assertEqual(decision.flip_pending_frames, 1)

    def test_stable_state_requires_long_confirmation_and_cancels_short_burst(self):
        selector = self.selector(confirmation=2)
        for index in range(4):
            decision = self.run_frame(
                selector, [pose(0), pose(180)], [.99, .01], index * 0.1
            )
        self.assertGreaterEqual(decision.orientation_stable_frames, 4)
        self.assertTrue(decision.orientation_state_locked)
        burst = [
            self.run_frame(selector, [pose(180), pose(170)], [.99, .01], time)
            for time in (0.4, 0.5, 0.6)
        ]
        self.assertTrue(all(item.temporal_fallback for item in burst))
        self.assertTrue(all(not item.flip_confirmed for item in burst))
        self.assertEqual(burst[-1].required_flip_confirmation_frames, 5)
        recovered = self.run_frame(
            selector, [pose(0), pose(180)], [.99, .01], 0.7
        )
        self.assertFalse(recovered.temporal_fallback)
        self.assertEqual(recovered.flip_pending_frames, 0)
        self.assertLess(rotation_error_deg(recovered.pose, pose(0)), 1.0)

    def test_stable_state_accepts_five_consistent_flip_frames(self):
        selector = self.selector(confirmation=2)
        for index in range(4):
            self.run_frame(selector, [pose(0), pose(180)], [.99, .01], index * 0.1)
        decisions = [
            self.run_frame(selector, [pose(180), pose(170)], [.99, .01], time)
            for time in (0.4, 0.5, 0.6, 0.7, 0.8)
        ]
        self.assertTrue(all(item.temporal_fallback for item in decisions[:4]))
        self.assertTrue(decisions[4].flip_confirmed)
        self.assertEqual(decisions[4].orientation_stable_frames, 1)

    def test_fixed_lag_backfills_entire_confirmed_pending_interval(self):
        selector = self.selector(confirmation=2)
        resolver = FixedLagOrientationResolver()
        for index in range(4):
            decision = self.run_frame(
                selector, [pose(0), pose(180)], [.99, .01], index * 0.1
            )
            self.assertEqual(resolver.update(index, decision), [])
        resolved = []
        decisions = []
        for index, time in enumerate((0.4, 0.5, 0.6, 0.7, 0.8), start=4):
            decision = self.run_frame(
                selector, [pose(180), pose(170)], [.99, .01], time
            )
            decisions.append(decision)
            resolved = resolver.update(index, decision)
        self.assertTrue(decisions[-1].flip_confirmed)
        # The confirming frame already emits the new visual orientation. The
        # four earlier fallback frames are the rows that need revision.
        self.assertEqual(resolved, [4, 5, 6, 7])
        self.assertEqual(resolver.pending_indices, [])

    def test_fixed_lag_cancellation_keeps_fallback_rows(self):
        selector = self.selector(confirmation=2)
        resolver = FixedLagOrientationResolver()
        for index in range(4):
            decision = self.run_frame(
                selector, [pose(0), pose(180)], [.99, .01], index * 0.1
            )
            resolver.update(index, decision)
        for index, time in enumerate((0.4, 0.5, 0.6), start=4):
            decision = self.run_frame(
                selector, [pose(180), pose(170)], [.99, .01], time
            )
            self.assertEqual(resolver.update(index, decision), [])
        cancellation = self.run_frame(
            selector, [pose(0), pose(180)], [.99, .01], 0.7
        )
        self.assertEqual(resolver.update(7, cancellation), [])
        self.assertEqual(resolver.pending_indices, [])

    def test_fixed_lag_inconsistent_flip_restarts_pending_interval(self):
        resolver = FixedLagOrientationResolver()
        # Use real decisions so this test also protects the state-machine
        # contract consumed by the resolver.
        selector = self.selector(confirmation=2)
        for index in range(4):
            stable = self.run_frame(
                selector, [pose(0), pose(180)], [.99, .01], index * 0.1
            )
            resolver.update(index, stable)
        first = self.run_frame(selector, [pose(180), pose(170)], [.99, .01], 0.4)
        resolver.update(4, first)
        # A substantially different flip-like orientation restarts at one.
        x_flip = rotate_pose(
            np.eye(4), np.asarray([math.pi, 0.0, 0.0]), side="right"
        )
        x_flip_nearby = rotate_pose(
            np.eye(4), np.asarray([math.radians(170), 0.0, 0.0]), side="right"
        )
        restarted = self.run_frame(
            selector, [x_flip, x_flip_nearby], [.99, .01], 0.5
        )
        self.assertEqual(restarted.flip_pending_frames, 1)
        resolver.update(5, restarted)
        self.assertEqual(resolver.pending_indices, [5])
        resolver.reset()
        self.assertEqual(resolver.pending_indices, [])


if __name__ == "__main__":
    unittest.main()
