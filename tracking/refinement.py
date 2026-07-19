"""Derivative-free CAD alignment refinement for adaptive pose candidates."""

from __future__ import annotations

import math

import numpy as np

from tracking.config import TrackerConfig
from tracking.geometry import rotate_pose, shift_projected_center
from tracking.scoring import CandidateScorer
from tracking.types import (
    Detection,
    EvaluatedHypothesis,
    FrameData,
    PoseHypothesis,
    TrackMode,
)


class CandidateRefiner:
    """Small coordinate search over image center, log-depth, and rotation."""

    def __init__(self, scorer: CandidateScorer, config: TrackerConfig):
        self.scorer = scorer
        self.config = config

    def _mode_settings(self, mode: TrackMode) -> tuple[int, int]:
        refinement = self.config.refinement
        if mode == TrackMode.NORMAL:
            return refinement.normal_iterations, refinement.normal_beam
        if mode == TrackMode.UNCERTAIN:
            return refinement.uncertain_iterations, refinement.uncertain_beam
        return refinement.lost_iterations, refinement.lost_beam

    def _score(
        self,
        hypotheses: list[PoseHypothesis],
        frame: FrameData,
        detection: Detection,
        motion_reference_pose: np.ndarray | None,
    ) -> list[EvaluatedHypothesis]:
        evaluated = [
            self.scorer.evaluate(
                item, frame, detection, motion_reference_pose, keep_render=False
            )
            for item in hypotheses
        ]
        return sorted(evaluated, key=lambda item: item.score.total_error)

    def refine(
        self,
        hypotheses: list[PoseHypothesis],
        frame: FrameData,
        detection: Detection,
        mode: TrackMode,
        motion_reference_pose: np.ndarray | None,
    ) -> list[EvaluatedHypothesis]:
        if not hypotheses:
            return []
        iterations, beam_width = self._mode_settings(mode)
        ranked = self._score(hypotheses, frame, detection, motion_reference_pose)
        beam = ranked[: max(beam_width, 1)]
        for iteration in range(max(iterations, 0)):
            decay = self.config.refinement.step_decay**iteration
            center_step = self.config.refinement.center_step_px * decay
            depth_step = self.config.refinement.log_depth_step * decay
            rotation_step = math.radians(
                self.config.refinement.rotation_step_deg * decay
            )
            expanded: list[PoseHypothesis] = [
                item.hypothesis.copy() for item in beam
            ]
            for evaluated in beam:
                seed = evaluated.hypothesis
                for axis in (0, 1):
                    for sign in (-1.0, 1.0):
                        delta = np.zeros(2)
                        delta[axis] = sign * center_step
                        expanded.append(
                            PoseHypothesis(
                                shift_projected_center(seed.pose, frame.K, delta),
                                f"{seed.source}|ref_uv{axis}{sign:+.0f}",
                                seed.measurement_score,
                                seed.obj_id,
                            )
                        )
                for sign in (-1.0, 1.0):
                    expanded.append(
                        PoseHypothesis(
                            shift_projected_center(
                                seed.pose,
                                frame.K,
                                delta_log_depth=sign * depth_step,
                            ),
                            f"{seed.source}|ref_logz{sign:+.0f}",
                            seed.measurement_score,
                            seed.obj_id,
                        )
                    )
                axes = (0, 1, 2) if self.config.refinement.refine_full_rotation else (2,)
                for axis in axes:
                    for sign in (-1.0, 1.0):
                        rotvec = np.zeros(3)
                        rotvec[axis] = sign * rotation_step
                        expanded.append(
                            PoseHypothesis(
                                rotate_pose(seed.pose, rotvec, side="left"),
                                f"{seed.source}|ref_R{axis}{sign:+.0f}",
                                seed.measurement_score,
                                seed.obj_id,
                            )
                        )
            ranked = self._score(expanded, frame, detection, motion_reference_pose)
            beam = ranked[: max(beam_width, 1)]

        # Re-render final beam so visualization can reuse its exact alignment.
        final = [
            self.scorer.evaluate(
                item.hypothesis,
                frame,
                detection,
                motion_reference_pose,
                keep_render=True,
            )
            for item in beam
        ]
        return sorted(final, key=lambda item: item.score.total_error)
