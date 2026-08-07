"""Adaptive fixed-lag orientation guard for transformer outputs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from tracking.geometry import closest_rotation, rotation_error_deg, so3_exp, so3_log


@dataclass
class GuardDecision:
    pose: np.ndarray
    temporal_fallback: bool
    flip_confirmed: bool
    required_confirmation: int
    rotation_jump_deg: float
    stable_frames: int


class AdaptiveFixedLagOrientationGuard:
    """Use future window evidence to accept or reject a large orientation jump."""

    def __init__(
        self,
        *,
        soft_start_deg: float = 45.0,
        hard_limit_deg: float = 90.0,
        flip_min_deg: float = 135.0,
        provisional_confirmation: int = 2,
        stable_after_frames: int = 4,
        stable_confirmation: int = 5,
        consistency_deg: float = 45.0,
        minimum_orientation_probability: float = 0.5,
        maximum_prediction_step_deg: float = 30.0,
        velocity_window: int = 5,
    ):
        self.soft_start_deg = float(soft_start_deg)
        self.hard_limit_deg = float(hard_limit_deg)
        self.flip_min_deg = float(flip_min_deg)
        self.provisional_confirmation = int(provisional_confirmation)
        self.stable_after_frames = int(stable_after_frames)
        self.stable_confirmation = int(stable_confirmation)
        self.consistency_deg = float(consistency_deg)
        self.minimum_orientation_probability = float(minimum_orientation_probability)
        self.maximum_prediction_step_deg = float(maximum_prediction_step_deg)
        self.velocity_window = int(velocity_window)
        self.reset()

    def reset(self) -> None:
        self.history: deque[tuple[float | None, np.ndarray]] = deque(
            maxlen=self.velocity_window
        )
        self.stable_frames = 0
        self.flip_state = 0

    def _predict(self, time_s: float | None) -> np.ndarray | None:
        if not self.history:
            return None
        current_time, current = self.history[-1]
        if len(self.history) < 2 or time_s is None or current_time is None:
            return current.copy()
        velocities = []
        history = list(self.history)
        for (ta, ra), (tb, rb) in zip(history, history[1:]):
            if ta is None or tb is None or tb - ta <= 1e-5:
                continue
            velocities.append(so3_log(rb @ ra.T) / (tb - ta))
        dt = time_s - current_time
        if not velocities or dt <= 0:
            return current.copy()
        step = np.median(np.stack(velocities), axis=0) * dt
        maximum = math.radians(self.maximum_prediction_step_deg)
        norm = float(np.linalg.norm(step))
        if norm > maximum:
            step *= maximum / max(norm, 1e-9)
        return closest_rotation(so3_exp(step) @ current)

    @staticmethod
    def _pose_rotation(pose: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        output = np.asarray(pose, dtype=float).copy()
        output[:3, :3] = closest_rotation(rotation)
        return output

    def resolve(
        self,
        pose: np.ndarray,
        *,
        time_s: float | None,
        future_poses: list[np.ndarray],
        future_orientation_probabilities: list[float],
    ) -> GuardDecision:
        pose = np.asarray(pose, dtype=float).copy()
        predicted = self._predict(time_s)
        if predicted is None:
            self.history.append((time_s, pose[:3, :3].copy()))
            self.stable_frames = 1
            return GuardDecision(pose, False, False, self.provisional_confirmation, 0.0, 1)

        predicted_pose = self._pose_rotation(pose, predicted)
        jump = rotation_error_deg(predicted_pose, pose)
        required = (
            self.stable_confirmation
            if self.stable_frames >= self.stable_after_frames
            else self.provisional_confirmation
        )
        confirmed = False
        fallback = False
        if jump >= self.hard_limit_deg:
            evidence = list(zip(future_poses, future_orientation_probabilities))[:required]
            consistent = len(evidence) == required and jump >= self.flip_min_deg
            previous = None
            for future_pose, probability in evidence:
                if probability < self.minimum_orientation_probability:
                    consistent = False
                    break
                if previous is not None and rotation_error_deg(previous, future_pose) > self.consistency_deg:
                    consistent = False
                    break
                previous = future_pose
            confirmed = bool(consistent)
            if confirmed:
                self.flip_state = 1 - self.flip_state
                self.history.clear()
                self.stable_frames = 1
            else:
                pose[:3, :3] = predicted
                fallback = True
        elif jump <= self.soft_start_deg:
            self.stable_frames += 1
        else:
            self.stable_frames = max(1, self.stable_frames - 1)

        self.history.append((time_s, pose[:3, :3].copy()))
        return GuardDecision(
            pose, fallback, confirmed, required, float(jump), self.stable_frames
        )
