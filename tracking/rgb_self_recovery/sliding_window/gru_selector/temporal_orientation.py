"""Causal temporal-orientation candidates, reranking, and flip hysteresis."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from tracking.geometry import closest_rotation, flipped_pose, rotation_error_deg, so3_exp, so3_log


@dataclass
class OrientationDecision:
    pose: np.ndarray
    source_suffix: str
    candidate_index: int
    predicted_rotation: np.ndarray
    rotation_from_prediction_deg: float
    temporal_reranked: bool
    temporal_fallback: bool
    flip_like: bool
    flip_confirmed: bool
    flip_pending_frames: int
    flip_state: int
    orientation_stable_frames: int
    orientation_state_locked: bool
    required_flip_confirmation_frames: int
    best_nonflip_probability: float
    temporal_candidate_names: tuple[str, ...]


class FixedLagOrientationResolver:
    """Track unresolved flip frames until hysteresis accepts or rejects them.

    The causal selector continues to use its safe temporal fallback internally.
    Because ``select.py`` writes its files only after replaying the sequence, the
    returned frame indices can be revised before any output is committed.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.pending_indices: list[int] = []

    def update(self, frame_index: int, decision: OrientationDecision) -> list[int]:
        """Return earlier pending frames that a confirmed flip should backfill."""
        if decision.flip_confirmed:
            resolved = self.pending_indices.copy()
            self.pending_indices.clear()
            return resolved
        if decision.temporal_fallback and decision.flip_pending_frames > 0:
            # A value of one means either a new burst or an inconsistent flip
            # hypothesis that restarted confirmation. Older pending frames must
            # remain in the established orientation mode in either case.
            if decision.flip_pending_frames == 1:
                self.pending_indices.clear()
            self.pending_indices.append(int(frame_index))
        else:
            # Ordinary same-mode evidence cancels an unresolved transition.
            self.pending_indices.clear()
        return []


class TemporalOrientationSelector:
    """Maintain accepted orientation and reject unsupported front/rear switches.

    Translation always comes from the GRU-selected visual candidate. Only its
    orientation can be replaced by a causal temporal candidate. No ground truth
    enters this state.
    """

    def __init__(
        self,
        *,
        soft_start_deg: float = 45.0,
        hard_limit_deg: float = 90.0,
        flip_min_deg: float = 135.0,
        rotation_penalty_weight: float = 1.0,
        flip_confirmation_frames: int = 2,
        stable_orientation_frames: int = 4,
        stable_flip_confirmation_frames: int = 5,
        flip_min_probability: float = 0.15,
        flip_min_margin: float = 0.03,
        flip_consistency_deg: float = 45.0,
        maximum_prediction_step_deg: float = 30.0,
        velocity_window: int = 5,
        temporal_offsets_deg: tuple[float, ...] = (10.0, 20.0),
    ):
        if not 0 <= soft_start_deg < hard_limit_deg < flip_min_deg <= 180:
            raise ValueError("Require 0 <= soft start < hard limit < flip minimum <= 180")
        if flip_confirmation_frames < 1 or velocity_window < 2:
            raise ValueError("Flip confirmation and velocity window must be positive")
        if stable_orientation_frames < 1:
            raise ValueError("Stable orientation frames must be positive")
        if stable_flip_confirmation_frames < flip_confirmation_frames:
            raise ValueError(
                "Stable flip confirmation cannot be shorter than provisional confirmation"
            )
        self.soft_start_deg = float(soft_start_deg)
        self.hard_limit_deg = float(hard_limit_deg)
        self.flip_min_deg = float(flip_min_deg)
        self.rotation_penalty_weight = float(rotation_penalty_weight)
        self.flip_confirmation_frames = int(flip_confirmation_frames)
        self.stable_orientation_frames = int(stable_orientation_frames)
        self.stable_flip_confirmation_frames = int(stable_flip_confirmation_frames)
        self.flip_min_probability = float(flip_min_probability)
        self.flip_min_margin = float(flip_min_margin)
        self.flip_consistency_deg = float(flip_consistency_deg)
        self.maximum_prediction_step_deg = float(maximum_prediction_step_deg)
        self.velocity_window = int(velocity_window)
        self.temporal_offsets_deg = tuple(float(value) for value in temporal_offsets_deg)
        self.reset()

    def reset(self) -> None:
        self.history: deque[tuple[float | None, np.ndarray]] = deque(
            maxlen=self.velocity_window
        )
        self.pending_rotation: np.ndarray | None = None
        self.pending_frames = 0
        self.flip_state = 0
        self.orientation_stable_frames = 0

    def _predict_rotation(self, time_s: float | None) -> np.ndarray | None:
        if not self.history:
            return None
        current_time, current = self.history[-1]
        if len(self.history) < 2 or time_s is None or current_time is None:
            return current.copy()
        velocities = []
        history = list(self.history)
        for (previous_time, previous), (next_time, next_rotation) in zip(
            history, history[1:]
        ):
            if previous_time is None or next_time is None:
                continue
            dt = next_time - previous_time
            if dt <= 1e-5:
                continue
            velocities.append(so3_log(next_rotation @ previous.T) / dt)
        dt_next = time_s - current_time
        if not velocities or dt_next <= 0:
            return current.copy()
        angular_velocity = np.median(np.stack(velocities), axis=0)
        step = angular_velocity * dt_next
        maximum = math.radians(self.maximum_prediction_step_deg)
        norm = float(np.linalg.norm(step))
        if norm > maximum:
            step *= maximum / max(norm, 1e-9)
        return closest_rotation(so3_exp(step) @ current)

    @staticmethod
    def _pose_with_rotation(pose: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        output = np.asarray(pose, dtype=float).copy()
        output[:3, :3] = closest_rotation(rotation)
        return output

    def _temporal_candidates(
        self, selected_pose: np.ndarray, predicted_rotation: np.ndarray
    ) -> dict[str, np.ndarray]:
        base = self._pose_with_rotation(selected_pose, predicted_rotation)
        output = {"constant_angular_velocity": base}
        if self.history:
            output["previous_accepted"] = self._pose_with_rotation(
                selected_pose, self.history[-1][1]
            )
        innovation = so3_log(selected_pose[:3, :3] @ predicted_rotation.T)
        magnitude = float(np.linalg.norm(innovation))
        if magnitude > 1e-8:
            axis = innovation / magnitude
            for degrees in self.temporal_offsets_deg:
                for sign in (-1.0, 1.0):
                    rotation = so3_exp(axis * math.radians(sign * degrees)) @ predicted_rotation
                    output[f"temporal_{sign * degrees:+g}deg"] = self._pose_with_rotation(
                        selected_pose, rotation
                    )
        output["temporal_flip180"] = self._pose_with_rotation(
            selected_pose, flipped_pose(base, "z")[:3, :3]
        )
        return output

    def rerank(
        self,
        poses: np.ndarray,
        probabilities: np.ndarray,
        valid: np.ndarray,
        time_s: float | None,
    ) -> tuple[int, np.ndarray | None, np.ndarray, bool]:
        raw = int(np.argmax(np.where(valid, probabilities, -np.inf)))
        predicted = self._predict_rotation(time_s)
        if predicted is None:
            return raw, None, np.full(len(poses), np.nan), False
        angles = np.asarray(
            [
                rotation_error_deg(
                    self._pose_with_rotation(poses[index], predicted), poses[index]
                )
                if valid[index] else np.inf
                for index in range(len(poses))
            ],
            dtype=float,
        )
        # A flip-like visual winner must reach the hysteresis state machine.
        # Soft reranking it away here would make a wrong orientation accepted
        # at sequence bootstrap impossible to correct later.
        if angles[raw] >= self.flip_min_deg:
            return raw, predicted, angles, False
        span = max(self.hard_limit_deg - self.soft_start_deg, 1e-6)
        excess = np.maximum(angles - self.soft_start_deg, 0.0) / span
        adjusted = np.log(np.clip(probabilities, 1e-12, 1.0))
        adjusted -= self.rotation_penalty_weight * np.square(excess)
        adjusted[~valid] = -np.inf
        selected = int(np.argmax(adjusted))
        return selected, predicted, angles, selected != raw

    def enforce(
        self,
        selected_pose: np.ndarray,
        selected_index: int,
        probabilities: np.ndarray,
        valid: np.ndarray,
        predicted_rotation: np.ndarray | None,
        angles_deg: np.ndarray,
        time_s: float | None,
    ) -> OrientationDecision:
        if predicted_rotation is None:
            accepted = np.asarray(selected_pose, dtype=float).copy()
            self.history.append((time_s, accepted[:3, :3].copy()))
            self.orientation_stable_frames = 1
            return OrientationDecision(
                accepted, "", selected_index, accepted[:3, :3].copy(), 0.0,
                False, False, False, False, 0, self.flip_state,
                self.orientation_stable_frames, False,
                self.flip_confirmation_frames, 0.0, (),
            )

        angle = float(angles_deg[selected_index])
        candidates = self._temporal_candidates(selected_pose, predicted_rotation)
        nonflip = valid & (angles_deg < self.hard_limit_deg)
        best_nonflip_probability = float(
            np.max(probabilities[nonflip]) if np.any(nonflip) else 0.0
        )
        flip_like = angle >= self.flip_min_deg
        strong_flip_evidence = (
            flip_like
            and probabilities[selected_index] >= self.flip_min_probability
            and probabilities[selected_index] - best_nonflip_probability
            >= self.flip_min_margin
        )
        if strong_flip_evidence:
            consistent = (
                self.pending_rotation is not None
                and rotation_error_deg(
                    self._pose_with_rotation(selected_pose, self.pending_rotation),
                    selected_pose,
                ) <= self.flip_consistency_deg
            )
            self.pending_frames = self.pending_frames + 1 if consistent else 1
            self.pending_rotation = selected_pose[:3, :3].copy()
        else:
            self.pending_frames = 0
            self.pending_rotation = None

        orientation_state_locked_before_update = (
            self.orientation_stable_frames >= self.stable_orientation_frames
        )
        required_confirmation = (
            self.stable_flip_confirmation_frames
            if orientation_state_locked_before_update
            else self.flip_confirmation_frames
        )
        flip_confirmed = (
            strong_flip_evidence
            and self.pending_frames >= required_confirmation
        )
        temporal_fallback = angle >= self.hard_limit_deg and not flip_confirmed
        if temporal_fallback:
            accepted = candidates["constant_angular_velocity"].copy()
            suffix = "|temporal_orientation_fallback"
            output_index = -1
        else:
            accepted = np.asarray(selected_pose, dtype=float).copy()
            suffix = "|flip_transition_confirmed" if flip_confirmed else ""
            output_index = selected_index
            if flip_confirmed:
                self.flip_state = 1 - self.flip_state
                self.pending_frames = 0
                self.pending_rotation = None
                self.history.clear()

        if flip_confirmed:
            self.orientation_stable_frames = 1
        elif temporal_fallback:
            # Holding the established state adds no new visual evidence, but
            # it must not erase confidence accumulated before a short burst.
            pass
        elif angle <= self.soft_start_deg:
            self.orientation_stable_frames += 1
        else:
            self.orientation_stable_frames = max(
                self.orientation_stable_frames - 1, 1
            )
        orientation_state_locked = (
            self.orientation_stable_frames >= self.stable_orientation_frames
        )

        self.history.append((time_s, accepted[:3, :3].copy()))
        return OrientationDecision(
            pose=accepted,
            source_suffix=suffix,
            candidate_index=output_index,
            predicted_rotation=predicted_rotation,
            rotation_from_prediction_deg=angle,
            temporal_reranked=False,
            temporal_fallback=temporal_fallback,
            flip_like=flip_like,
            flip_confirmed=flip_confirmed,
            flip_pending_frames=self.pending_frames,
            flip_state=self.flip_state,
            orientation_stable_frames=self.orientation_stable_frames,
            orientation_state_locked=orientation_state_locked,
            required_flip_confirmation_frames=required_confirmation,
            best_nonflip_probability=best_nonflip_probability,
            temporal_candidate_names=tuple(candidates),
        )
