"""Binary normal/flipped fixed-lag orientation optimization."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from tracking.geometry import closest_rotation, rotation_error_deg


@dataclass(frozen=True)
class BootstrapConfig:
    window_length: int = 5
    motion_weight: float = 1.0
    motion_scale_deg: float = 20.0
    switch_penalty: float = 2.0

    def validate(self) -> None:
        if self.window_length < 1:
            raise ValueError("window_length must be positive")
        if self.motion_scale_deg <= 0:
            raise ValueError("motion_scale_deg must be positive")
        if self.motion_weight < 0 or self.switch_penalty < 0:
            raise ValueError("optimizer weights must be non-negative")


def polarity_rotations(rotation: np.ndarray, flip_rotation: np.ndarray) -> np.ndarray:
    """Return the input orientation and its object-frame 180-degree alternative."""

    value = closest_rotation(np.asarray(rotation, dtype=float).reshape(3, 3))
    flip = closest_rotation(np.asarray(flip_rotation, dtype=float).reshape(3, 3))
    return np.stack([value, closest_rotation(value @ flip)])


def _path_cost(
    rotations: np.ndarray,
    unary: np.ndarray,
    states: tuple[int, ...],
    config: BootstrapConfig,
    previous_rotation: np.ndarray | None,
    previous_state: int | None,
) -> float:
    total = 0.0
    prior_rotation = previous_rotation
    prior_state = previous_state
    for local_index, state in enumerate(states):
        current = rotations[local_index, state]
        total += float(unary[local_index, state])
        if prior_rotation is not None:
            angle = rotation_error_deg(current, prior_rotation)
            total += config.motion_weight * (angle / config.motion_scale_deg) ** 2
        if prior_state is not None and state != prior_state:
            total += config.switch_penalty
        prior_rotation = current
        prior_state = state
    return total


def optimize_polarity_fixed_lag(
    rotations: np.ndarray,
    unary_costs: np.ndarray,
    config: BootstrapConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose normal/flipped states using a fixed-lag exhaustive Viterbi window.

    The first decision is made after seeing ``window_length`` frames. Each later
    decision uses the same amount of look-ahead where available. This function is
    deliberately offline: callers can backfill the buffered frames before writing
    the output CSV.
    """

    config.validate()
    alternatives = np.asarray(rotations, dtype=float)
    unary = np.asarray(unary_costs, dtype=float)
    if alternatives.ndim != 4 or alternatives.shape[1:] != (2, 3, 3):
        raise ValueError("rotations must have shape [frames, 2, 3, 3]")
    if unary.shape != alternatives.shape[:2]:
        raise ValueError("unary_costs must have shape [frames, 2]")
    if not np.isfinite(alternatives).all() or not np.isfinite(unary).all():
        raise ValueError("optimizer inputs must be finite")

    count = len(alternatives)
    states = np.zeros(count, dtype=np.int64)
    costs = np.zeros(count, dtype=float)
    previous_rotation = None
    previous_state = None
    for start in range(count):
        stop = min(count, start + config.window_length)
        length = stop - start
        best_states: tuple[int, ...] | None = None
        best_cost = np.inf
        for candidate_states in product((0, 1), repeat=length):
            value = _path_cost(
                alternatives[start:stop], unary[start:stop], candidate_states,
                config, previous_rotation, previous_state,
            )
            if value < best_cost:
                best_cost = value
                best_states = candidate_states
        assert best_states is not None
        selected = int(best_states[0])
        states[start] = selected
        costs[start] = best_cost
        previous_rotation = alternatives[start, selected]
        previous_state = selected
    return states, costs


def optimize_candidate_fixed_lag(
    poses: np.ndarray,
    unary_costs: np.ndarray,
    valid: np.ndarray,
    polarity: np.ndarray,
    config: BootstrapConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-lag Viterbi selection over a variable set of pose candidates.

    Unlike :func:`optimize_polarity_fixed_lag`, this supports all visual
    candidates (and optional center-preserving flipped copies). Dynamic
    programming keeps the cost practical for 16--32 states per frame.
    """

    config.validate()
    values = np.asarray(poses, dtype=float)
    unary = np.asarray(unary_costs, dtype=float)
    available = np.asarray(valid, dtype=bool)
    modes = np.asarray(polarity, dtype=np.int64).reshape(-1)
    if values.ndim != 4 or values.shape[2:] != (4, 4):
        raise ValueError("poses must have shape [frames, states, 4, 4]")
    if unary.shape != values.shape[:2] or available.shape != values.shape[:2]:
        raise ValueError("unary_costs and valid must have shape [frames, states]")
    if modes.shape != (values.shape[1],):
        raise ValueError("polarity must have one entry per state")
    if not np.all(np.any(available, axis=1)):
        raise ValueError("Every frame must have at least one valid state")

    frame_count, state_count = unary.shape
    selected = np.zeros(frame_count, dtype=np.int64)
    path_costs = np.zeros(frame_count, dtype=float)
    previous_pose = None
    previous_polarity = None
    for start in range(frame_count):
        stop = min(frame_count, start + config.window_length)
        window = stop - start
        backpointers = np.full((window, state_count), -1, dtype=np.int64)
        costs = np.full(state_count, np.inf)
        for state in np.flatnonzero(available[start]):
            value = float(unary[start, state])
            if previous_pose is not None:
                angle = rotation_error_deg(values[start, state], previous_pose)
                value += config.motion_weight * (angle / config.motion_scale_deg) ** 2
            if previous_polarity is not None and modes[state] != previous_polarity:
                value += config.switch_penalty
            costs[state] = value

        for offset in range(1, window):
            frame = start + offset
            following = np.full(state_count, np.inf)
            previous_states = np.flatnonzero(np.isfinite(costs))
            for state in np.flatnonzero(available[frame]):
                pair = np.empty(len(previous_states), dtype=float)
                for pair_index, prior_state in enumerate(previous_states):
                    angle = rotation_error_deg(
                        values[frame, state], values[frame - 1, prior_state]
                    )
                    pair[pair_index] = (
                        costs[prior_state]
                        + config.motion_weight * (angle / config.motion_scale_deg) ** 2
                        + config.switch_penalty * float(modes[state] != modes[prior_state])
                    )
                winner = int(np.argmin(pair))
                following[state] = float(unary[frame, state]) + pair[winner]
                backpointers[offset, state] = int(previous_states[winner])
            costs = following

        end_state = int(np.argmin(costs))
        path_costs[start] = float(costs[end_state])
        path = [end_state]
        for offset in range(window - 1, 0, -1):
            path.append(int(backpointers[offset, path[-1]]))
        committed = int(path[-1])
        selected[start] = committed
        previous_pose = values[start, committed]
        previous_polarity = int(modes[committed])
    return selected, path_costs
