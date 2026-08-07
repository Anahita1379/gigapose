"""Ground-truth-free fixed-lag translation outlier detection and repair."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SOURCE_NAMES = ("gru_kalman", "selector_measurement", "gigapose", "motion_regression")


@dataclass(frozen=True)
class FixedLagConfig:
    window_size: int = 5
    lag: int = 2
    minimum_gate_m: float = 1.5
    depth_gate_fraction: float = 0.02
    mad_multiplier: float = 3.5
    maximum_neighbor_fit_residual_m: float = 4.0
    minimum_candidate_improvement_m: float = 1.0
    candidate_acceptance_gate_multiplier: float = 1.5
    selector_penalty_m: float = 0.15
    gigapose_penalty_m: float = 0.25
    allow_motion_only_repair: bool = False
    motion_only_penalty_m: float = 1.0
    motion_blend: float = 0.10
    maximum_repair_m: float = 30.0
    acceleration_regularization: float = 0.05
    irls_iterations: int = 4

    def validate(self) -> None:
        if self.window_size < 3 or self.window_size % 2 != 1:
            raise ValueError("window_size must be an odd integer of at least 3")
        if self.lag != self.window_size // 2:
            raise ValueError("lag must equal window_size // 2 for a centered fixed-lag window")
        if min(
            self.minimum_gate_m,
            self.depth_gate_fraction,
            self.mad_multiplier,
            self.maximum_neighbor_fit_residual_m,
            self.minimum_candidate_improvement_m,
            self.candidate_acceptance_gate_multiplier,
            self.maximum_repair_m,
            self.acceleration_regularization,
        ) < 0:
            raise ValueError("fixed-lag distances, scales, and regularization must be nonnegative")
        if not 0 <= self.motion_blend <= 1:
            raise ValueError("motion_blend must be in [0, 1]")
        if self.irls_iterations < 1:
            raise ValueError("irls_iterations must be positive")


@dataclass
class FixedLagResult:
    positions: np.ndarray
    repair_mask: np.ndarray
    selected_source: np.ndarray
    motion_prediction: np.ndarray
    estimated_velocity: np.ndarray
    estimated_acceleration: np.ndarray
    causal_residual_m: np.ndarray
    selected_residual_m: np.ndarray
    adaptive_gate_m: np.ndarray
    neighbor_fit_residual_m: np.ndarray
    repair_magnitude_m: np.ndarray
    replay_start_local_index: np.ndarray


def _robust_motion_regression(
    relative_time: np.ndarray,
    positions: np.ndarray,
    acceleration_regularization: float,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit p(t)=p0+v*t+0.5*a*t^2 with vector-valued Huber IRLS."""
    design = np.column_stack((
        np.ones(len(relative_time)), relative_time, 0.5 * relative_time**2,
    ))
    weights = np.ones(len(relative_time), dtype=float)
    coefficients = np.zeros((3, 3), dtype=float)
    regularizer = np.diag((1e-8, 1e-8, acceleration_regularization))
    for _ in range(iterations):
        weighted = design * weights[:, None]
        coefficients = np.linalg.solve(
            design.T @ weighted + regularizer,
            design.T @ (positions * weights[:, None]),
        )
        residual_vectors = positions - design @ coefficients
        residuals = np.linalg.norm(residual_vectors, axis=1)
        median = float(np.median(residuals))
        scale = max(1.4826 * float(np.median(np.abs(residuals - median))), 0.05)
        normalized = residuals / (1.345 * scale)
        weights = np.where(normalized <= 1.0, 1.0, 1.0 / normalized)
    residuals = np.linalg.norm(positions - design @ coefficients, axis=1)
    return coefficients[0], coefficients[1], coefficients[2], residuals


def smooth_segment(
    causal_positions: np.ndarray,
    measurement_positions: np.ndarray,
    gigapose_positions: np.ndarray,
    time_s: np.ndarray,
    config: FixedLagConfig,
) -> FixedLagResult:
    """Repair centered frames using only their configured fixed-lag window."""
    config.validate()
    causal = np.asarray(causal_positions, dtype=float)
    measurement = np.asarray(measurement_positions, dtype=float)
    gigapose = np.asarray(gigapose_positions, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    if causal.ndim != 2 or causal.shape[1] != 3:
        raise ValueError("causal_positions must have shape [frames, 3]")
    if measurement.shape != causal.shape or gigapose.shape != causal.shape:
        raise ValueError("all candidate position arrays must have the same shape")
    if time_s.shape != (len(causal),):
        raise ValueError("time_s must have one value per frame")
    if len(time_s) > 1 and np.any(np.diff(time_s) <= 0):
        raise ValueError("time_s must be strictly increasing inside a segment")

    count = len(causal)
    output = causal.copy()
    repair = np.zeros(count, dtype=bool)
    selected_source = np.full(count, "gru_kalman", dtype="<U32")
    prediction = np.full_like(causal, np.nan)
    velocity = np.full_like(causal, np.nan)
    acceleration = np.full_like(causal, np.nan)
    causal_residual = np.full(count, np.nan)
    selected_residual = np.full(count, np.nan)
    gate = np.full(count, np.nan)
    fit_residual = np.full(count, np.nan)
    repair_magnitude = np.zeros(count, dtype=float)
    replay_start = np.full(count, -1, dtype=np.int64)

    radius = config.window_size // 2
    for target in range(radius, count - radius):
        window = np.arange(target - radius, target + radius + 1)
        neighbors = window[window != target]
        relative_time = time_s[neighbors] - time_s[target]
        try:
            predicted, estimated_v, estimated_a, neighbor_residuals = _robust_motion_regression(
                relative_time,
                causal[neighbors],
                config.acceleration_regularization,
                config.irls_iterations,
            )
        except np.linalg.LinAlgError:
            continue
        prediction[target] = predicted
        velocity[target] = estimated_v
        acceleration[target] = estimated_a
        fit_median = float(np.median(neighbor_residuals))
        fit_residual[target] = fit_median
        residual_median = float(np.median(neighbor_residuals))
        residual_mad = float(np.median(np.abs(neighbor_residuals - residual_median)))
        depth_gate = config.depth_gate_fraction * max(abs(predicted[2]), 0.0)
        adaptive_gate = max(
            config.minimum_gate_m,
            depth_gate,
            residual_median + config.mad_multiplier * 1.4826 * residual_mad,
        )
        gate[target] = adaptive_gate
        candidate_positions = [causal[target], measurement[target], gigapose[target]]
        candidate_penalties = [0.0, config.selector_penalty_m, config.gigapose_penalty_m]
        candidate_names = list(SOURCE_NAMES[:3])
        if config.allow_motion_only_repair:
            candidate_positions.append(predicted)
            candidate_penalties.append(config.motion_only_penalty_m)
            candidate_names.append(SOURCE_NAMES[3])
        residuals = np.asarray([
            np.linalg.norm(value - predicted) for value in candidate_positions
        ])
        causal_residual[target] = residuals[0]
        if (
            not np.isfinite(residuals).all()
            or fit_median > config.maximum_neighbor_fit_residual_m
            or residuals[0] <= adaptive_gate
        ):
            selected_residual[target] = residuals[0]
            continue
        scores = residuals + np.asarray(candidate_penalties)
        best = int(np.argmin(scores))
        improvement = residuals[0] - residuals[best]
        if (
            best == 0
            or improvement < config.minimum_candidate_improvement_m
            or residuals[best] > adaptive_gate * config.candidate_acceptance_gate_multiplier
        ):
            selected_residual[target] = residuals[0]
            continue
        anchor = candidate_positions[best]
        corrected = (1.0 - config.motion_blend) * anchor + config.motion_blend * predicted
        magnitude = float(np.linalg.norm(corrected - causal[target]))
        if magnitude > config.maximum_repair_m:
            corrected = causal[target] + (
                config.maximum_repair_m / max(magnitude, 1e-9)
            ) * (corrected - causal[target])
            magnitude = config.maximum_repair_m
        output[target] = corrected
        repair[target] = True
        selected_source[target] = candidate_names[best]
        selected_residual[target] = float(np.linalg.norm(corrected - predicted))
        repair_magnitude[target] = magnitude
        replay_start[target] = target - radius

    return FixedLagResult(
        positions=output,
        repair_mask=repair,
        selected_source=selected_source,
        motion_prediction=prediction,
        estimated_velocity=velocity,
        estimated_acceleration=acceleration,
        causal_residual_m=causal_residual,
        selected_residual_m=selected_residual,
        adaptive_gate_m=gate,
        neighbor_fit_residual_m=fit_residual,
        repair_magnitude_m=repair_magnitude,
        replay_start_local_index=replay_start,
    )
