"""Sparse robust SE(3) sequence optimization with hard max-mixture assignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from tracking.geometry import closest_rotation, so3_exp, so3_log


@dataclass(frozen=True)
class FactorGraphConfig:
    measurement_translation_sigma_m: float = 2.0
    measurement_rotation_sigma_deg: float = 20.0
    prior_translation_sigma_m: float = 5.0
    prior_rotation_sigma_deg: float = 45.0
    acceleration_sigma_mps2: float = 20.0
    angular_acceleration_sigma_radps2: float = 3.0
    jerk_sigma_mps3: float = 100.0
    visual_assignment_weight: float = 0.5
    assignment_current_weight: float = 0.25
    assignment_motion_weight: float = 1.5
    assignment_translation_scale_m: float = 3.0
    assignment_rotation_scale_deg: float = 30.0
    outer_iterations: int = 3
    maximum_nfev: int = 100
    maximum_translation_step_m: float = 20.0
    maximum_rotation_step_deg: float = 90.0
    robust_loss: str = "huber"
    robust_scale: float = 1.0
    solver_tolerance: float = 1e-4
    finite_difference_step: float = 1e-4

    def validate(self) -> None:
        positive = (
            self.measurement_translation_sigma_m,
            self.measurement_rotation_sigma_deg,
            self.prior_translation_sigma_m,
            self.prior_rotation_sigma_deg,
            self.acceleration_sigma_mps2,
            self.angular_acceleration_sigma_radps2,
            self.jerk_sigma_mps3,
            self.assignment_translation_scale_m,
            self.assignment_rotation_scale_deg,
            self.maximum_translation_step_m,
            self.maximum_rotation_step_deg,
            self.robust_scale,
            self.solver_tolerance,
            self.finite_difference_step,
        )
        if min(positive) <= 0:
            raise ValueError("Factor-graph scales and bounds must be positive")
        if min(
            self.visual_assignment_weight,
            self.assignment_current_weight,
            self.assignment_motion_weight,
        ) < 0:
            raise ValueError("Candidate-assignment weights must be nonnegative")
        if self.outer_iterations < 1 or self.maximum_nfev < 1:
            raise ValueError("Optimization iteration limits must be positive")
        if self.robust_loss not in {"linear", "soft_l1", "huber", "cauchy", "arctan"}:
            raise ValueError(f"Unsupported robust loss: {self.robust_loss}")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class FactorGraphResult:
    poses: np.ndarray
    selected_candidate: np.ndarray
    selected_visual_cost: np.ndarray
    initial_objective: float
    final_objective: float
    outer_iterations_completed: int
    optimizer_success: bool
    optimizer_message: str
    optimizer_nfev: int
    assignment_changes: int
    solution_accepted: bool


def normalized_visual_cost(total_error: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Normalize per-frame RGB/CAD costs to [0, 1] without GT information."""
    values = np.asarray(total_error, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    output = np.full(values.shape, np.inf, dtype=float)
    for frame in range(len(values)):
        indices = np.flatnonzero(valid[frame] & np.isfinite(values[frame]))
        if not len(indices):
            raise ValueError(f"Frame {frame} has no valid finite candidate cost")
        selected = values[frame, indices]
        low, high = float(selected.min()), float(selected.max())
        if high - low < 1e-9:
            output[frame, indices] = 0.0
        else:
            output[frame, indices] = (selected - low) / (high - low)
    return output


def _pose_distance(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    translation = float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
    rotation = float(np.degrees(np.linalg.norm(
        so3_log(first[:3, :3].T @ second[:3, :3])
    )))
    return translation, rotation


def assign_candidates(
    current_poses: np.ndarray,
    candidate_poses: np.ndarray,
    valid: np.ndarray,
    visual_cost: np.ndarray,
    time_s: np.ndarray,
    config: FactorGraphConfig,
) -> np.ndarray:
    """Hard max-mixture assignment using pose consistency plus RGB/CAD cost."""
    count = len(current_poses)
    selected = np.zeros(count, dtype=np.int64)
    for frame in range(count):
        motion_pose = None
        if 0 < frame < count - 1:
            span = max(float(time_s[frame + 1] - time_s[frame - 1]), 1e-6)
            alpha = float((time_s[frame] - time_s[frame - 1]) / span)
            motion_pose = np.eye(4)
            motion_pose[:3, 3] = (
                (1.0 - alpha) * current_poses[frame - 1, :3, 3]
                + alpha * current_poses[frame + 1, :3, 3]
            )
            relative = (
                current_poses[frame - 1, :3, :3].T
                @ current_poses[frame + 1, :3, :3]
            )
            motion_pose[:3, :3] = closest_rotation(
                current_poses[frame - 1, :3, :3]
                @ so3_exp(alpha * so3_log(relative))
            )
        best_cost, best_index = np.inf, -1
        for index in np.flatnonzero(valid[frame]):
            translation, rotation = _pose_distance(
                current_poses[frame], candidate_poses[frame, index]
            )
            cost = config.assignment_current_weight * (
                translation / config.assignment_translation_scale_m
                + rotation / config.assignment_rotation_scale_deg
            )
            if motion_pose is not None:
                motion_translation, motion_rotation = _pose_distance(
                    motion_pose, candidate_poses[frame, index]
                )
                cost += config.assignment_motion_weight * (
                    motion_translation / config.assignment_translation_scale_m
                    + motion_rotation / config.assignment_rotation_scale_deg
                )
            cost += (
                + config.visual_assignment_weight * visual_cost[frame, index]
            )
            if cost < best_cost:
                best_cost, best_index = cost, int(index)
        if best_index < 0:
            raise ValueError(f"Frame {frame} has no valid candidate")
        selected[frame] = best_index
    return selected


def visual_initialization(
    candidate_poses: np.ndarray,
    valid: np.ndarray,
    visual_cost: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray([
        int(indices[np.argmin(visual_cost[frame, indices])])
        for frame in range(len(candidate_poses))
        for indices in [np.flatnonzero(valid[frame])]
    ], dtype=np.int64)
    return candidate_poses[np.arange(len(selected)), selected].copy(), selected


def _apply_delta(reference: np.ndarray, delta: np.ndarray) -> np.ndarray:
    output = np.asarray(reference, dtype=float).copy()
    values = np.asarray(delta, dtype=float).reshape(len(reference), 6)
    output[:, :3, 3] += values[:, :3]
    for index in range(len(output)):
        output[index, :3, :3] = closest_rotation(
            reference[index, :3, :3] @ so3_exp(values[index, 3:])
        )
    return output


def _kinematics(poses: np.ndarray, time_s: np.ndarray):
    dt = np.diff(time_s).clip(0.02, 1.0)
    velocity = np.diff(poses[:, :3, 3], axis=0) / dt[:, None]
    angular_velocity = np.asarray([
        so3_log(poses[index, :3, :3].T @ poses[index + 1, :3, :3]) / dt[index]
        for index in range(len(poses) - 1)
    ])
    if len(poses) >= 3:
        acceleration_dt = 0.5 * (dt[1:] + dt[:-1])
        acceleration = np.diff(velocity, axis=0) / acceleration_dt[:, None]
        angular_acceleration = np.diff(angular_velocity, axis=0) / acceleration_dt[:, None]
    else:
        acceleration_dt = np.empty(0)
        acceleration = np.empty((0, 3))
        angular_acceleration = np.empty((0, 3))
    if len(poses) >= 4:
        jerk_dt = 0.5 * (acceleration_dt[1:] + acceleration_dt[:-1])
        jerk = np.diff(acceleration, axis=0) / jerk_dt[:, None]
    else:
        jerk = np.empty((0, 3))
    return acceleration, angular_acceleration, jerk


def _residuals(
    delta: np.ndarray,
    reference: np.ndarray,
    measurement: np.ndarray,
    time_s: np.ndarray,
    config: FactorGraphConfig,
    prior: np.ndarray | None,
) -> np.ndarray:
    poses = _apply_delta(reference, delta)
    residuals = []
    rotation_sigma = np.deg2rad(config.measurement_rotation_sigma_deg)
    for pose, observed in zip(poses, measurement):
        residuals.extend((pose[:3, 3] - observed[:3, 3]) / config.measurement_translation_sigma_m)
        residuals.extend(so3_log(observed[:3, :3].T @ pose[:3, :3]) / rotation_sigma)
    if prior is not None:
        prior_rotation_sigma = np.deg2rad(config.prior_rotation_sigma_deg)
        for pose, prior_pose in zip(poses, prior):
            residuals.extend(
                (pose[:3, 3] - prior_pose[:3, 3])
                / config.prior_translation_sigma_m
            )
            residuals.extend(
                so3_log(prior_pose[:3, :3].T @ pose[:3, :3])
                / prior_rotation_sigma
            )
    acceleration, angular_acceleration, jerk = _kinematics(poses, time_s)
    for linear_value, angular_value in zip(acceleration, angular_acceleration):
        residuals.extend(linear_value / config.acceleration_sigma_mps2)
        residuals.extend(
            angular_value / config.angular_acceleration_sigma_radps2
        )
    residuals.extend((jerk / config.jerk_sigma_mps3).reshape(-1))
    return np.asarray(residuals, dtype=float)


def _jacobian_sparsity(count: int, include_prior: bool):
    rows = 6 * count + (6 * count if include_prior else 0)
    rows += 6 * max(count - 2, 0) + 3 * max(count - 3, 0)
    pattern = lil_matrix((rows, 6 * count), dtype=np.int8)
    row = 0
    for frame in range(count):
        pattern[row:row + 6, 6 * frame:6 * frame + 6] = 1
        row += 6
    if include_prior:
        for frame in range(count):
            pattern[row:row + 6, 6 * frame:6 * frame + 6] = 1
            row += 6
    for center in range(1, count - 1):
        pattern[row:row + 6, 6 * (center - 1):6 * (center + 2)] = 1
        row += 6
    for right in range(3, count):
        pattern[row:row + 3, 6 * (right - 3):6 * (right + 1)] = 1
        row += 3
    if row != rows:
        raise RuntimeError("Factor-graph sparsity construction is inconsistent")
    return pattern.tocsr()


def optimize_sequence(
    candidate_poses: np.ndarray,
    valid: np.ndarray,
    visual_cost: np.ndarray,
    time_s: np.ndarray,
    *,
    initial_poses: np.ndarray,
    config: FactorGraphConfig,
    prior_poses: np.ndarray | None = None,
) -> FactorGraphResult:
    """Optimize one contiguous sequence; no ground truth is accepted or used."""
    config.validate()
    candidates = np.asarray(candidate_poses, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    visual_cost = np.asarray(visual_cost, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    reference = np.asarray(initial_poses, dtype=float).copy()
    prior = None if prior_poses is None else np.asarray(prior_poses, dtype=float)
    count = len(reference)
    if candidates.shape[:2] != valid.shape or candidates.shape[0] != count:
        raise ValueError("Candidate, validity, and initialization shapes disagree")
    if visual_cost.shape != valid.shape or time_s.shape != (count,):
        raise ValueError("Visual costs or timestamps have the wrong shape")
    if prior is not None and prior.shape != reference.shape:
        raise ValueError("External prior and initialization shapes differ")
    if count == 0 or (count > 1 and np.any(np.diff(time_s) <= 0)):
        raise ValueError("Sequence timestamps must be nonempty and strictly increasing")

    visual_selected = np.asarray([
        int(indices[np.argmin(visual_cost[frame, indices])])
        for frame in range(count)
        for indices in [np.flatnonzero(valid[frame])]
    ], dtype=np.int64)
    selected = assign_candidates(
        reference, candidates, valid, visual_cost, time_s, config
    )
    first_selected = selected.copy()
    initial_measurement = candidates[np.arange(count), selected]
    initial_residual = _residuals(
        np.zeros(6 * count), reference, initial_measurement, time_s, config, prior
    )
    initial_objective = float(np.mean(initial_residual**2))
    total_nfev, success, message = 0, True, ""
    completed = 0
    sparsity = _jacobian_sparsity(count, prior is not None)
    rotation_bound = np.deg2rad(config.maximum_rotation_step_deg)
    lower = np.tile(
        [-config.maximum_translation_step_m] * 3 + [-rotation_bound] * 3, count
    )
    upper = -lower
    previous = None
    for outer in range(config.outer_iterations):
        measurement = candidates[np.arange(count), selected]
        result = least_squares(
            _residuals,
            np.zeros(6 * count),
            args=(reference, measurement, time_s, config, prior),
            jac_sparsity=sparsity,
            bounds=(lower, upper),
            method="trf",
            loss=config.robust_loss,
            f_scale=config.robust_scale,
            max_nfev=config.maximum_nfev,
            x_scale="jac",
            ftol=config.solver_tolerance,
            xtol=config.solver_tolerance,
            gtol=config.solver_tolerance,
            diff_step=config.finite_difference_step,
        )
        reference = _apply_delta(reference, result.x)
        total_nfev += int(result.nfev)
        success = success and bool(result.success)
        message = str(result.message)
        completed = outer + 1
        previous, selected = selected, assign_candidates(
            reference, candidates, valid, visual_cost, time_s, config
        )
        if np.array_equal(previous, selected) and np.linalg.norm(result.x) < 1e-4:
            break
    final_measurement = candidates[np.arange(count), selected]
    final_residual = _residuals(
        np.zeros(6 * count), reference, final_measurement, time_s, config, prior
    )
    final_objective = float(np.mean(final_residual**2))
    accepted = bool(
        np.isfinite(reference).all()
        and np.isfinite(final_objective)
        and final_objective <= initial_objective * 1.05
    )
    if not accepted:
        reference = np.asarray(initial_poses, dtype=float).copy()
        selected = first_selected
        final_objective = initial_objective
    return FactorGraphResult(
        poses=reference,
        selected_candidate=selected,
        selected_visual_cost=visual_cost[np.arange(count), selected],
        initial_objective=initial_objective,
        final_objective=final_objective,
        outer_iterations_completed=completed,
        optimizer_success=success,
        optimizer_message=message,
        optimizer_nfev=total_nfev,
        assignment_changes=int(np.sum(selected != visual_selected)),
        solution_accepted=accepted,
    )
