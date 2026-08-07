"""Batch GTSAM optimization with alternating RGB/CAD candidate assignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from tracking.geometry import closest_rotation, so3_log


def require_gtsam():
    try:
        import gtsam  # type: ignore
    except ImportError as error:
        raise ImportError(
            "GTSAM is required for this package. Install it with "
            "`python3 -m pip install gtsam==4.2` in the active environment."
        ) from error
    return gtsam


@dataclass(frozen=True)
class GTSAMGraphConfig:
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
    maximum_iterations: int = 100
    maximum_translation_step_m: float = 20.0
    maximum_rotation_step_deg: float = 180.0
    robust_loss: str = "huber"
    robust_scale: float = 1.0
    solver_tolerance: float = 1e-5
    numerical_derivative_step: float = 1e-5

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
            self.numerical_derivative_step,
        )
        if min(positive) <= 0:
            raise ValueError("Factor-graph scales and limits must be positive")
        if min(
            self.visual_assignment_weight,
            self.assignment_current_weight,
            self.assignment_motion_weight,
        ) < 0:
            raise ValueError("Candidate-assignment weights must be nonnegative")
        if self.outer_iterations < 1 or self.maximum_iterations < 1:
            raise ValueError("Optimization iteration limits must be positive")
        if self.robust_loss not in {"none", "huber", "cauchy", "tukey"}:
            raise ValueError(f"Unsupported GTSAM robust loss: {self.robust_loss}")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class GTSAMGraphResult:
    poses: np.ndarray
    selected_candidate: np.ndarray
    selected_visual_cost: np.ndarray
    initial_objective: float
    final_objective: float
    outer_iterations_completed: int
    optimizer_success: bool
    optimizer_message: str
    optimizer_iterations: int
    assignment_changes: int
    solution_accepted: bool


def normalized_visual_cost(total_error: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.asarray(total_error, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    output = np.full(values.shape, np.inf, dtype=float)
    for frame in range(len(values)):
        indices = np.flatnonzero(valid[frame] & np.isfinite(values[frame]))
        if not len(indices):
            raise ValueError(f"Frame {frame} has no valid finite candidate cost")
        selected = values[frame, indices]
        low, high = float(selected.min()), float(selected.max())
        output[frame, indices] = (
            0.0 if high - low < 1e-9 else (selected - low) / (high - low)
        )
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
    config: GTSAMGraphConfig,
) -> np.ndarray:
    """Hard max-mixture approximation using visual and bidirectional motion costs."""
    selected = np.zeros(len(current_poses), dtype=np.int64)
    for frame in range(len(current_poses)):
        motion_translation = None
        motion_rotation = None
        if 0 < frame < len(current_poses) - 1:
            span = max(float(time_s[frame + 1] - time_s[frame - 1]), 1e-6)
            alpha = float((time_s[frame] - time_s[frame - 1]) / span)
            motion_translation = (
                (1.0 - alpha) * current_poses[frame - 1, :3, 3]
                + alpha * current_poses[frame + 1, :3, 3]
            )
            relative = (
                current_poses[frame - 1, :3, :3].T
                @ current_poses[frame + 1, :3, :3]
            )
            from tracking.geometry import so3_exp
            motion_rotation = closest_rotation(
                current_poses[frame - 1, :3, :3]
                @ so3_exp(alpha * so3_log(relative))
            )
        best_cost, best_index = np.inf, -1
        for candidate in np.flatnonzero(valid[frame]):
            translation, rotation = _pose_distance(
                current_poses[frame], candidate_poses[frame, candidate]
            )
            cost = config.assignment_current_weight * (
                translation / config.assignment_translation_scale_m
                + rotation / config.assignment_rotation_scale_deg
            )
            if motion_translation is not None and motion_rotation is not None:
                motion_pose = np.eye(4)
                motion_pose[:3, 3] = motion_translation
                motion_pose[:3, :3] = motion_rotation
                translation, rotation = _pose_distance(
                    motion_pose, candidate_poses[frame, candidate]
                )
                cost += config.assignment_motion_weight * (
                    translation / config.assignment_translation_scale_m
                    + rotation / config.assignment_rotation_scale_deg
                )
            cost += config.visual_assignment_weight * visual_cost[frame, candidate]
            if cost < best_cost:
                best_cost, best_index = cost, int(candidate)
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


def _keys(gtsam, index: int) -> tuple[int, int]:
    return gtsam.symbol("p", index), gtsam.symbol("r", index)


def _robust_noise(gtsam, sigmas: np.ndarray, config: GTSAMGraphConfig):
    base = gtsam.noiseModel.Diagonal.Sigmas(np.asarray(sigmas, dtype=float))
    if config.robust_loss == "none":
        return base
    estimators = {
        "huber": gtsam.noiseModel.mEstimator.Huber,
        "cauchy": gtsam.noiseModel.mEstimator.Cauchy,
        "tukey": gtsam.noiseModel.mEstimator.Tukey,
    }
    estimator = estimators[config.robust_loss].Create(config.robust_scale)
    return gtsam.noiseModel.Robust.Create(estimator, base)


def _point_factor(gtsam, keys: list[int], coefficients: np.ndarray, sigma: float, config):
    coefficients = np.asarray(coefficients, dtype=float)

    def error(_factor, values, jacobians):
        residual = sum(
            coefficient * np.asarray(values.atPoint3(key), dtype=float)
            for key, coefficient in zip(keys, coefficients)
        )
        if jacobians is not None:
            for index, coefficient in enumerate(coefficients):
                jacobians[index] = np.asfortranarray(coefficient * np.eye(3))
        return np.asarray(residual, dtype=float)

    noise = _robust_noise(gtsam, np.full(3, sigma), config)
    return gtsam.CustomFactor(noise, keys, error)


def _rotation_error(rotations, dt0: float, dt1: float) -> np.ndarray:
    average_dt = 0.5 * (dt0 + dt1)
    first = np.asarray(rotations[0].logmap(rotations[1]), dtype=float) / dt0
    second = np.asarray(rotations[1].logmap(rotations[2]), dtype=float) / dt1
    return (second - first) / average_dt


def _rotation_acceleration_factor(
    gtsam, keys: list[int], dt0: float, dt1: float, config: GTSAMGraphConfig
):
    epsilon = config.numerical_derivative_step

    def error(_factor, values, jacobians):
        rotations = [values.atRot3(key) for key in keys]
        residual = _rotation_error(rotations, dt0, dt1)
        if jacobians is not None:
            for factor_index in range(3):
                jacobian = np.empty((3, 3), dtype=float)
                for axis in range(3):
                    delta = np.zeros(3)
                    delta[axis] = epsilon
                    plus = list(rotations)
                    minus = list(rotations)
                    plus[factor_index] = rotations[factor_index].retract(delta)
                    minus[factor_index] = rotations[factor_index].retract(-delta)
                    jacobian[:, axis] = (
                        _rotation_error(plus, dt0, dt1)
                        - _rotation_error(minus, dt0, dt1)
                    ) / (2.0 * epsilon)
                jacobians[factor_index] = np.asfortranarray(jacobian)
        return residual

    noise = _robust_noise(
        gtsam, np.full(3, config.angular_acceleration_sigma_radps2), config
    )
    return gtsam.CustomFactor(noise, keys, error)


def _acceleration_coefficients(dt0: float, dt1: float) -> np.ndarray:
    average_dt = 0.5 * (dt0 + dt1)
    return np.asarray([
        1.0 / (dt0 * average_dt),
        -(1.0 / dt0 + 1.0 / dt1) / average_dt,
        1.0 / (dt1 * average_dt),
    ])


def _jerk_coefficients(dt0: float, dt1: float, dt2: float) -> np.ndarray:
    first = _acceleration_coefficients(dt0, dt1)
    second = _acceleration_coefficients(dt1, dt2)
    jerk_dt = 0.5 * (0.5 * (dt0 + dt1) + 0.5 * (dt1 + dt2))
    output = np.zeros(4)
    output[:3] -= first / jerk_dt
    output[1:] += second / jerk_dt
    return output


def _build_graph(
    selected_measurements: np.ndarray,
    prior_poses: np.ndarray | None,
    time_s: np.ndarray,
    config: GTSAMGraphConfig,
    use_prior_rotation: bool,
):
    gtsam = require_gtsam()
    graph = gtsam.NonlinearFactorGraph()
    measurement_point_noise = _robust_noise(
        gtsam, np.full(3, config.measurement_translation_sigma_m), config
    )
    measurement_rotation_noise = _robust_noise(
        gtsam, np.full(3, np.deg2rad(config.measurement_rotation_sigma_deg)), config
    )
    prior_point_noise = _robust_noise(
        gtsam, np.full(3, config.prior_translation_sigma_m), config
    )
    prior_rotation_noise = _robust_noise(
        gtsam, np.full(3, np.deg2rad(config.prior_rotation_sigma_deg)), config
    )
    for index, pose in enumerate(selected_measurements):
        point_key, rotation_key = _keys(gtsam, index)
        graph.add(gtsam.PriorFactorPoint3(
            point_key, np.asarray(pose[:3, 3], dtype=float), measurement_point_noise
        ))
        graph.add(gtsam.PriorFactorRot3(
            rotation_key, gtsam.Rot3(pose[:3, :3]), measurement_rotation_noise
        ))
        if prior_poses is not None:
            prior = prior_poses[index]
            graph.add(gtsam.PriorFactorPoint3(
                point_key, np.asarray(prior[:3, 3], dtype=float), prior_point_noise
            ))
            if use_prior_rotation:
                graph.add(gtsam.PriorFactorRot3(
                    rotation_key, gtsam.Rot3(prior[:3, :3]), prior_rotation_noise
                ))
    dt = np.diff(time_s).clip(0.02, 1.0)
    for center in range(1, len(selected_measurements) - 1):
        point_keys = [_keys(gtsam, item)[0] for item in range(center - 1, center + 2)]
        rotation_keys = [_keys(gtsam, item)[1] for item in range(center - 1, center + 2)]
        graph.add(_point_factor(
            gtsam,
            point_keys,
            _acceleration_coefficients(float(dt[center - 1]), float(dt[center])),
            config.acceleration_sigma_mps2,
            config,
        ))
        graph.add(_rotation_acceleration_factor(
            gtsam,
            rotation_keys,
            float(dt[center - 1]),
            float(dt[center]),
            config,
        ))
    for right in range(3, len(selected_measurements)):
        graph.add(_point_factor(
            gtsam,
            [_keys(gtsam, item)[0] for item in range(right - 3, right + 1)],
            _jerk_coefficients(
                float(dt[right - 3]), float(dt[right - 2]), float(dt[right - 1])
            ),
            config.jerk_sigma_mps3,
            config,
        ))
    return graph


def _values_from_poses(poses: np.ndarray):
    gtsam = require_gtsam()
    values = gtsam.Values()
    for index, pose in enumerate(poses):
        point_key, rotation_key = _keys(gtsam, index)
        values.insert(point_key, np.asarray(pose[:3, 3], dtype=float))
        values.insert(rotation_key, gtsam.Rot3(pose[:3, :3]))
    return values


def _poses_from_values(values, count: int) -> np.ndarray:
    gtsam = require_gtsam()
    poses = np.repeat(np.eye(4)[None], count, axis=0)
    for index in range(count):
        point_key, rotation_key = _keys(gtsam, index)
        poses[index, :3, 3] = np.asarray(values.atPoint3(point_key), dtype=float)
        poses[index, :3, :3] = closest_rotation(
            np.asarray(values.atRot3(rotation_key).matrix(), dtype=float)
        )
    return poses


def _maximum_pose_change(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    distances = [_pose_distance(a, b) for a, b in zip(first, second)]
    if not distances:
        return 0.0, 0.0
    return max(item[0] for item in distances), max(item[1] for item in distances)


def optimize_sequence(
    candidate_poses: np.ndarray,
    valid: np.ndarray,
    visual_cost: np.ndarray,
    time_s: np.ndarray,
    *,
    initial_poses: np.ndarray,
    config: GTSAMGraphConfig,
    prior_poses: np.ndarray | None = None,
    use_prior_rotation: bool = True,
) -> GTSAMGraphResult:
    """Optimize one complete contiguous segment; ground truth is never accepted."""
    gtsam = require_gtsam()
    config.validate()
    candidates = np.asarray(candidate_poses, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    visual_cost = np.asarray(visual_cost, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    reference = np.asarray(initial_poses, dtype=float).copy()
    prior = None if prior_poses is None else np.asarray(prior_poses, dtype=float)
    count = len(reference)
    if count == 0 or (count > 1 and np.any(np.diff(time_s) <= 0)):
        raise ValueError("Sequence timestamps must be nonempty and strictly increasing")
    if candidates.shape[:2] != valid.shape or candidates.shape[0] != count:
        raise ValueError("Candidate, validity, and initialization shapes disagree")
    if visual_cost.shape != valid.shape:
        raise ValueError("Visual costs and candidate validity disagree")
    if prior is not None and prior.shape != reference.shape:
        raise ValueError("External prior and initialization shapes differ")

    visual_selected = np.asarray([
        int(indices[np.argmin(visual_cost[frame, indices])])
        for frame in range(count)
        for indices in [np.flatnonzero(valid[frame])]
    ], dtype=np.int64)
    selected = assign_candidates(reference, candidates, valid, visual_cost, time_s, config)
    first_selected = selected.copy()
    first_reference = reference.copy()
    total_iterations = 0
    completed = 0
    success = True
    messages = []
    initial_objective = None
    final_objective = None

    for outer in range(config.outer_iterations):
        measurements = candidates[np.arange(count), selected]
        graph = _build_graph(
            measurements, prior, time_s, config, use_prior_rotation
        )
        initial_values = _values_from_poses(reference)
        graph_initial = float(graph.error(initial_values))
        if initial_objective is None:
            initial_objective = graph_initial / max(int(graph.size()), 1)
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(config.maximum_iterations)
        params.setRelativeErrorTol(config.solver_tolerance)
        params.setAbsoluteErrorTol(config.solver_tolerance)
        try:
            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, params)
            optimized_values = optimizer.optimize()
            iterations = int(optimizer.iterations())
            candidate_reference = _poses_from_values(optimized_values, count)
            graph_final = float(graph.error(optimized_values))
            total_iterations += iterations
            translation_step, rotation_step = _maximum_pose_change(
                reference, candidate_reference
            )
            outer_valid = bool(
                np.isfinite(candidate_reference).all()
                and np.isfinite(graph_final)
                and graph_final <= graph_initial * 1.05 + 1e-12
                and translation_step <= config.maximum_translation_step_m
                and rotation_step <= config.maximum_rotation_step_deg + 1e-8
            )
            messages.append(
                f"outer={outer + 1} iterations={iterations} "
                f"error={graph_initial:.6g}->{graph_final:.6g} "
                f"step={translation_step:.3f}m/{rotation_step:.3f}deg "
                f"accepted={outer_valid}"
            )
            if not outer_valid:
                success = False
                break
            reference = candidate_reference
            final_objective = graph_final / max(int(graph.size()), 1)
        except Exception as error:
            success = False
            messages.append(f"outer={outer + 1} {type(error).__name__}: {error}")
            break
        completed = outer + 1
        reassigned = assign_candidates(
            reference, candidates, valid, visual_cost, time_s, config
        )
        if np.array_equal(selected, reassigned):
            messages.append("candidate assignments stabilized")
            break
        if outer + 1 < config.outer_iterations:
            selected = reassigned
        else:
            messages.append(
                "final reassignment was not exported because it was not followed "
                "by a continuous optimization"
            )

    if initial_objective is None:
        initial_objective = float("inf")
    accepted = bool(
        completed > 0
        and np.isfinite(reference).all()
        and final_objective is not None
        and np.isfinite(final_objective)
    )
    if not accepted:
        reference = first_reference
        selected = first_selected
        final_objective = initial_objective
    return GTSAMGraphResult(
        poses=reference,
        selected_candidate=selected,
        selected_visual_cost=visual_cost[np.arange(count), selected],
        initial_objective=float(initial_objective),
        final_objective=float(final_objective),
        outer_iterations_completed=completed,
        optimizer_success=success,
        optimizer_message="; ".join(messages),
        optimizer_iterations=total_iterations,
        assignment_changes=int(np.sum(selected != visual_selected)),
        solution_accepted=accepted,
    )
