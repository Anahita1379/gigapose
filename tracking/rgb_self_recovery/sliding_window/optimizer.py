"""Five-frame candidate-path selection and fixed-window pose smoothing."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from itertools import product
import math

import numpy as np
from scipy.optimize import least_squares

from tracking.geometry import so3_exp, so3_log


def _robust_square(value: float) -> float:
    """Smoothly quadratic near zero and linear for large normalized errors."""
    return float(2.0 * (math.sqrt(1.0 + float(value) ** 2) - 1.0))


def frame_time_s(frame, fps: float) -> float:
    for key, scale in (
        ("timestamp_ns", 1e-9),
        ("image_timestamp_ns", 1e-9),
        ("timestamp_us", 1e-6),
        ("timestamp_ms", 1e-3),
        ("time", 1.0),
    ):
        value = frame.metadata.get(key)
        if value not in (None, ""):
            return float(value) * scale
    return float(frame.im_id) / max(float(fps), 1e-6)


@dataclass
class WindowFrame:
    time_s: float
    candidates: list


class SlidingWindowSelector:
    """Select a coherent candidate path and smooth its current endpoint.

    The neural network is unchanged. Its candidate costs are unary terms; the
    window adds speed, acceleration, and SO(3)-smoothness terms.
    """

    def __init__(self, args):
        self.window_size = int(args.window_size)
        self.candidate_limit = int(args.window_candidates)
        self.fps = float(args.window_fps)
        self.unary_weight = float(args.window_unary_weight)
        self.max_speed = float(args.window_max_relative_speed_mps)
        self.speed_sigma = float(args.window_speed_sigma_mps)
        self.accel_sigma = float(args.window_acceleration_sigma_mps2)
        self.rotation_sigma = math.radians(float(args.window_rotation_sigma_deg))
        self.angular_accel_sigma = math.radians(
            float(args.window_angular_acceleration_sigma_deg_s2)
        )
        self.anchor_translation_sigma = float(args.window_anchor_translation_sigma_m)
        self.anchor_rotation_sigma = math.radians(
            float(args.window_anchor_rotation_sigma_deg)
        )
        self.continuous = bool(args.window_continuous_refinement)
        self.histories: dict[int, deque[WindowFrame]] = {}
        self.selected_anchors: dict[int, object] = {}
        if self.window_size < 1:
            raise ValueError("--window-size must be at least 1")
        if self.candidate_limit < 1:
            raise ValueError("--window-candidates must be positive")

    def remove(self, track_id: int) -> None:
        self.histories.pop(int(track_id), None)
        self.selected_anchors.pop(int(track_id), None)

    def clear(self) -> None:
        """Drop every window when the input sequence changes scene/session."""
        self.histories.clear()
        self.selected_anchors.clear()

    def selected_anchor(self, track_id: int):
        """Return the visually scored candidate selected before smoothing."""
        return self.selected_anchors.get(int(track_id))

    def _sequence_cost(self, frames, indices) -> tuple[float, dict]:
        chosen = [frame.candidates[index] for frame, index in zip(frames, indices)]
        times = np.asarray([frame.time_s for frame in frames], dtype=float)
        poses = [np.asarray(item.pose, dtype=float) for item in chosen]
        unary = 0.0
        for frame, index in zip(frames, indices):
            errors = np.asarray([item.total_error for item in frame.candidates])
            scale = max(float(np.median(np.abs(errors - np.median(errors)))), 0.1)
            unary += (float(errors[index]) - float(errors.min())) / scale
        speed_cost = acceleration_cost = rotation_cost = angular_cost = 0.0
        velocities = []
        angular_velocities = []
        for i in range(1, len(poses)):
            dt = max(times[i] - times[i - 1], 1.0 / max(self.fps, 1e-6))
            velocity = (poses[i][:3, 3] - poses[i - 1][:3, 3]) / dt
            velocities.append(velocity)
            excess = max(0.0, float(np.linalg.norm(velocity)) - self.max_speed)
            speed_cost += _robust_square(excess / max(self.speed_sigma, 1e-6))
            delta_rotation = so3_log(
                poses[i][:3, :3] @ poses[i - 1][:3, :3].T
            )
            omega = delta_rotation / dt
            angular_velocities.append(omega)
            rotation_cost += _robust_square(
                float(np.linalg.norm(delta_rotation))
                / max(self.rotation_sigma, 1e-6)
            )
        for i in range(1, len(velocities)):
            dt = max(times[i + 1] - times[i], 1.0 / max(self.fps, 1e-6))
            acceleration_cost += _robust_square(
                float(np.linalg.norm(velocities[i] - velocities[i - 1]))
                / dt
                / max(self.accel_sigma, 1e-6)
            )
            angular_cost += _robust_square(
                float(
                    np.linalg.norm(
                        angular_velocities[i] - angular_velocities[i - 1]
                    )
                )
                / dt
                / max(self.angular_accel_sigma, 1e-6)
            )
        total = (
            self.unary_weight * unary
            + speed_cost
            + acceleration_cost
            + rotation_cost
            + angular_cost
        )
        return total, {
            "window_unary_cost": unary,
            "window_speed_cost": speed_cost,
            "window_acceleration_cost": acceleration_cost,
            "window_rotation_cost": rotation_cost,
            "window_angular_acceleration_cost": angular_cost,
        }

    def _smooth(self, frames, selected):
        anchors = [
            np.asarray(frame.candidates[index].pose, dtype=float)
            for frame, index in zip(frames, selected)
        ]
        if len(anchors) < 3 or not self.continuous:
            return anchors
        times = np.asarray([frame.time_s for frame in frames], dtype=float)
        base_rotations = [pose[:3, :3].copy() for pose in anchors]
        x0 = np.zeros((len(anchors), 6), dtype=float)
        x0[:, :3] = np.stack([pose[:3, 3] for pose in anchors])

        def unpack(values):
            values = values.reshape(len(anchors), 6)
            output = []
            for index, anchor in enumerate(anchors):
                pose = anchor.copy()
                pose[:3, 3] = values[index, :3]
                pose[:3, :3] = so3_exp(values[index, 3:]) @ base_rotations[index]
                output.append(pose)
            return output

        def residual(values):
            poses = unpack(values)
            result = []
            for pose, anchor in zip(poses, anchors):
                result.extend(
                    ((pose[:3, 3] - anchor[:3, 3]) / self.anchor_translation_sigma).tolist()
                )
                result.extend(
                    (
                        so3_log(pose[:3, :3] @ anchor[:3, :3].T)
                        / self.anchor_rotation_sigma
                    ).tolist()
                )
            velocities = []
            angular = []
            for i in range(1, len(poses)):
                dt = max(times[i] - times[i - 1], 1.0 / self.fps)
                velocities.append((poses[i][:3, 3] - poses[i - 1][:3, 3]) / dt)
                angular.append(
                    so3_log(poses[i][:3, :3] @ poses[i - 1][:3, :3].T) / dt
                )
            for i in range(1, len(velocities)):
                dt = max(times[i + 1] - times[i], 1.0 / self.fps)
                result.extend(
                    ((velocities[i] - velocities[i - 1]) / dt / self.accel_sigma).tolist()
                )
                result.extend(
                    (
                        (angular[i] - angular[i - 1])
                        / dt
                        / self.angular_accel_sigma
                    ).tolist()
                )
            return np.asarray(result)

        solved = least_squares(residual, x0.ravel(), loss="soft_l1", max_nfev=40)
        return unpack(solved.x)

    def select(self, track_id: int, frame, candidates):
        candidates = list(candidates[: self.candidate_limit])
        if not candidates:
            raise ValueError("Sliding window received no candidates")
        history = self.histories.setdefault(
            int(track_id), deque(maxlen=self.window_size)
        )
        history.append(WindowFrame(frame_time_s(frame, self.fps), candidates))
        frames = list(history)
        best_indices = None
        best_cost = float("inf")
        best_parts = {}
        for indices in product(*(range(len(item.candidates)) for item in frames)):
            cost, parts = self._sequence_cost(frames, indices)
            if cost < best_cost:
                best_cost, best_indices, best_parts = cost, indices, parts
        assert best_indices is not None
        smoothed = self._smooth(frames, best_indices)
        selected_rank = int(best_indices[-1])
        selected = candidates[selected_rank]
        self.selected_anchors[int(track_id)] = selected
        selected = replace(
            selected,
            pose=smoothed[-1],
            source=f"{selected.source}|window{len(frames)}",
        )
        ordered = [selected] + [
            item for index, item in enumerate(candidates) if index != selected_rank
        ]
        return selected, ordered, {
            "window_size_used": len(frames),
            "window_selected_rank": selected_rank,
            "window_path_cost": best_cost,
            "window_continuous_refinement": int(self.continuous and len(frames) >= 3),
            **best_parts,
        }
