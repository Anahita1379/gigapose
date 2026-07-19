"""Numerically stable NumPy pose geometry in OpenCV camera coordinates."""

from __future__ import annotations

import math

import numpy as np


EPS = 1e-9


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def closest_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=float).reshape(3, 3))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def so3_exp(rotvec: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotvec, dtype=float).reshape(3)
    theta = float(np.linalg.norm(vector))
    K = skew(vector)
    if theta < 1e-8:
        return closest_rotation(np.eye(3) + K + 0.5 * (K @ K))
    a = math.sin(theta) / theta
    b = (1.0 - math.cos(theta)) / (theta * theta)
    return closest_rotation(np.eye(3) + a * K + b * (K @ K))


def so3_log(rotation: np.ndarray) -> np.ndarray:
    R = closest_rotation(rotation)
    vector = np.asarray(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )
    sine = 0.5 * float(np.linalg.norm(vector))
    cosine = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.atan2(sine, cosine)
    if sine < 1e-8:
        if cosine > 0:
            return 0.5 * vector
        values, vectors = np.linalg.eigh(0.5 * (R + np.eye(3)))
        axis = vectors[:, int(np.argmax(values))]
        axis *= 1.0 if axis[int(np.argmax(np.abs(axis)))] >= 0 else -1.0
        return axis * theta
    return vector * (theta / (2.0 * sine))


def pose_from_rt(rotation: np.ndarray, translation_m: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = closest_rotation(rotation)
    pose[:3, 3] = np.asarray(translation_m, dtype=float).reshape(3)
    validate_pose(pose)
    return pose


def validate_pose(pose: np.ndarray) -> None:
    T = np.asarray(pose, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError("Pose must be a finite 4x4 matrix.")
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("Pose bottom row must be [0, 0, 0, 1].")
    if not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-4):
        raise ValueError("Pose rotation must be orthonormal.")
    if np.linalg.det(T[:3, :3]) < 0.999:
        raise ValueError("Pose rotation must have determinant +1.")


def rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3] @ second[:3, :3].T
    return float(np.degrees(np.linalg.norm(so3_log(relative))))


def translation_error_m(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(first[:3, 3] - second[:3, 3]))


def project_center(pose: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, bool]:
    translation = np.asarray(pose[:3, 3], dtype=float)
    if translation[2] <= EPS:
        return np.asarray([np.nan, np.nan]), False
    uvw = np.asarray(K, dtype=float).reshape(3, 3) @ translation
    uv = uvw[:2] / uvw[2]
    return uv, bool(np.isfinite(uv).all())


def translation_from_pixel_depth(
    uv: np.ndarray, depth_m: float, K: np.ndarray
) -> np.ndarray:
    ray = np.linalg.solve(
        np.asarray(K, dtype=float).reshape(3, 3),
        np.asarray([uv[0], uv[1], 1.0], dtype=float),
    )
    ray /= max(float(ray[2]), EPS)
    return ray * max(float(depth_m), EPS)


def shift_projected_center(
    pose: np.ndarray,
    K: np.ndarray,
    delta_uv_px: np.ndarray = np.zeros(2),
    delta_log_depth: float = 0.0,
) -> np.ndarray:
    output = np.asarray(pose, dtype=float).copy()
    center, valid = project_center(output, K)
    if not valid:
        return output
    depth = float(output[2, 3]) * math.exp(float(delta_log_depth))
    output[:3, 3] = translation_from_pixel_depth(
        center + np.asarray(delta_uv_px, dtype=float).reshape(2), depth, K
    )
    return output


def rotate_pose(
    pose: np.ndarray,
    rotvec_rad: np.ndarray,
    *,
    side: str = "left",
) -> np.ndarray:
    output = np.asarray(pose, dtype=float).copy()
    delta = so3_exp(rotvec_rad)
    if side == "left":
        output[:3, :3] = delta @ output[:3, :3]
    elif side == "right":
        output[:3, :3] = output[:3, :3] @ delta
    else:
        raise ValueError("Rotation side must be left or right.")
    output[:3, :3] = closest_rotation(output[:3, :3])
    return output


def flipped_pose(pose: np.ndarray, axis: str = "z") -> np.ndarray:
    index = {"x": 0, "y": 1, "z": 2}[axis]
    vector = np.zeros(3)
    vector[index] = math.pi
    return rotate_pose(pose, vector, side="right")


def translate_pose(pose: np.ndarray, delta_m: np.ndarray) -> np.ndarray:
    output = np.asarray(pose, dtype=float).copy()
    output[:3, 3] += np.asarray(delta_m, dtype=float).reshape(3)
    return output


def propagate_constant_velocity(
    previous_pose: np.ndarray | None, current_pose: np.ndarray
) -> np.ndarray:
    current = np.asarray(current_pose, dtype=float)
    if previous_pose is None:
        return current.copy()
    previous = np.asarray(previous_pose, dtype=float)
    output = current.copy()
    rotation_delta = current[:3, :3] @ previous[:3, :3].T
    output[:3, :3] = closest_rotation(rotation_delta @ current[:3, :3])
    output[:3, 3] = current[:3, 3] + (
        current[:3, 3] - previous[:3, 3]
    )
    return output


def propagate_pose_window(
    history: list[np.ndarray],
    window_size: int = 5,
    robust: bool = True,
) -> np.ndarray:
    """Predict one step using translation/SO(3) velocities from a short window."""

    if not history:
        raise ValueError("Pose history is empty.")
    if len(history) < 2:
        return np.asarray(history[-1], dtype=float).copy()
    poses = [np.asarray(value, dtype=float) for value in history[-max(window_size, 2) :]]
    translation_steps = np.stack(
        [current[:3, 3] - previous[:3, 3] for previous, current in zip(poses, poses[1:])]
    )
    rotation_steps = np.stack(
        [
            so3_log(current[:3, :3] @ previous[:3, :3].T)
            for previous, current in zip(poses, poses[1:])
        ]
    )
    reducer = np.median if robust else np.mean
    translation_velocity = reducer(translation_steps, axis=0)
    rotation_velocity = reducer(rotation_steps, axis=0)
    output = poses[-1].copy()
    output[:3, 3] += translation_velocity
    output[:3, :3] = closest_rotation(
        so3_exp(rotation_velocity) @ output[:3, :3]
    )
    return output


def apply_direct_residual(
    pose: np.ndarray,
    rotation_residual_rad: np.ndarray,
    translation_residual_m: np.ndarray,
) -> np.ndarray:
    output = rotate_pose(pose, rotation_residual_rad, side="left")
    output[:3, 3] = (
        np.asarray(pose[:3, 3], dtype=float)
        + np.asarray(translation_residual_m, dtype=float)
    )
    return output
