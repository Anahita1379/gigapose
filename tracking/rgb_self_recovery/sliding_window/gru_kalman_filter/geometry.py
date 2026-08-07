"""Differentiable SO(3) operations used by the learned filter."""

from __future__ import annotations

import torch


def hat(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(*vector.shape[:-1], 3, 3)


def so3_exp(rotvec: torch.Tensor) -> torch.Tensor:
    theta2 = (rotvec * rotvec).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(1e-12))
    small = theta2 < 1e-8
    a = torch.where(
        small,
        1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
        torch.sin(theta) / theta,
    )
    b = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-12),
    )
    skew = hat(rotvec)
    identity = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    identity = identity.expand(*rotvec.shape[:-1], 3, 3)
    return identity + a[..., None] * skew + b[..., None] * (skew @ skew)


def so3_log(rotation: torch.Tensor) -> torch.Tensor:
    """Return a stable rotation vector, including rotations close to pi.

    The usual ``theta / (2 sin(theta)) * vee(R - R.T)`` expression has an
    unbounded derivative at pi. Real predictions contain front/rear errors
    arbitrarily close to 180 degrees, so that expression can produce finite
    forward values but NaN gradients.
    """
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    skew_vector = 0.5 * torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    sine = torch.sqrt((skew_vector * skew_vector).sum(dim=-1).clamp_min(1e-12))
    theta = torch.atan2(sine, cosine)
    regular = skew_vector * (theta / sine.clamp_min(1e-6))[..., None]

    # At pi, recover the axis from the symmetric/diagonal part of R. Build a
    # stable candidate around each diagonal and select the largest one.
    diagonal = rotation.diagonal(dim1=-2, dim2=-1)
    axis_x = torch.sqrt(((diagonal[..., 0] + 1.0) * 0.5).clamp_min(1e-8))
    axis_y = torch.sqrt(((diagonal[..., 1] + 1.0) * 0.5).clamp_min(1e-8))
    axis_z = torch.sqrt(((diagonal[..., 2] + 1.0) * 0.5).clamp_min(1e-8))
    four_x = (4.0 * axis_x).clamp_min(1e-4)
    four_y = (4.0 * axis_y).clamp_min(1e-4)
    four_z = (4.0 * axis_z).clamp_min(1e-4)
    candidate_x = torch.stack(
        (
            axis_x,
            (rotation[..., 0, 1] + rotation[..., 1, 0]) / four_x,
            (rotation[..., 0, 2] + rotation[..., 2, 0]) / four_x,
        ), dim=-1,
    )
    candidate_y = torch.stack(
        (
            (rotation[..., 0, 1] + rotation[..., 1, 0]) / four_y,
            axis_y,
            (rotation[..., 1, 2] + rotation[..., 2, 1]) / four_y,
        ), dim=-1,
    )
    candidate_z = torch.stack(
        (
            (rotation[..., 0, 2] + rotation[..., 2, 0]) / four_z,
            (rotation[..., 1, 2] + rotation[..., 2, 1]) / four_z,
            axis_z,
        ), dim=-1,
    )
    largest = diagonal.argmax(dim=-1)
    near_pi_axis = torch.where(
        (largest == 0)[..., None], candidate_x,
        torch.where((largest == 1)[..., None], candidate_y, candidate_z),
    )
    near_pi_axis = near_pi_axis / torch.linalg.norm(
        near_pi_axis, dim=-1, keepdim=True
    ).clamp_min(1e-6)
    near_pi = theta[..., None] * near_pi_axis

    result = torch.where((theta < 1e-4)[..., None], skew_vector, regular)
    return torch.where((cosine < -0.9999)[..., None], near_pi, result)


def rotation_angle(rotation_a: torch.Tensor, rotation_b: torch.Tensor) -> torch.Tensor:
    relative = rotation_a.transpose(-1, -2) @ rotation_b
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    skew_vector = 0.5 * torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ), dim=-1,
    )
    sine = torch.sqrt((skew_vector * skew_vector).sum(dim=-1).clamp_min(1e-12))
    return torch.atan2(sine, cosine)


def make_pose(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    shape = translation.shape[:-1]
    pose = torch.zeros(*shape, 4, 4, dtype=translation.dtype, device=translation.device)
    pose[..., :3, :3] = rotation
    pose[..., :3, 3] = translation
    pose[..., 3, 3] = 1.0
    return pose
