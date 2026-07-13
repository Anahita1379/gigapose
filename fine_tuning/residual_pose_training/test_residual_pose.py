"""Focused mathematical tests for residual pose training."""

from __future__ import annotations

import math

import torch

from .geometry import apply_pose_residual, so3_exp
from .heads import PoseResidualHead
from .losses import residual_pose_losses


def identity_pose(batch_size: int = 1, depth: float = 10.0) -> torch.Tensor:
    pose = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    pose[:, 2, 3] = depth
    return pose


def test_head_starts_as_exact_zero_residual():
    head = PoseResidualHead(descriptor_size=8, hidden_dim=16)
    output = head(torch.randn(4, 19))
    assert torch.equal(output["translation"], torch.zeros(4, 3))
    assert torch.equal(output["rotation"], torch.zeros(4, 3))


def test_zero_residual_preserves_pose_and_rotation_toggle():
    pose = identity_pose(2, depth=10.0)
    K = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
    M = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
    zero = torch.zeros(2, 3)
    for enabled in (False, True):
        refined = apply_pose_residual(
            baseline_pose=pose,
            tar_K=K,
            tar_M=M,
            raw_translation=zero,
            raw_rotation=zero,
            max_center_offset_px=56.0,
            max_log_depth_residual=0.5,
            max_rotation_rad=math.radians(20.0),
            enable_rotation=enabled,
        )
        assert torch.allclose(refined.pose, pose, atol=1e-6)


def test_log_depth_and_center_residual_follow_camera_geometry():
    pose = identity_pose(depth=10.0)
    K = torch.eye(3).unsqueeze(0)
    M = torch.eye(3).unsqueeze(0)
    desired_delta_log_z = math.log(2.0)
    max_delta = 1.0
    raw_log_z = math.atanh(desired_delta_log_z / max_delta)
    raw_translation = torch.tensor([[math.atanh(0.5), 0.0, raw_log_z]])
    refined = apply_pose_residual(
        baseline_pose=pose,
        tar_K=K,
        tar_M=M,
        raw_translation=raw_translation,
        raw_rotation=torch.zeros(1, 3),
        max_center_offset_px=2.0,
        max_log_depth_residual=max_delta,
        max_rotation_rad=math.radians(20.0),
        enable_rotation=False,
    )
    # du=1 pixel, depth doubles from 10 to 20.  With identity intrinsics,
    # unprojecting [1, 0, 1] at z=20 gives [20, 0, 20].
    assert torch.allclose(
        refined.pose[0, :3, 3], torch.tensor([20.0, 0.0, 20.0]), atol=1e-5
    )


def test_so3_exp_is_proper_rotation():
    rotvec = torch.tensor([[0.2, -0.1, 0.3]], requires_grad=True)
    rotation = so3_exp(rotvec)
    identity = torch.eye(3).unsqueeze(0)
    assert torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=1e-5)
    assert torch.allclose(torch.linalg.det(rotation), torch.ones(1), atol=1e-5)
    rotation.sum().backward()
    assert torch.isfinite(rotvec.grad).all()


def test_residual_losses_backpropagate_to_translation_and_rotation():
    baseline = identity_pose(depth=10.0)
    target = identity_pose(depth=12.0)
    target[:, 0, 3] = 1.0
    target[:, :3, :3] = so3_exp(torch.tensor([[0.0, 0.1, 0.0]]))
    K = torch.eye(3).unsqueeze(0)
    M = torch.eye(3).unsqueeze(0)
    raw_translation = torch.zeros(1, 3, requires_grad=True)
    raw_rotation = torch.zeros(1, 3, requires_grad=True)
    refined = apply_pose_residual(
        baseline_pose=baseline,
        tar_K=K,
        tar_M=M,
        raw_translation=raw_translation,
        raw_rotation=raw_rotation,
        max_center_offset_px=2.0,
        max_log_depth_residual=0.5,
        max_rotation_rad=math.radians(20.0),
        enable_rotation=True,
    )
    outputs = residual_pose_losses(
        baseline_pose=baseline,
        refined_pose=refined.pose,
        target_pose=target,
        target_K=K,
        target_M=M,
        valid=torch.tensor([True]),
        delta_uv_px=refined.delta_uv_px,
        delta_log_depth=refined.delta_log_depth,
        delta_rotvec=refined.delta_rotvec,
        crop_diagonal_px=math.sqrt(2.0),
        translation_scale=1.0,
        center_beta=0.05,
        log_depth_beta=0.05,
        translation_beta=0.05,
        rotation_beta=0.05,
    )
    loss = (
        outputs.center
        + outputs.log_depth
        + outputs.translation
        + outputs.rotation
    )
    loss.backward()
    assert torch.isfinite(raw_translation.grad).all()
    assert raw_translation.grad.abs().sum() > 0
    assert torch.isfinite(raw_rotation.grad).all()
    assert raw_rotation.grad.abs().sum() > 0
