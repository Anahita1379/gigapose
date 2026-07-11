"""Focused CPU tests for the isolated mathematical objectives."""

from __future__ import annotations

import math

import torch

from .losses import pose_aware_ist_losses, soft_pose_aware_template_loss


def _identity_geometry(batch_size: int):
    pose = torch.eye(4).repeat(batch_size, 1, 1)
    pose[:, 2, 3] = 1000.0
    K = torch.tensor(
        [[[100.0, 0.0, 112.0], [0.0, 100.0, 112.0], [0.0, 0.0, 1.0]]]
    ).repeat(batch_size, 1, 1)
    M = torch.eye(3).repeat(batch_size, 1, 1)
    return pose, K, M


def test_perfect_ist_prediction_has_zero_losses_and_pose_errors():
    src_pts = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0]]])
    tar_pts = src_pts.clone()
    pose, K, M = _identity_geometry(1)
    scale = torch.ones(2, requires_grad=True)
    inplane = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    output = pose_aware_ist_losses(
        pred_scale=scale,
        pred_inplane=inplane,
        src_pts=src_pts,
        tar_pts=tar_pts,
        gt_scale=torch.ones(1),
        gt_inplane=torch.zeros(1),
        src_pose=pose,
        tar_pose=pose,
        src_K=K,
        tar_K=K,
        src_M=M,
        tar_M=M,
        patch_size=14,
        log_depth_beta=0.1,
        reprojection_beta=0.05,
        direct_translation_scale=1000.0,
        direct_depth_scale=1000.0,
        direct_translation_beta=0.05,
        direct_depth_beta=0.05,
        direct_rotation_beta=0.05,
        anti_flip_margin=0.25,
    )
    assert output.log_depth.item() == 0.0
    assert output.instance_log_scale.item() == 0.0
    assert output.scale_consistency.item() == 0.0
    assert output.inplane.item() == 0.0
    assert output.reprojection.item() == 0.0
    assert output.direct_translation.item() == 0.0
    assert output.direct_depth.item() == 0.0
    assert output.direct_rotation.item() == 0.0
    assert output.direct_reprojection.item() == 0.0
    assert output.translation_error.item() == 0.0
    assert output.rotation_error_deg.item() == 0.0
    assert output.scale_signed_log_bias.item() == 0.0
    assert output.scale_abs_log_error.item() == 0.0
    assert output.scale_median_ratio.item() == 1.0
    assert output.scale_within_5pct.item() == 1.0
    (
        output.log_depth
        + output.inplane
        + output.reprojection
        + output.direct_translation
        + output.direct_depth
        + output.direct_rotation
    ).backward()
    assert torch.isfinite(scale.grad).all()
    assert torch.isfinite(inplane.grad).all()


def test_soft_template_loss_backpropagates_to_both_descriptor_sets():
    batch_size = 3
    src = torch.randn(batch_size, 8, 4, 4, requires_grad=True)
    tar = torch.randn(batch_size, 8, 4, 4, requires_grad=True)
    pose, _, _ = _identity_geometry(batch_size)
    loss, _ = soft_pose_aware_template_loss(
        src_features=src,
        tar_features=tar,
        src_masks=torch.ones(batch_size, 16, 16),
        tar_masks=torch.ones(batch_size, 16, 16),
        src_rotations=pose[:, :3, :3],
        tar_rotations=pose[:, :3, :3],
        labels=torch.ones(batch_size, dtype=torch.long),
        prediction_temperature=0.1,
        target_temperature_rad=0.25,
    )
    loss.backward()
    assert torch.isfinite(src.grad).all()
    assert torch.isfinite(tar.grad).all()


def test_nonpositive_predicted_scale_does_not_drop_inplane_supervision():
    src_pts = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])
    tar_pts = src_pts.clone()
    pose, K, M = _identity_geometry(1)
    scale = torch.tensor([-1.0, 1.0], requires_grad=True)
    inplane = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    output = pose_aware_ist_losses(
        pred_scale=scale,
        pred_inplane=inplane,
        src_pts=src_pts,
        tar_pts=tar_pts,
        gt_scale=torch.ones(1),
        gt_inplane=torch.zeros(1),
        src_pose=pose,
        tar_pose=pose,
        src_K=K,
        tar_K=K,
        src_M=M,
        tar_M=M,
        patch_size=14,
        log_depth_beta=0.1,
        reprojection_beta=0.05,
        direct_translation_scale=1000.0,
        direct_depth_scale=1000.0,
        direct_translation_beta=0.05,
        direct_depth_beta=0.05,
        direct_rotation_beta=0.05,
        anti_flip_margin=0.25,
    )
    assert output.valid_patch_pairs.item() == 2.0
    assert torch.isfinite(output.log_depth)
    assert torch.isfinite(output.inplane)
    assert torch.isfinite(output.reprojection)
    (output.log_depth + output.inplane + output.reprojection).backward()
    assert torch.isfinite(scale.grad).all()
    assert torch.isfinite(inplane.grad).all()


def test_exact_depth_and_inplane_prediction_reconstructs_pose_metrics():
    src_pts = torch.tensor([[[8.0, 8.0], [8.0, 8.0]]])
    tar_pts = src_pts.clone()
    src_pose, K, M = _identity_geometry(1)
    tar_pose = src_pose.clone()
    tar_pose[:, 2, 3] = 500.0
    angle = math.radians(30.0)
    rotation_z = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    tar_pose[:, :3, :3] = rotation_z
    scale = torch.full((2,), 2.0, requires_grad=True)
    inplane = torch.tensor(
        [[math.cos(angle), math.sin(angle)], [math.cos(angle), math.sin(angle)]],
        requires_grad=True,
    )
    output = pose_aware_ist_losses(
        pred_scale=scale,
        pred_inplane=inplane,
        src_pts=src_pts,
        tar_pts=tar_pts,
        gt_scale=torch.full((1,), 2.0),
        gt_inplane=torch.full((1,), angle),
        src_pose=src_pose,
        tar_pose=tar_pose,
        src_K=K,
        tar_K=K,
        src_M=M,
        tar_M=M,
        patch_size=14,
        log_depth_beta=0.1,
        reprojection_beta=0.05,
        direct_translation_scale=1000.0,
        direct_depth_scale=1000.0,
        direct_translation_beta=0.05,
        direct_depth_beta=0.05,
        direct_rotation_beta=0.05,
        anti_flip_margin=0.25,
    )
    assert output.log_depth.item() == 0.0
    assert torch.isclose(output.depth_abs_error, torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(output.translation_error, torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(output.rotation_error_deg, torch.tensor(0.0), atol=1e-4)


def test_flipped_inplane_prediction_is_detected_by_monitor():
    angle = math.radians(30.0)
    src_pts = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])
    tar_pts = src_pts.clone()
    pose, K, M = _identity_geometry(1)
    scale = torch.ones(2, requires_grad=True)
    # The flipped alternative for +30 degrees is -30 degrees.
    inplane = torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.cos(angle), -math.sin(angle)]],
        requires_grad=True,
    )
    output = pose_aware_ist_losses(
        pred_scale=scale,
        pred_inplane=inplane,
        src_pts=src_pts,
        tar_pts=tar_pts,
        gt_scale=torch.ones(1),
        gt_inplane=torch.full((1,), angle),
        src_pose=pose,
        tar_pose=pose,
        src_K=K,
        tar_K=K,
        src_M=M,
        tar_M=M,
        patch_size=14,
        log_depth_beta=0.1,
        reprojection_beta=0.05,
        direct_translation_scale=1000.0,
        direct_depth_scale=1000.0,
        direct_translation_beta=0.05,
        direct_depth_beta=0.05,
        direct_rotation_beta=0.05,
        anti_flip_margin=0.25,
    )
    assert output.flip_closer_fraction.item() == 1.0
    assert output.flip_margin.item() < 0.0
    assert output.anti_flip.item() > 0.25


def test_scale_losses_are_instance_balanced_and_penalize_patch_spread():
    src_pts = torch.tensor(
        [
            [[1.0, 1.0], [-1.0, -1.0], [-1.0, -1.0]],
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        ]
    )
    tar_pts = src_pts.clone()
    pose, K, M = _identity_geometry(2)
    scale = torch.tensor([1.0, 2.0, 2.0, 2.0], requires_grad=True)
    inplane = torch.tensor([[1.0, 0.0]] * 4, requires_grad=True)
    output = pose_aware_ist_losses(
        pred_scale=scale,
        pred_inplane=inplane,
        src_pts=src_pts,
        tar_pts=tar_pts,
        gt_scale=torch.ones(2),
        gt_inplane=torch.zeros(2),
        src_pose=pose,
        tar_pose=pose,
        src_K=K,
        tar_K=K,
        src_M=M,
        tar_M=M,
        patch_size=14,
        log_depth_beta=0.1,
        reprojection_beta=0.05,
        direct_translation_scale=1000.0,
        direct_depth_scale=1000.0,
        direct_translation_beta=0.05,
        direct_depth_beta=0.05,
        direct_rotation_beta=0.05,
        anti_flip_margin=0.25,
    )
    expected = 0.5 * (math.log(2.0) - 0.05)
    assert torch.isclose(output.log_depth, torch.tensor(expected), atol=1e-6)
    assert torch.isclose(
        output.instance_log_scale, torch.tensor(expected), atol=1e-6
    )
    assert output.scale_consistency.item() == 0.0

    spread_scale = torch.tensor([0.5, 2.0], requires_grad=True)
    spread_pts = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])
    pose1, K1, M1 = _identity_geometry(1)
    spread_output = pose_aware_ist_losses(
        pred_scale=spread_scale,
        pred_inplane=torch.tensor(
            [[1.0, 0.0], [1.0, 0.0]], requires_grad=True
        ),
        src_pts=spread_pts,
        tar_pts=spread_pts,
        gt_scale=torch.ones(1),
        gt_inplane=torch.zeros(1),
        src_pose=pose1,
        tar_pose=pose1,
        src_K=K1,
        tar_K=K1,
        src_M=M1,
        tar_M=M1,
        patch_size=14,
        log_depth_beta=0.1,
        reprojection_beta=0.05,
        direct_translation_scale=1000.0,
        direct_depth_scale=1000.0,
        direct_translation_beta=0.05,
        direct_depth_beta=0.05,
        direct_rotation_beta=0.05,
        anti_flip_margin=0.25,
    )
    assert torch.isclose(spread_output.instance_log_scale, torch.tensor(0.0))
    assert spread_output.scale_consistency.item() > 0.0
