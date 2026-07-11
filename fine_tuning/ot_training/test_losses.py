"""Focused CPU tests for OT correspondence objectives."""

from __future__ import annotations

import torch

from .losses import (
    _instance_balanced_mean,
    masked_sinkhorn,
    ot_correspondence_losses,
    patch_grid,
)


def _identity_geometry(batch_size: int):
    pose = torch.eye(4).repeat(batch_size, 1, 1)
    pose[:, 2, 3] = 1000.0
    K = torch.tensor(
        [[[100.0, 0.0, 112.0], [0.0, 100.0, 112.0], [0.0, 0.0, 1.0]]]
    ).repeat(batch_size, 1, 1)
    M = torch.eye(3).repeat(batch_size, 1, 1)
    return pose, K, M


def test_masked_sinkhorn_respects_valid_uniform_marginals():
    logits = torch.eye(4).reshape(1, 4, 4) * 5.0
    src_valid = torch.tensor([[True, True, True, False]])
    tar_valid = torch.tensor([[True, True, False, True]])
    transport = masked_sinkhorn(
        logits,
        src_valid,
        tar_valid,
        iterations=30,
    )
    assert torch.allclose(
        transport.sum(dim=2)[src_valid],
        torch.full((3,), 1.0 / 3.0),
        atol=1e-4,
    )
    assert torch.allclose(
        transport.sum(dim=1)[tar_valid],
        torch.full((3,), 1.0 / 3.0),
        atol=1e-4,
    )
    assert transport[~(src_valid[:, :, None] & tar_valid[:, None, :])].sum() == 0


def test_patch_grid_uses_patch_centers_and_balancing_is_per_instance():
    grid = patch_grid(
        2,
        2,
        patch_size=14,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert torch.equal(
        grid,
        torch.tensor([[7.0, 7.0], [21.0, 7.0], [7.0, 21.0], [21.0, 21.0]]),
    )
    values = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    valid = torch.tensor([[True, True, True], [True, False, False]])
    assert _instance_balanced_mean(values, valid).item() == 1.0


def test_ot_correspondence_loss_backpropagates_to_features():
    batch_size = 1
    channels = 4
    height = width = 2
    src_features = torch.eye(channels).reshape(1, channels, height, width).clone()
    src_features.requires_grad_(True)
    tar_features = src_features.detach().clone().requires_grad_(True)
    src_pts = torch.tensor([[[0, 0], [1, 0], [0, 1], [1, 1]]])
    tar_pts = src_pts.clone()
    pose, K, M = _identity_geometry(batch_size)
    output = ot_correspondence_losses(
        src_features=src_features,
        tar_features=tar_features,
        src_mask=torch.ones(batch_size, 224, 224),
        tar_mask=torch.ones(batch_size, 224, 224),
        src_pts=src_pts,
        tar_pts=tar_pts,
        src_pose=pose,
        tar_pose=pose,
        src_K=K,
        tar_K=K,
        src_M=M,
        tar_M=M,
        feature_temperature=0.07,
        sinkhorn_iterations=20,
        patch_size=14,
        correspondence_weight=1.0,
        soft_patch_reprojection_weight=0.25,
        soft_affine_center_weight=0.25,
        entropy_weight=0.0,
        reprojection_beta=0.05,
        hard_match_confidence=0.05,
    )
    assert torch.isfinite(output.total)
    assert output.valid_gt_pairs.item() == 4.0
    assert output.gt_top1_accuracy.item() == 1.0
    assert output.gt_confident_mutual_accuracy.item() == 1.0
    assert output.confident_mutual_matches_per_instance.item() == 4.0
    output.total.backward()
    assert torch.isfinite(src_features.grad).all()
    assert torch.isfinite(tar_features.grad).all()
