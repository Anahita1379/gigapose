"""Masked optimal-transport correspondence losses for AE/DINO fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from fine_tuning.pose_aware_training.losses import EPS, project_object_center, smooth_l1


@dataclass
class OTOutputs:
    """Losses, diagnostics, and hard matches from one OT forward pass."""

    total: torch.Tensor
    correspondence: torch.Tensor
    soft_patch_reprojection: torch.Tensor
    soft_affine_center: torch.Tensor
    entropy: torch.Tensor
    mean_gt_probability: torch.Tensor
    gt_top1_accuracy: torch.Tensor
    gt_confident_mutual_accuracy: torch.Tensor
    mean_confident_match_score: torch.Tensor
    valid_gt_pairs: torch.Tensor
    valid_soft_rows: torch.Tensor
    mutual_matches_per_instance: torch.Tensor
    confident_mutual_matches_per_instance: torch.Tensor
    transport: torch.Tensor
    hard_src_pts: torch.Tensor
    hard_tar_pts: torch.Tensor
    hard_match_scores: torch.Tensor


def patch_grid(
    height: int,
    width: int,
    *,
    patch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return flattened patch-center coordinates in crop pixels."""
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    # DINO token (x, y) represents the center of its patch, not its upper-left
    # corner. Keeping this convention is especially important when an affine
    # fitted on the patch grid is evaluated at a projected object center.
    return (
        torch.stack([x, y], dim=-1).reshape(-1, 2) + 0.5
    ) * float(patch_size)


def flatten_normalized_features(features: torch.Tensor) -> torch.Tensor:
    features = F.normalize(features, dim=1, eps=EPS)
    return features.flatten(2).transpose(1, 2)


def mask_to_patch_valid(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    resized = F.interpolate(
        mask[:, None].to(torch.float32),
        size=size,
        mode="nearest",
    )
    return resized[:, 0].flatten(1) > 0.5


def masked_sinkhorn(
    logits: torch.Tensor,
    src_valid: torch.Tensor,
    tar_valid: torch.Tensor,
    *,
    iterations: int,
) -> torch.Tensor:
    """Balanced masked Sinkhorn in log-space.

    The resulting transport matrix has total mass 1 per sample, with uniform
    marginal mass over valid source and target patches.
    """
    batch_size, num_src, num_tar = logits.shape
    src_count = src_valid.sum(dim=1).clamp_min(1).to(logits.dtype)
    tar_count = tar_valid.sum(dim=1).clamp_min(1).to(logits.dtype)
    log_mu = torch.full_like(src_valid.to(logits.dtype), -1e4)
    log_nu = torch.full_like(tar_valid.to(logits.dtype), -1e4)
    log_mu[src_valid] = -torch.log(src_count).repeat_interleave(
        src_valid.sum(dim=1)
    )
    log_nu[tar_valid] = -torch.log(tar_count).repeat_interleave(
        tar_valid.sum(dim=1)
    )

    valid_pair = src_valid[:, :, None] & tar_valid[:, None, :]
    log_kernel = logits.masked_fill(~valid_pair, -1e4)
    log_u = logits.new_zeros(batch_size, num_src)
    log_v = logits.new_zeros(batch_size, num_tar)
    for _ in range(iterations):
        log_u = log_mu - torch.logsumexp(log_kernel + log_v[:, None, :], dim=2)
        log_u = log_u.masked_fill(~src_valid, -1e4)
        log_v = log_nu - torch.logsumexp(log_kernel + log_u[:, :, None], dim=1)
        log_v = log_v.masked_fill(~tar_valid, -1e4)
    log_transport = log_kernel + log_u[:, :, None] + log_v[:, None, :]
    return torch.exp(log_transport).masked_fill(~valid_pair, 0.0)


def _valid_gt_pairs(
    src_pts: torch.Tensor,
    tar_pts: torch.Tensor,
    *,
    height: int,
    width: int,
    src_valid: torch.Tensor,
    tar_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = (
        (src_pts[..., 0] >= 0)
        & (src_pts[..., 1] >= 0)
        & (tar_pts[..., 0] >= 0)
        & (tar_pts[..., 1] >= 0)
        & (src_pts[..., 0] < width)
        & (src_pts[..., 1] < height)
        & (tar_pts[..., 0] < width)
        & (tar_pts[..., 1] < height)
    )
    src_index = (src_pts[..., 1].long() * width + src_pts[..., 0].long()).clamp(
        0,
        height * width - 1,
    )
    tar_index = (tar_pts[..., 1].long() * width + tar_pts[..., 0].long()).clamp(
        0,
        height * width - 1,
    )
    batch_index = torch.arange(
        src_pts.shape[0], device=src_pts.device
    )[:, None].expand_as(src_index)
    valid = valid & src_valid[batch_index, src_index] & tar_valid[batch_index, tar_index]
    return valid, src_index, tar_index


def _instance_balanced_mean(
    values: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Average within each batch instance, then average valid instances."""
    weights = valid.to(values.dtype)
    counts = weights.sum(dim=1)
    valid_instances = counts > 0
    if not valid_instances.any():
        return values.sum() * 0.0
    per_instance = (values * weights).sum(dim=1) / counts.clamp_min(1.0)
    return per_instance[valid_instances].mean()


def _weighted_affine(
    src_xy: torch.Tensor,
    tar_xy: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    ones = torch.ones_like(src_xy[:, :1])
    design = torch.cat([src_xy, ones], dim=1)
    weights = weights.clamp_min(0.0)
    normal = design.transpose(0, 1) @ (design * weights[:, None])
    regularizer = torch.eye(3, dtype=src_xy.dtype, device=src_xy.device) * 1e-4
    rhs = design.transpose(0, 1) @ (tar_xy * weights[:, None])
    return torch.linalg.solve(normal + regularizer, rhs)


def _soft_affine_center_loss(
    transport: torch.Tensor,
    src_grid: torch.Tensor,
    tar_expected: torch.Tensor,
    src_valid: torch.Tensor,
    src_pose: torch.Tensor,
    tar_pose: torch.Tensor,
    src_K: torch.Tensor,
    tar_K: torch.Tensor,
    src_M: torch.Tensor,
    tar_M: torch.Tensor,
    *,
    crop_diagonal: float,
    beta: float,
) -> torch.Tensor:
    src_centers = project_object_center(src_pose, src_K, src_M)
    tar_centers = project_object_center(tar_pose, tar_K, tar_M)
    row_mass = transport.sum(dim=2)
    losses = []
    for batch_idx in range(transport.shape[0]):
        valid = src_valid[batch_idx] & (row_mass[batch_idx] > EPS)
        if valid.sum() < 3:
            continue
        affine = _weighted_affine(
            src_grid[valid],
            tar_expected[batch_idx, valid],
            row_mass[batch_idx, valid],
        )
        center_h = torch.cat(
            [
                src_centers[batch_idx],
                src_centers.new_ones(1),
            ]
        )
        predicted_center = center_h @ affine
        error = torch.linalg.vector_norm(predicted_center - tar_centers[batch_idx])
        losses.append(smooth_l1(error / crop_diagonal, beta))
    if not losses:
        return transport.sum() * 0.0
    return torch.stack(losses).mean()


def ot_correspondence_losses(
    *,
    src_features: torch.Tensor,
    tar_features: torch.Tensor,
    src_mask: torch.Tensor,
    tar_mask: torch.Tensor,
    src_pts: torch.Tensor,
    tar_pts: torch.Tensor,
    src_pose: torch.Tensor,
    tar_pose: torch.Tensor,
    src_K: torch.Tensor,
    tar_K: torch.Tensor,
    src_M: torch.Tensor,
    tar_M: torch.Tensor,
    feature_temperature: float,
    sinkhorn_iterations: int,
    patch_size: int,
    correspondence_weight: float,
    soft_patch_reprojection_weight: float,
    soft_affine_center_weight: float,
    entropy_weight: float,
    reprojection_beta: float,
    hard_match_confidence: float,
) -> OTOutputs:
    """Compute OT losses from source/target patch features and GT patch pairs."""
    batch_size, _, height, width = src_features.shape
    src_flat = flatten_normalized_features(src_features)
    tar_flat = flatten_normalized_features(tar_features)
    src_valid = mask_to_patch_valid(src_mask, (height, width))
    tar_valid = mask_to_patch_valid(tar_mask, (height, width))
    logits = src_flat @ tar_flat.transpose(1, 2)
    logits = logits / max(float(feature_temperature), EPS)
    transport = masked_sinkhorn(
        logits,
        src_valid,
        tar_valid,
        iterations=int(sinkhorn_iterations),
    )

    row_mass = transport.sum(dim=2)
    row_conditional = transport / row_mass.clamp_min(EPS)[:, :, None]
    valid_gt, src_index, tar_index = _valid_gt_pairs(
        src_pts,
        tar_pts,
        height=height,
        width=width,
        src_valid=src_valid,
        tar_valid=tar_valid,
    )
    batch_index = torch.arange(batch_size, device=src_features.device)[:, None]
    if valid_gt.any():
        gt_probability = row_conditional[batch_index, src_index, tar_index]
        correspondence_loss = _instance_balanced_mean(
            -torch.log(gt_probability.clamp_min(EPS)), valid_gt
        )
        mean_gt_probability = _instance_balanced_mean(gt_probability, valid_gt)
    else:
        correspondence_loss = transport.sum() * 0.0
        mean_gt_probability = transport.detach().new_tensor(float("nan"))

    grid = patch_grid(
        height,
        width,
        patch_size=patch_size,
        device=src_features.device,
        dtype=src_features.dtype,
    )
    expected_tar = row_conditional @ grid
    gt_tar_xy = grid[tar_index.clamp(0, height * width - 1)]
    if valid_gt.any():
        soft_error = torch.linalg.vector_norm(
            expected_tar[batch_index, src_index] - gt_tar_xy,
            dim=-1,
        )
        crop_diagonal = (
            float(patch_size * height) ** 2
            + float(patch_size * width) ** 2
        ) ** 0.5
        soft_patch_reprojection = _instance_balanced_mean(
            smooth_l1(soft_error / crop_diagonal, reprojection_beta),
            valid_gt,
        )
    else:
        crop_diagonal = (
            float(patch_size * height) ** 2
            + float(patch_size * width) ** 2
        ) ** 0.5
        soft_patch_reprojection = transport.sum() * 0.0

    soft_affine_center = _soft_affine_center_loss(
        transport,
        grid,
        expected_tar,
        src_valid,
        src_pose,
        tar_pose,
        src_K,
        tar_K,
        src_M,
        tar_M,
        crop_diagonal=crop_diagonal,
        beta=reprojection_beta,
    )
    valid_transport = transport[transport > 0]
    entropy = -(valid_transport * torch.log(valid_transport.clamp_min(EPS))).sum(
        dim=0
    ) / max(batch_size, 1)

    hard_scores, hard_tar_index = row_conditional.max(dim=2)
    column_mass = transport.sum(dim=1)
    column_conditional = transport / column_mass.clamp_min(EPS)[:, None, :]
    hard_src_index = column_conditional.argmax(dim=1)
    src_grid_index = torch.arange(height * width, device=src_features.device)
    reverse_src_index = torch.gather(hard_src_index, 1, hard_tar_index)
    mutual = reverse_src_index == src_grid_index[None, :]
    confident = hard_scores >= float(hard_match_confidence)
    all_candidate_rows = src_valid & (row_mass > EPS)
    mutual_valid = all_candidate_rows & mutual
    confident_mutual = mutual_valid & confident

    predicted_gt_tar = torch.gather(hard_tar_index, 1, src_index)
    gt_top1_correct = valid_gt & (predicted_gt_tar == tar_index)
    gt_src_mutual = torch.gather(mutual, 1, src_index)
    gt_src_confident = torch.gather(confident, 1, src_index)
    gt_confident_mutual_correct = (
        gt_top1_correct & gt_src_mutual & gt_src_confident
    )
    gt_top1_accuracy = _instance_balanced_mean(
        gt_top1_correct.to(src_features.dtype), valid_gt
    )
    gt_confident_mutual_accuracy = _instance_balanced_mean(
        gt_confident_mutual_correct.to(src_features.dtype), valid_gt
    )
    if confident_mutual.any():
        mean_confident_match_score = _instance_balanced_mean(
            hard_scores,
            confident_mutual,
        )
    else:
        mean_confident_match_score = transport.detach().new_tensor(float("nan"))
    hard_src_pts = torch.full(
        (batch_size, height * width, 2),
        -1,
        dtype=torch.long,
        device=src_features.device,
    )
    hard_tar_pts = torch.full_like(hard_src_pts, -1)
    src_grid_xy = torch.stack([src_grid_index % width, src_grid_index // width], dim=1)
    hard_tar_xy = torch.stack(
        [hard_tar_index % width, hard_tar_index // width],
        dim=2,
    )
    hard_valid = confident_mutual
    for batch_idx in range(batch_size):
        hard_src_pts[batch_idx, hard_valid[batch_idx]] = src_grid_xy[
            hard_valid[batch_idx]
        ]
        hard_tar_pts[batch_idx, hard_valid[batch_idx]] = hard_tar_xy[
            batch_idx,
            hard_valid[batch_idx],
        ]
    total = (
        float(correspondence_weight) * correspondence_loss
        + float(soft_patch_reprojection_weight) * soft_patch_reprojection
        + float(soft_affine_center_weight) * soft_affine_center
        + float(entropy_weight) * entropy
    )
    return OTOutputs(
        total=total,
        correspondence=correspondence_loss,
        soft_patch_reprojection=soft_patch_reprojection,
        soft_affine_center=soft_affine_center,
        entropy=entropy,
        mean_gt_probability=mean_gt_probability,
        gt_top1_accuracy=gt_top1_accuracy,
        gt_confident_mutual_accuracy=gt_confident_mutual_accuracy,
        mean_confident_match_score=mean_confident_match_score,
        valid_gt_pairs=valid_gt.sum().to(src_features.dtype),
        valid_soft_rows=hard_valid.sum().to(src_features.dtype),
        mutual_matches_per_instance=(
            mutual_valid.to(src_features.dtype).sum(dim=1).mean()
        ),
        confident_mutual_matches_per_instance=(
            confident_mutual.to(src_features.dtype).sum(dim=1).mean()
        ),
        transport=transport,
        hard_src_pts=hard_src_pts,
        hard_tar_pts=hard_tar_pts,
        hard_match_scores=hard_scores,
    )
