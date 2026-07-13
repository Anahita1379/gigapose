"""Losses and interpretable metrics for residual-pose refinement."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fine_tuning.pose_aware_training.losses import (
    project_object_center,
    rotation_geodesic,
    smooth_l1,
)


EPS = 1e-6


@dataclass
class ResidualPoseOutputs:
    center: torch.Tensor
    log_depth: torch.Tensor
    translation: torch.Tensor
    rotation: torch.Tensor
    regularization: torch.Tensor
    refined_translation_error: torch.Tensor
    refined_depth_error: torch.Tensor
    refined_rotation_error_deg: torch.Tensor
    refined_center_error_px: torch.Tensor
    baseline_translation_error: torch.Tensor
    baseline_depth_error: torch.Tensor
    baseline_rotation_error_deg: torch.Tensor
    baseline_center_error_px: torch.Tensor
    center_offset_px: torch.Tensor
    abs_log_depth_residual: torch.Tensor
    rotation_residual_deg: torch.Tensor


def residual_pose_losses(
    *,
    baseline_pose: torch.Tensor,
    refined_pose: torch.Tensor,
    target_pose: torch.Tensor,
    target_K: torch.Tensor,
    target_M: torch.Tensor,
    valid: torch.Tensor,
    delta_uv_px: torch.Tensor,
    delta_log_depth: torch.Tensor,
    delta_rotvec: torch.Tensor,
    crop_diagonal_px: float,
    translation_scale: float,
    center_beta: float,
    log_depth_beta: float,
    translation_beta: float,
    rotation_beta: float,
) -> ResidualPoseOutputs:
    if not valid.any():
        raise ValueError("Residual pose batch contains no valid instances")
    base = baseline_pose[valid]
    refined = refined_pose[valid]
    target = target_pose[valid]
    K = target_K[valid]
    M = target_M[valid]

    target_center = project_object_center(target, K, M)
    base_center = project_object_center(base, K, M)
    refined_center = project_object_center(refined, K, M)
    base_center_error = torch.linalg.vector_norm(
        base_center - target_center, dim=-1
    )
    refined_center_error = torch.linalg.vector_norm(
        refined_center - target_center, dim=-1
    )

    target_z = target[:, 2, 3].clamp_min(EPS)
    refined_z = refined[:, 2, 3].clamp_min(EPS)
    depth_log_error = torch.log(refined_z) - torch.log(target_z)
    refined_translation_error = torch.linalg.vector_norm(
        refined[:, :3, 3] - target[:, :3, 3], dim=-1
    )
    base_translation_error = torch.linalg.vector_norm(
        base[:, :3, 3] - target[:, :3, 3], dim=-1
    )
    refined_rotation_error = rotation_geodesic(
        refined[:, :3, :3], target[:, :3, :3]
    )
    base_rotation_error = rotation_geodesic(
        base[:, :3, :3], target[:, :3, :3]
    )

    center_loss = smooth_l1(
        refined_center_error / max(float(crop_diagonal_px), EPS),
        center_beta,
    ).mean()
    log_depth_loss = smooth_l1(depth_log_error, log_depth_beta).mean()
    translation_loss = smooth_l1(
        refined_translation_error / max(float(translation_scale), EPS),
        translation_beta,
    ).mean()
    rotation_loss = smooth_l1(
        refined_rotation_error / torch.pi, rotation_beta
    ).mean()

    valid_uv = delta_uv_px[valid]
    valid_log_z = delta_log_depth[valid]
    valid_rotvec = delta_rotvec[valid]
    regularization = (
        (valid_uv / max(float(crop_diagonal_px), EPS)).square().sum(-1)
        + valid_log_z.square()
        + valid_rotvec.square().sum(-1)
    ).mean()
    return ResidualPoseOutputs(
        center=center_loss,
        log_depth=log_depth_loss,
        translation=translation_loss,
        rotation=rotation_loss,
        regularization=regularization,
        refined_translation_error=refined_translation_error.mean(),
        refined_depth_error=(refined_z - target_z).abs().mean(),
        refined_rotation_error_deg=torch.rad2deg(refined_rotation_error.mean()),
        refined_center_error_px=refined_center_error.mean(),
        baseline_translation_error=base_translation_error.mean(),
        baseline_depth_error=(base[:, 2, 3] - target_z).abs().mean(),
        baseline_rotation_error_deg=torch.rad2deg(base_rotation_error.mean()),
        baseline_center_error_px=base_center_error.mean(),
        center_offset_px=torch.linalg.vector_norm(valid_uv, dim=-1).mean(),
        abs_log_depth_residual=valid_log_z.abs().mean(),
        rotation_residual_deg=torch.rad2deg(
            torch.linalg.vector_norm(valid_rotvec, dim=-1).mean()
        ),
    )

