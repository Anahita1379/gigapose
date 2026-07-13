"""Differentiable IST pose aggregation and residual pose reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from fine_tuning.pose_aware_training.losses import project_object_center


EPS = 1e-6


@dataclass
class AggregatedPose:
    pose: torch.Tensor
    crop_center: torch.Tensor
    valid_instances: torch.Tensor


@dataclass
class RefinedPose:
    pose: torch.Tensor
    crop_center: torch.Tensor
    delta_uv_px: torch.Tensor
    delta_log_depth: torch.Tensor
    delta_rotvec: torch.Tensor


def scatter_mean(
    values: torch.Tensor,
    instance_ids: torch.Tensor,
    num_instances: int,
) -> torch.Tensor:
    output = values.new_zeros((num_instances,) + values.shape[1:])
    counts = values.new_zeros(num_instances)
    output.index_add_(0, instance_ids, values)
    counts.index_add_(
        0,
        instance_ids,
        torch.ones_like(instance_ids, dtype=values.dtype),
    )
    shape = (num_instances,) + (1,) * (values.ndim - 1)
    return output / counts.clamp_min(1).view(shape)


def rotation_from_cos_sin(cos_sin: torch.Tensor) -> torch.Tensor:
    cos_sin = F.normalize(cos_sin, dim=-1, eps=EPS)
    cosine, sine = cos_sin.unbind(dim=-1)
    zero = torch.zeros_like(cosine)
    one = torch.ones_like(cosine)
    return torch.stack(
        [
            torch.stack([cosine, -sine, zero], dim=-1),
            torch.stack([sine, cosine, zero], dim=-1),
            torch.stack([zero, zero, one], dim=-1),
        ],
        dim=-2,
    )


def so3_exp(rotvec: torch.Tensor) -> torch.Tensor:
    """Stable exponential map from an axis-angle vector to SO(3)."""
    theta = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    x, y, z = rotvec.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        [
            torch.stack([zero, -z, y], dim=-1),
            torch.stack([z, zero, -x], dim=-1),
            torch.stack([-y, x, zero], dim=-1),
        ],
        dim=-2,
    )
    theta_matrix = theta.unsqueeze(-1)
    a = torch.sinc(theta_matrix / torch.pi)
    b = 0.5 * torch.sinc(theta_matrix / (2.0 * torch.pi)).square()
    identity = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    identity = identity.expand(rotvec.shape[:-1] + (3, 3))
    return identity + a * skew + b * (skew @ skew)


def bounded_rotvec(raw: torch.Tensor, max_angle_rad: float) -> torch.Tensor:
    """Bound the axis-angle magnitude without independently clipping axes."""
    magnitude = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    ratio = torch.tanh(magnitude) / magnitude.clamp_min(EPS)
    # tanh(x) / x -> 1 at zero.  The explicit series branch preserves the
    # nonzero first derivative needed by a zero-initialized rotation head.
    ratio = torch.where(
        magnitude > 1e-4,
        ratio,
        1.0 - magnitude.square() / 3.0,
    )
    return raw * ratio * float(max_angle_rad)


def crop_center_to_translation(
    crop_center: torch.Tensor,
    depth: torch.Tensor,
    K: torch.Tensor,
    crop_M: torch.Tensor,
) -> torch.Tensor:
    ones = torch.ones_like(crop_center[..., :1])
    crop_h = torch.cat([crop_center, ones], dim=-1)
    full_h = torch.linalg.solve(crop_M, crop_h.unsqueeze(-1)).squeeze(-1)
    full_h = full_h / full_h[..., 2:].clamp_min(EPS)
    ray = torch.linalg.solve(K, full_h.unsqueeze(-1)).squeeze(-1)
    ray = ray / ray[..., 2:].clamp_min(EPS)
    return ray * depth.unsqueeze(-1)


def aggregate_ist_pose(
    *,
    pred_scale: torch.Tensor,
    pred_inplane: torch.Tensor,
    src_pts: torch.Tensor,
    tar_pts: torch.Tensor,
    src_pose: torch.Tensor,
    src_K: torch.Tensor,
    tar_K: torch.Tensor,
    src_M: torch.Tensor,
    tar_M: torch.Tensor,
    patch_size: int,
) -> AggregatedPose:
    """Reconstruct one differentiable baseline pose per training instance."""
    pair_valid = (
        (src_pts[..., 0] >= 0)
        & (src_pts[..., 1] >= 0)
        & (tar_pts[..., 0] >= 0)
        & (tar_pts[..., 1] >= 0)
    )
    valid_indices = pair_valid.nonzero(as_tuple=False)
    batch_size = src_pts.shape[0]
    if pred_scale.shape[0] != len(valid_indices):
        raise ValueError(
            "IST output count does not match valid patch pairs: "
            f"{pred_scale.shape[0]} versus {len(valid_indices)}"
        )
    instance_ids = valid_indices[:, 0]
    finite = (
        torch.isfinite(pred_scale)
        & torch.isfinite(pred_inplane).all(dim=-1)
        & (pred_scale > EPS)
    )
    instance_ids = instance_ids[finite]
    scale = pred_scale[finite].clamp_min(EPS)
    inplane = F.normalize(pred_inplane[finite], dim=-1, eps=EPS)
    counts = torch.bincount(instance_ids, minlength=batch_size)
    valid_instances = counts > 0

    instance_log_scale = scatter_mean(
        torch.log(scale), instance_ids, batch_size
    )
    instance_scale = instance_log_scale.exp()
    instance_inplane = F.normalize(
        scatter_mean(inplane, instance_ids, batch_size),
        dim=-1,
        eps=EPS,
    )

    src_px = src_pts[pair_valid][finite].to(scale.dtype) * float(patch_size)
    tar_px = tar_pts[pair_valid][finite].to(scale.dtype) * float(patch_size)
    cosine, sine = inplane.unbind(dim=-1)
    rotated_src = torch.stack(
        [
            cosine * src_px[:, 0] - sine * src_px[:, 1],
            sine * src_px[:, 0] + cosine * src_px[:, 1],
        ],
        dim=-1,
    )
    translation_2d = tar_px - scale[:, None] * rotated_src

    src_center = project_object_center(src_pose, src_K, src_M)[instance_ids]
    rotated_center = torch.stack(
        [
            cosine * src_center[:, 0] - sine * src_center[:, 1],
            sine * src_center[:, 0] + cosine * src_center[:, 1],
        ],
        dim=-1,
    )
    center_per_pair = scale[:, None] * rotated_center + translation_2d
    crop_center = scatter_mean(center_per_pair, instance_ids, batch_size)

    src_z = src_pose[:, 2, 3]
    src_crop_scale = torch.linalg.vector_norm(src_M[:, :2, 0], dim=1)
    tar_crop_scale = torch.linalg.vector_norm(tar_M[:, :2, 0], dim=1)
    depth = (
        src_z
        * (tar_crop_scale / src_crop_scale.clamp_min(EPS))
        * (tar_K[:, 0, 0] / src_K[:, 0, 0].clamp_min(EPS))
        / instance_scale.clamp_min(EPS)
    )
    translation = crop_center_to_translation(crop_center, depth, tar_K, tar_M)
    rotation = rotation_from_cos_sin(instance_inplane) @ src_pose[:, :3, :3]
    pose = src_pose.clone()
    pose[:, :3, :3] = rotation
    pose[:, :3, 3] = translation
    return AggregatedPose(pose, crop_center, valid_instances)


def paired_instance_descriptor(
    *,
    src_features: torch.Tensor,
    tar_features: torch.Tensor,
    src_pts: torch.Tensor,
    tar_pts: torch.Tensor,
    baseline_pose: torch.Tensor,
    tar_K: torch.Tensor,
    tar_M: torch.Tensor,
    translation_unit: float,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool aligned IST features and append normalized pose context."""
    pair_valid = (
        (src_pts[..., 0] >= 0)
        & (src_pts[..., 1] >= 0)
        & (tar_pts[..., 0] >= 0)
        & (tar_pts[..., 1] >= 0)
    )
    indices = pair_valid.nonzero(as_tuple=False)
    batch_size = src_pts.shape[0]
    instance_ids = indices[:, 0]
    src_xy = src_pts[pair_valid].long()
    tar_xy = tar_pts[pair_valid].long()
    src_vector = src_features[
        instance_ids, :, src_xy[:, 1], src_xy[:, 0]
    ]
    tar_vector = tar_features[
        instance_ids, :, tar_xy[:, 1], tar_xy[:, 0]
    ]
    pair_vector = torch.cat([src_vector, tar_vector], dim=-1)
    pooled = scatter_mean(pair_vector, instance_ids, batch_size)
    counts = torch.bincount(instance_ids, minlength=batch_size)

    center = project_object_center(baseline_pose, tar_K, tar_M)
    crop_width = float(tar_features.shape[-1] * patch_size)
    crop_height = float(tar_features.shape[-2] * patch_size)
    center_context = torch.stack(
        [
            center[:, 0] / max(crop_width, 1.0) - 0.5,
            center[:, 1] / max(crop_height, 1.0) - 0.5,
        ],
        dim=-1,
    )
    log_depth = torch.log(
        baseline_pose[:, 2, 3].abs().clamp_min(EPS)
        / max(float(translation_unit), EPS)
    ).unsqueeze(-1)
    return torch.cat([pooled, center_context, log_depth], dim=-1), counts > 0


def apply_pose_residual(
    *,
    baseline_pose: torch.Tensor,
    tar_K: torch.Tensor,
    tar_M: torch.Tensor,
    raw_translation: torch.Tensor,
    raw_rotation: torch.Tensor,
    max_center_offset_px: float,
    max_log_depth_residual: float,
    max_rotation_rad: float,
    enable_rotation: bool,
) -> RefinedPose:
    baseline_center = project_object_center(baseline_pose, tar_K, tar_M)
    delta_uv = (
        torch.tanh(raw_translation[:, :2]) * float(max_center_offset_px)
    )
    delta_log_depth = (
        torch.tanh(raw_translation[:, 2]) * float(max_log_depth_residual)
    )
    refined_center = baseline_center + delta_uv
    refined_depth = (
        baseline_pose[:, 2, 3].clamp_min(EPS)
        * torch.exp(delta_log_depth)
    )
    refined_translation = crop_center_to_translation(
        refined_center, refined_depth, tar_K, tar_M
    )

    if enable_rotation:
        delta_rotvec = bounded_rotvec(raw_rotation, max_rotation_rad)
        refined_rotation = so3_exp(delta_rotvec) @ baseline_pose[:, :3, :3]
    else:
        delta_rotvec = torch.zeros_like(raw_rotation)
        refined_rotation = baseline_pose[:, :3, :3]

    pose = baseline_pose.clone()
    pose[:, :3, :3] = refined_rotation
    pose[:, :3, 3] = refined_translation
    return RefinedPose(
        pose=pose,
        crop_center=refined_center,
        delta_uv_px=delta_uv,
        delta_log_depth=delta_log_depth,
        delta_rotvec=delta_rotvec,
    )
