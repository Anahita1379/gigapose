"""Differentiable pose-aware losses and pose-monitoring utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


EPS = 1e-6


@dataclass
class PoseAwareOutputs:
    """Losses and reconstructed pose metrics for one IST forward pass."""

    log_depth: torch.Tensor
    inplane: torch.Tensor
    reprojection: torch.Tensor
    direct_translation: torch.Tensor
    direct_depth: torch.Tensor
    direct_rotation: torch.Tensor
    direct_reprojection: torch.Tensor
    translation_error: torch.Tensor
    rotation_error_deg: torch.Tensor
    depth_abs_error: torch.Tensor
    reprojection_error_px: torch.Tensor
    valid_instances: torch.Tensor
    valid_patch_pairs: torch.Tensor


def smooth_l1(error: torch.Tensor, beta: float) -> torch.Tensor:
    """Huber/Smooth-L1 applied to an already-computed residual."""
    if beta <= 0:
        return error.abs()
    absolute = error.abs()
    return torch.where(
        absolute < beta,
        0.5 * absolute.square() / beta,
        absolute - 0.5 * beta,
    )


def project_object_center(
    pose: torch.Tensor,
    K: torch.Tensor,
    crop_M: torch.Tensor,
) -> torch.Tensor:
    """Project the object-frame origin into 224x224 crop coordinates."""
    translation = pose[:, :3, 3]
    full_h = torch.einsum("bij,bj->bi", K, translation)
    full_xy = full_h[:, :2] / full_h[:, 2:].clamp_min(EPS)
    ones = torch.ones_like(full_xy[:, :1])
    full_h = torch.cat([full_xy, ones], dim=1)
    crop_h = torch.einsum("bij,bj->bi", crop_M, full_h)
    return crop_h[:, :2] / crop_h[:, 2:].clamp_min(EPS)


def _rotation_from_cos_sin(cos_sin: torch.Tensor) -> torch.Tensor:
    cos_sin = F.normalize(cos_sin, dim=-1, eps=EPS)
    c, s = cos_sin.unbind(dim=-1)
    row0 = torch.stack([c, -s, torch.zeros_like(c)], dim=-1)
    row1 = torch.stack([s, c, torch.zeros_like(c)], dim=-1)
    row2 = torch.stack(
        [torch.zeros_like(c), torch.zeros_like(c), torch.ones_like(c)],
        dim=-1,
    )
    return torch.stack([row0, row1, row2], dim=-2)


def rotation_geodesic(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """SO(3) geodesic angle in radians using atan2(sin(theta), cos(theta))."""
    relative = predicted.transpose(-1, -2) @ target
    return torch.atan2(
        rotation_sine_from_relative(relative),
        rotation_cosine_from_relative(relative),
    )


def rotation_cosine(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Cosine of the SO(3) geodesic angle."""
    relative = predicted.transpose(-1, -2) @ target
    return rotation_cosine_from_relative(relative)


def rotation_cosine_from_relative(relative: torch.Tensor) -> torch.Tensor:
    """Cosine of SO(3) geodesic angle from a relative rotation matrix."""
    return ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(
        -1.0,
        1.0,
    )


def rotation_sine_from_relative(relative: torch.Tensor) -> torch.Tensor:
    """Sine magnitude of SO(3) geodesic angle from a relative rotation matrix."""
    skew_vector = torch.stack(
        [
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ],
        dim=-1,
    )
    return 0.5 * torch.linalg.vector_norm(skew_vector, dim=-1)


def _scatter_mean(
    values: torch.Tensor,
    instance_ids: torch.Tensor,
    num_instances: int,
) -> torch.Tensor:
    output = values.new_zeros((num_instances,) + values.shape[1:])
    count = values.new_zeros(num_instances)
    output.index_add_(0, instance_ids, values)
    count.index_add_(0, instance_ids, torch.ones_like(instance_ids, dtype=values.dtype))
    shape = (num_instances,) + (1,) * (values.ndim - 1)
    return output / count.clamp_min(1).view(shape)


def pose_aware_ist_losses(
    *,
    pred_scale: torch.Tensor,
    pred_inplane: torch.Tensor,
    src_pts: torch.Tensor,
    tar_pts: torch.Tensor,
    gt_scale: torch.Tensor,
    gt_inplane: torch.Tensor,
    src_pose: torch.Tensor,
    tar_pose: torch.Tensor,
    src_K: torch.Tensor,
    tar_K: torch.Tensor,
    src_M: torch.Tensor,
    tar_M: torch.Tensor,
    patch_size: int,
    log_depth_beta: float,
    reprojection_beta: float,
    direct_translation_scale: float,
    direct_depth_scale: float,
    direct_translation_beta: float,
    direct_depth_beta: float,
    direct_rotation_beta: float,
) -> PoseAwareOutputs:
    """Compute IST losses and reconstruct monitoring-only metric poses.

    IST returns predictions only for valid correspondence pairs. This function
    maps those flattened predictions back to their owning instances.
    """
    pair_valid = (
        (src_pts[..., 0] >= 0)
        & (src_pts[..., 1] >= 0)
        & (tar_pts[..., 0] >= 0)
        & (tar_pts[..., 1] >= 0)
    )
    valid_indices = pair_valid.nonzero(as_tuple=False)
    num_instances = src_pts.shape[0]
    zero = pred_scale.sum() * 0.0 + pred_inplane.sum() * 0.0
    if len(valid_indices) == 0 or pred_scale.numel() == 0:
        nan = zero.detach().new_tensor(float("nan"))
        return PoseAwareOutputs(
            log_depth=zero,
            inplane=zero,
            reprojection=zero,
            direct_translation=zero,
            direct_depth=zero,
            direct_rotation=zero,
            direct_reprojection=zero,
            translation_error=nan,
            rotation_error_deg=nan,
            depth_abs_error=nan,
            reprojection_error_px=nan,
            valid_instances=zero.detach(),
            valid_patch_pairs=zero.detach(),
        )
    if pred_scale.shape[0] != len(valid_indices):
        raise ValueError(
            "IST output count does not match valid patch pairs: "
            f"{pred_scale.shape[0]} versus {len(valid_indices)}"
        )

    instance_ids = valid_indices[:, 0]
    gt_scale_flat = gt_scale[instance_ids]
    gt_inplane_flat = gt_inplane[instance_ids]
    finite = (
        torch.isfinite(pred_scale)
        & torch.isfinite(pred_inplane).all(dim=-1)
        & torch.isfinite(gt_scale_flat)
        & torch.isfinite(gt_inplane_flat)
        & (gt_scale_flat > EPS)
    )
    if not finite.any():
        nan = zero.detach().new_tensor(float("nan"))
        return PoseAwareOutputs(
            log_depth=zero,
            inplane=zero,
            reprojection=zero,
            direct_translation=zero,
            direct_depth=zero,
            direct_rotation=zero,
            direct_reprojection=zero,
            translation_error=nan,
            rotation_error_deg=nan,
            depth_abs_error=nan,
            reprojection_error_px=nan,
            valid_instances=zero.detach(),
            valid_patch_pairs=zero.detach(),
        )

    instance_ids = instance_ids[finite]
    pred_scale = pred_scale[finite]
    pred_scale_safe = pred_scale.clamp_min(EPS)
    pred_inplane = F.normalize(pred_inplane[finite], dim=-1, eps=EPS)
    gt_scale_flat = gt_scale_flat[finite]
    gt_cos_sin = torch.stack(
        [torch.cos(gt_inplane_flat[finite]), torch.sin(gt_inplane_flat[finite])],
        dim=-1,
    )

    # Since z_target is proportional to 1/scale, this is exactly the robust
    # relative log-depth residual up to sign.
    log_depth_residual = torch.log(pred_scale_safe) - torch.log(
        gt_scale_flat.clamp_min(EPS)
    )
    log_depth_loss = smooth_l1(log_depth_residual, log_depth_beta).mean()

    # 1-cos(delta angle) is bounded and has smoother gradients than acos near
    # perfect alignment.
    inplane_loss = (1.0 - (pred_inplane * gt_cos_sin).sum(-1)).mean()

    src_px = src_pts[pair_valid][finite].to(pred_scale.dtype) * float(patch_size)
    tar_px = tar_pts[pair_valid][finite].to(pred_scale.dtype) * float(patch_size)
    c, s = pred_inplane.unbind(-1)
    rotated_src_x = c * src_px[:, 0] - s * src_px[:, 1]
    rotated_src_y = s * src_px[:, 0] + c * src_px[:, 1]
    transformed_src = pred_scale_safe[:, None] * torch.stack(
        [rotated_src_x, rotated_src_y], dim=-1
    )
    translation_2d = tar_px - transformed_src

    src_centers = project_object_center(src_pose, src_K, src_M)
    tar_centers = project_object_center(tar_pose, tar_K, tar_M)
    source_center = src_centers[instance_ids]
    rotated_center_x = c * source_center[:, 0] - s * source_center[:, 1]
    rotated_center_y = s * source_center[:, 0] + c * source_center[:, 1]
    predicted_center_per_patch = (
        pred_scale_safe[:, None]
        * torch.stack([rotated_center_x, rotated_center_y], dim=-1)
        + translation_2d
    )
    target_center_per_patch = tar_centers[instance_ids]
    reprojection_error = torch.linalg.vector_norm(
        predicted_center_per_patch - target_center_per_patch,
        dim=-1,
    )
    crop_diagonal = 224.0 * 2.0**0.5
    reprojection_loss = smooth_l1(
        reprojection_error / crop_diagonal,
        reprojection_beta,
    ).mean()

    # Aggregate patch predictions to one differentiable approximate pose per
    # instance. By default the caller detaches these values for monitoring only;
    # the isolated training entry point can also opt into optimizing their
    # normalized robust losses.
    mean_log_scale = _scatter_mean(
        torch.log(pred_scale_safe), instance_ids, num_instances
    )
    instance_scale = mean_log_scale.exp()
    instance_inplane = F.normalize(
        _scatter_mean(pred_inplane, instance_ids, num_instances),
        dim=-1,
        eps=EPS,
    )
    predicted_crop_center = _scatter_mean(
        predicted_center_per_patch, instance_ids, num_instances
    )
    counts = torch.bincount(instance_ids, minlength=num_instances)
    valid_instances = counts > 0

    src_z = src_pose[:, 2, 3]
    src_crop_scale = torch.linalg.vector_norm(src_M[:, :2, 0], dim=1)
    tar_crop_scale = torch.linalg.vector_norm(tar_M[:, :2, 0], dim=1)
    predicted_z = (
        src_z
        * (tar_crop_scale / src_crop_scale.clamp_min(EPS))
        * (tar_K[:, 0, 0] / src_K[:, 0, 0].clamp_min(EPS))
        / instance_scale.clamp_min(EPS)
    )

    ones = torch.ones_like(predicted_crop_center[:, :1])
    crop_h = torch.cat([predicted_crop_center, ones], dim=1)
    full_h = torch.linalg.solve(tar_M, crop_h.unsqueeze(-1)).squeeze(-1)
    full_h = full_h / full_h[:, 2:].clamp_min(EPS)
    camera_ray = torch.linalg.solve(tar_K, full_h.unsqueeze(-1)).squeeze(-1)
    camera_ray = camera_ray / camera_ray[:, 2:].clamp_min(EPS)
    predicted_translation = camera_ray * predicted_z[:, None]

    inplane_R = _rotation_from_cos_sin(instance_inplane)
    predicted_rotation = inplane_R @ src_pose[:, :3, :3]
    translation_errors = torch.linalg.vector_norm(
        predicted_translation[valid_instances]
        - tar_pose[valid_instances, :3, 3],
        dim=-1,
    )
    rotation_errors = rotation_geodesic(
        predicted_rotation[valid_instances],
        tar_pose[valid_instances, :3, :3],
    )
    depth_errors = (
        predicted_z[valid_instances] - tar_pose[valid_instances, 2, 3]
    ).abs()

    translation_error = translation_errors.mean()
    rotation_error = rotation_errors.mean()
    depth_abs_error = depth_errors.mean()
    direct_translation_loss = smooth_l1(
        translation_errors / max(float(direct_translation_scale), EPS),
        direct_translation_beta,
    ).mean()
    direct_depth_loss = smooth_l1(
        depth_errors / max(float(direct_depth_scale), EPS),
        direct_depth_beta,
    ).mean()
    direct_rotation_loss = smooth_l1(
        rotation_errors / torch.pi,
        direct_rotation_beta,
    ).mean()

    return PoseAwareOutputs(
        log_depth=log_depth_loss,
        inplane=inplane_loss,
        reprojection=reprojection_loss,
        direct_translation=direct_translation_loss,
        direct_depth=direct_depth_loss,
        direct_rotation=direct_rotation_loss,
        direct_reprojection=reprojection_loss,
        translation_error=translation_error,
        rotation_error_deg=torch.rad2deg(rotation_error),
        depth_abs_error=depth_abs_error,
        reprojection_error_px=reprojection_error.mean(),
        valid_instances=valid_instances.sum().to(pred_scale.dtype),
        valid_patch_pairs=finite.sum().to(pred_scale.dtype),
    )


def masked_global_descriptors(
    features: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Mask-pool a dense descriptor map into one normalized vector per crop."""
    resized_mask = F.interpolate(
        masks[:, None].to(features.dtype),
        size=features.shape[-2:],
        mode="nearest",
    )
    pooled = (features * resized_mask).sum(dim=(-2, -1))
    denominator = resized_mask.sum(dim=(-2, -1)).clamp_min(1.0)
    return F.normalize(pooled / denominator, dim=-1, eps=EPS)


def soft_pose_aware_template_loss(
    *,
    src_features: torch.Tensor,
    tar_features: torch.Tensor,
    src_masks: torch.Tensor,
    tar_masks: torch.Tensor,
    src_rotations: torch.Tensor,
    tar_rotations: torch.Tensor,
    labels: torch.Tensor,
    prediction_temperature: float,
    target_temperature_rad: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Pose-aware in-batch soft template selection.

    Candidate templates with another object label are excluded. The target is
    a soft distribution based on SO(3) distance, avoiding a nondifferentiable
    nearest-template argmax.
    """
    src_desc = masked_global_descriptors(src_features, src_masks)
    tar_desc = masked_global_descriptors(tar_features, tar_masks)
    logits = tar_desc @ src_desc.transpose(0, 1)
    logits = logits / max(prediction_temperature, EPS)

    batch_size = logits.shape[0]
    src_R = src_rotations.unsqueeze(0).expand(batch_size, -1, -1, -1)
    tar_R = tar_rotations.unsqueeze(1).expand(-1, batch_size, -1, -1)
    pose_distance = rotation_geodesic(src_R, tar_R)
    same_label = labels[:, None] == labels[None, :]
    masked_logits = logits.masked_fill(~same_label, -1e4)
    target_logits = (-pose_distance / max(target_temperature_rad, EPS)).masked_fill(
        ~same_label, -1e4
    )
    target_probability = torch.softmax(target_logits, dim=1).detach()
    log_probability = torch.log_softmax(masked_logits, dim=1)
    loss = -(target_probability * log_probability).sum(dim=1).mean()

    with torch.no_grad():
        hard_index = masked_logits.argmax(dim=1)
        selected_error = pose_distance[
            torch.arange(batch_size, device=logits.device), hard_index
        ]
        expected_error = (torch.softmax(masked_logits, dim=1) * pose_distance).sum(1)
        entropy = -(
            torch.softmax(masked_logits, dim=1) * log_probability
        ).sum(1)
    return loss, {
        "soft_template_selected_rotation_deg": torch.rad2deg(selected_error).mean(),
        "soft_template_expected_rotation_deg": torch.rad2deg(expected_error).mean(),
        "soft_template_entropy": entropy.mean(),
        "soft_template_top1_diagonal": (
            hard_index == torch.arange(batch_size, device=logits.device)
        ).float().mean(),
    }
