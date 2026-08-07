"""Supervised, auxiliary-depth, and privileged-distillation objectives."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tracking.rgb_self_recovery.model import decode_outputs, rotation_geodesic_atan2
from tracking.rgb_self_recovery.train import anti_flip_quality_loss, ranking_loss
from tracking.rgb_self_recovery.sliding_window.matching_unet.train import heatmap_kl_loss


def forward_grouped(model, batch, device, *, privileged_depth=False):
    rgb = batch["rgb"].to(device, non_blocking=True)
    observed = batch["observed_mask"].to(device, non_blocking=True)
    rendered = batch["rendered"].to(device, non_blocking=True)
    batch_size, candidates = rendered.shape[:2]
    if privileged_depth:
        encoding = model.encode_image_with_depth(
            rgb, observed,
            batch["observed_depth_m"].to(device, non_blocking=True),
            batch["observed_depth_valid"].to(device, non_blocking=True),
        )
    else:
        encoding = model.encode_image(rgb, observed)
    expanded = [
        feature[:, None].expand(-1, candidates, -1, -1, -1).reshape(
            batch_size * candidates, *feature.shape[1:]
        )
        for feature in encoding
    ]
    raw = model.forward_encoded(
        expanded, rendered.reshape(batch_size * candidates, *rendered.shape[2:])
    )
    return raw, encoding, batch_size, candidates


def supervised_objective(raw, batch, args, device, batch_size, candidates):
    decoded = decode_outputs(
        raw,
        max_center_px=args.max_center_crop_px,
        max_log_depth=args.max_log_depth,
        max_rotation_deg=args.max_rotation_deg,
    )
    correctable = batch["correctable_targets"].to(device).reshape(-1)
    center_target = batch["center_targets"].to(device).reshape(-1, 2)
    depth_target = batch["log_depth_targets"].to(device).reshape(-1)
    rotation_target = batch["rotation_targets"].to(device).reshape(-1, 3)
    confidence_target = batch["confidence_targets"].to(device).reshape(-1)
    quality_target = batch["quality_targets"].to(device).reshape(-1)
    zero = raw["quality_raw"].sum() * 0.0
    if correctable.any():
        center_loss = F.smooth_l1_loss(
            decoded["center_px"][correctable] / args.max_center_crop_px,
            center_target[correctable] / args.max_center_crop_px,
        )
        depth_loss = F.smooth_l1_loss(
            decoded["log_depth"][correctable] / args.max_log_depth,
            depth_target[correctable] / args.max_log_depth,
        )
        rotation_errors = rotation_geodesic_atan2(
            decoded["rotation_rad"][correctable], rotation_target[correctable]
        )
        rotation_loss = F.smooth_l1_loss(
            rotation_errors / math.radians(args.max_rotation_deg),
            torch.zeros_like(rotation_errors),
        )
        rotation_error = torch.rad2deg(rotation_errors).mean()
        center_error = torch.linalg.vector_norm(
            decoded["center_px"][correctable] - center_target[correctable], dim=-1
        ).mean()
        depth_error = torch.abs(
            decoded["log_depth"][correctable] - depth_target[correctable]
        ).mean()
    else:
        center_loss = depth_loss = rotation_loss = zero
        rotation_error = center_error = depth_error = zero
    heatmap = heatmap_kl_loss(
        raw["center_heatmap_logits"], center_target, correctable,
        max_center_px=args.max_center_crop_px, sigma_px=args.heatmap_sigma_px,
    )
    confidence = F.binary_cross_entropy_with_logits(
        raw["confidence_logit"], confidence_target
    )
    quality = F.smooth_l1_loss(decoded["quality"], quality_target)
    predicted_group = decoded["quality"].reshape(batch_size, candidates)
    target_group = quality_target.reshape(batch_size, candidates)
    ranking = ranking_loss(predicted_group, target_group, args.ranking_margin)
    anti_flip = anti_flip_quality_loss(
        predicted_group, target_group,
        rotation_target.reshape(batch_size, candidates, 3),
        margin=args.anti_flip_margin,
        minimum_angle_deg=args.anti_flip_min_angle_deg,
    )
    total = (
        args.center_weight * center_loss
        + args.log_depth_weight * depth_loss
        + args.rotation_weight * rotation_loss
        + args.confidence_weight * confidence
        + args.quality_weight * quality
        + args.ranking_weight * ranking
        + args.heatmap_weight * heatmap
        + (args.anti_flip_weight * anti_flip["loss"] if args.anti_flip_training else zero)
    )
    selected = predicted_group.argmin(dim=1)
    selected_quality = target_group[
        torch.arange(batch_size, device=device), selected
    ].mean()
    metrics = {
        "loss_supervised": float(total.detach()),
        "loss_center": float(center_loss.detach()),
        "loss_log_depth": float(depth_loss.detach()),
        "loss_rotation": float(rotation_loss.detach()),
        "loss_confidence": float(confidence.detach()),
        "loss_quality": float(quality.detach()),
        "loss_ranking": float(ranking.detach()),
        "loss_heatmap": float(heatmap.detach()),
        "loss_anti_flip": float(anti_flip["loss"].detach()),
        "center_error_crop_px": float(center_error.detach()),
        "log_depth_abs_error": float(depth_error.detach()),
        "rotation_error_deg": float(rotation_error.detach()),
        "selected_target_quality": float(selected_quality.detach()),
        "oracle_target_quality": float(target_group.min(dim=1).values.mean().detach()),
    }
    return total, metrics


def auxiliary_depth_objective(model, encoding, batch, args, device):
    logits = model.predict_auxiliary_depth(encoding)
    depth = batch["observed_depth_m"].to(device)
    valid = batch["observed_depth_valid"].to(device).bool()
    target = torch.log1p(torch.clamp(depth, min=0.0)) / math.log1p(
        args.maximum_depth_m
    )
    target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
    valid = F.interpolate(valid.float(), size=logits.shape[-2:], mode="nearest") > 0.5
    if not valid.any():
        return logits.sum() * 0.0
    return F.smooth_l1_loss(torch.sigmoid(logits)[valid], target[valid])


def distillation_objective(student_raw, student_encoding, teacher_raw,
                           teacher_encoding, args):
    output_loss = sum(
        F.smooth_l1_loss(student_raw[name], teacher_raw[name].detach())
        for name in (
            "center_raw", "log_depth_raw", "rotation_raw",
            "confidence_logit", "quality_raw",
        )
    ) / 5.0
    feature_loss = sum(
        F.smooth_l1_loss(student, teacher.detach())
        for student, teacher in zip(student_encoding[:4], teacher_encoding[:4])
    ) / 4.0
    return args.distillation_output_weight * output_loss + args.distillation_feature_weight * feature_loss, output_loss, feature_loss
