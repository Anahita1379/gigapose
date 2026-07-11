"""Validation visualizations specific to isolated pose-aware training."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

from src.libVis.numpy import create_edge_from_mask
from src.libVis.torch import convert_tensor_to_image


def plot_scale_alignment_batch(data, num_samples: int = 16, alpha: float = 0.42):
    """Show predicted and GT crop scale with the IST silhouette alignment.

    Red is the warped template prediction and green is the target mask. The
    printed scale ratio makes systematic over/under-scaling visible without
    treating the 2D silhouette warp as a complete 6D CAD-pose evaluation.
    """
    if not hasattr(data, "pred_relScale") or not hasattr(data, "pred_relInplane"):
        raise ValueError("Batch is missing IST predictions for visualization.")

    batch_size = min(data.src_img.shape[0], num_samples)
    src_masks = convert_tensor_to_image(data.src_mask, type="mask")
    tar_masks = convert_tensor_to_image(data.tar_mask, type="mask")
    if hasattr(data, "tar_actual_img"):
        tar_imgs = np.uint8(
            np.clip(data.tar_actual_img[:batch_size].detach().cpu().numpy(), 0, 1)
            .transpose(0, 2, 3, 1)
            * 255
        )
    else:
        tar_imgs = convert_tensor_to_image(
            data.tar_img[:batch_size], unnormalize=True
        )

    scales = data.pred_relScale[:batch_size].detach().cpu().numpy()
    rotations = data.pred_relInplane[:batch_size].detach().cpu().numpy()
    gt_scales = data.relScale[:batch_size].detach().cpu().numpy()
    overlays = []
    for idx in range(batch_size):
        valid = (
            np.isfinite(scales[idx])
            & np.isfinite(rotations[idx, :, 0])
            & np.isfinite(rotations[idx, :, 1])
            & (scales[idx] > 0)
        )
        image = tar_imgs[idx].copy()
        if not valid.any():
            overlays.append(torch.from_numpy(image / 255.0).permute(2, 0, 1))
            continue

        pred_scale = float(np.median(scales[idx][valid]))
        gt_scale = float(gt_scales[idx])
        scale_ratio = pred_scale / max(gt_scale, 1e-8)
        mean_vector = rotations[idx][valid].mean(axis=0)
        angle_deg = float(np.degrees(np.arctan2(mean_vector[1], mean_vector[0])))

        source_mask = src_masks[idx] > 127
        target_mask = tar_masks[idx] > 127
        source_grid = data.src_pts[idx].detach().cpu().numpy()
        target_grid = data.tar_pts[idx].detach().cpu().numpy()
        valid_points = valid & (source_grid[:, 0] != -1)
        valid_points &= target_grid[:, 0] != -1
        if not valid_points.any():
            overlays.append(torch.from_numpy(image / 255.0).permute(2, 0, 1))
            continue

        angle_rad = np.radians(angle_deg)
        linear = pred_scale * np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)],
            ],
            dtype=np.float32,
        )
        patch_size = 14.0
        source_points = source_grid[valid_points].astype(np.float32) * patch_size
        target_points = target_grid[valid_points].astype(np.float32) * patch_size
        translations = target_points - source_points @ linear.T
        translation = np.median(translations, axis=0)
        affine = np.concatenate([linear, translation[:, None]], axis=1)
        height, width = image.shape[:2]
        warped_mask = cv2.warpAffine(
            source_mask.astype(np.uint8),
            affine,
            (width, height),
            flags=cv2.INTER_NEAREST,
        ).astype(bool)

        color = np.zeros_like(image)
        color[:, :] = (0, 255, 70)
        image[warped_mask] = np.uint8(
            (1.0 - alpha) * image[warped_mask] + alpha * color[warped_mask]
        )
        predicted_edge = create_edge_from_mask(
            image_size=(width, height),
            mask=Image.fromarray(np.uint8(warped_mask) * 255),
        )
        target_edge = create_edge_from_mask(
            image_size=(width, height),
            mask=Image.fromarray(np.uint8(target_mask) * 255),
        )
        image[predicted_edge] = (255, 40, 40)
        image[target_edge] = (0, 255, 70)
        cv2.putText(
            image,
            (
                f"pred={pred_scale:.2f} gt={gt_scale:.2f} "
                f"ratio={scale_ratio:.2f} angle={angle_deg:.1f}"
            ),
            (4, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        overlays.append(torch.from_numpy(image / 255.0).permute(2, 0, 1))
    return torch.stack(overlays)
