"""Fixed-size RGB/segmentation and CAD-render input construction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import cv2
import numpy as np

from tracking.geometry import (
    project_center,
    shift_projected_center,
    so3_log,
)


class RenderProtocol(Protocol):
    def render(
        self,
        pose_m: np.ndarray,
        K: np.ndarray,
        image_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class CropSpec:
    """Affine mapping from the original image into a square network crop."""

    x0: float
    y0: float
    side: float
    output_size: int

    @property
    def scale(self) -> float:
        return float(self.output_size) / max(float(self.side), 1e-6)

    @property
    def affine(self) -> np.ndarray:
        scale = self.scale
        return np.asarray(
            [
                [scale, 0.0, -scale * self.x0],
                [0.0, scale, -scale * self.y0],
            ],
            dtype=np.float64,
        )

    @property
    def homography(self) -> np.ndarray:
        matrix = np.eye(3, dtype=np.float64)
        matrix[:2] = self.affine
        return matrix

    def transform_intrinsics(self, K: np.ndarray) -> np.ndarray:
        return self.homography @ np.asarray(K, dtype=np.float64).reshape(3, 3)


def crop_spec_from_bbox(
    bbox_xywh: np.ndarray,
    *,
    output_size: int = 128,
    crop_scale: float = 2.5,
    minimum_side_px: float = 48.0,
) -> CropSpec:
    x, y, width, height = np.asarray(bbox_xywh, dtype=float).reshape(4)
    center_x = x + 0.5 * width
    center_y = y + 0.5 * height
    side = max(
        float(minimum_side_px),
        float(crop_scale) * max(float(width), float(height), 1.0),
    )
    return CropSpec(
        x0=center_x - 0.5 * side,
        y0=center_y - 0.5 * side,
        side=side,
        output_size=int(output_size),
    )


def warp_rgb(image: np.ndarray, crop: CropSpec) -> np.ndarray:
    return cv2.warpAffine(
        np.asarray(image, dtype=np.uint8),
        crop.affine,
        (crop.output_size, crop.output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def warp_mask(mask: np.ndarray, crop: CropSpec) -> np.ndarray:
    output = cv2.warpAffine(
        np.asarray(mask, dtype=np.uint8),
        crop.affine,
        (crop.output_size, crop.output_size),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return output > 0


def _camera_points(depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    height, width = depth_m.shape
    y, x = np.mgrid[:height, :width]
    z = np.asarray(depth_m, dtype=np.float32)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    X = (x.astype(np.float32) - cx) * z / max(fx, 1e-6)
    Y = (y.astype(np.float32) - cy) * z / max(fy, 1e-6)
    return np.stack([X, Y, z], axis=-1)


def normals_from_rendered_depth(
    depth_m: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Estimate camera-space surface normals from CAD-rendered depth only."""

    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(depth) & (depth > 0)
    points = _camera_points(depth, K)
    dx = np.zeros_like(points)
    dy = np.zeros_like(points)
    dx[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dy[1:-1] = points[2:] - points[:-2]
    normals = np.cross(dx, dy)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    neighbor_valid = valid.copy()
    neighbor_valid[:, 1:-1] &= valid[:, :-2] & valid[:, 2:]
    neighbor_valid[1:-1] &= valid[:-2] & valid[2:]
    usable = neighbor_valid & (norm[..., 0] > 1e-9)
    normals[usable] /= norm[usable]
    normals[~usable] = 0.0
    # Keep a consistent orientation. The sign is not an observed-depth cue;
    # it is simply a deterministic CAD shape channel.
    flip = normals[..., 2] < 0
    normals[flip] *= -1.0
    return np.clip(normals, -1.0, 1.0)


def encode_render_channels(
    mask: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    candidate_depth_m: float,
) -> np.ndarray:
    """Quantize mask, camera normals, and relative CAD depth into five channels."""

    mask_bool = np.asarray(mask, dtype=bool)
    normals = normals_from_rendered_depth(depth_m, mask_bool, K)
    relative_depth = np.zeros_like(depth_m, dtype=np.float32)
    if candidate_depth_m > 1e-6:
        relative_depth[mask_bool] = np.log(
            np.maximum(depth_m[mask_bool], 1e-6) / float(candidate_depth_m)
        )
    relative_depth = np.clip(relative_depth / 0.5, -1.0, 1.0)

    encoded = np.full((5, *mask_bool.shape), 127, dtype=np.uint8)
    encoded[0] = mask_bool.astype(np.uint8) * 255
    encoded[1:4] = np.moveaxis(
        np.round((normals + 1.0) * 127.5).astype(np.uint8), -1, 0
    )
    encoded[4] = np.round((relative_depth + 1.0) * 127.5).astype(np.uint8)
    encoded[1:, ~mask_bool] = 127
    return encoded


def decode_render_channels(encoded: np.ndarray) -> np.ndarray:
    encoded = np.asarray(encoded, dtype=np.float32)
    if encoded.ndim < 3 or encoded.shape[-3] != 5:
        raise ValueError("Encoded render input must have five channels.")
    mask = encoded[..., :1, :, :] / 255.0
    geometry = (encoded[..., 1:, :, :] - 127.0) / 128.0
    geometry *= mask
    return np.concatenate([mask, geometry], axis=-3).astype(np.float32)


def hide_rendered_occluders(
    encoded: np.ndarray,
    occluder_mask: np.ndarray | None,
) -> np.ndarray:
    """Remove pixels known to belong to another visible instance.

    The neutral value for quantized geometry is 127. This lets both training
    and inference compare only the visible part of a candidate without using
    sensor depth.
    """

    output = np.asarray(encoded, dtype=np.uint8).copy()
    if occluder_mask is None:
        return output
    occluded = np.asarray(occluder_mask, dtype=bool)
    if output.shape[-2:] != occluded.shape:
        raise ValueError("Render and occluder masks must have equal image shape.")
    output[..., 0, :, :][..., occluded] = 0
    output[..., 1:, :, :][..., occluded] = 127
    return output


def render_candidate_channels(
    renderer: RenderProtocol,
    pose_m: np.ndarray,
    K_crop: np.ndarray,
    output_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask, depth = renderer.render(
        np.asarray(pose_m, dtype=float),
        np.asarray(K_crop, dtype=float),
        (int(output_size), int(output_size)),
    )
    encoded = encode_render_channels(
        mask,
        depth,
        K_crop,
        float(np.asarray(pose_m)[2, 3]),
    )
    return encoded, np.asarray(mask, dtype=bool), np.asarray(depth, dtype=np.float32)


def recovery_targets(
    candidate_pose: np.ndarray,
    gt_pose: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return image-center, log-depth, and left SO(3) correction targets."""

    candidate_center, candidate_valid = project_center(candidate_pose, K)
    gt_center, gt_valid = project_center(gt_pose, K)
    if not candidate_valid or not gt_valid:
        raise ValueError("Recovery target requires positive finite depth.")
    delta_uv = gt_center - candidate_center
    delta_log_depth = math.log(
        max(float(gt_pose[2, 3]), 1e-6)
        / max(float(candidate_pose[2, 3]), 1e-6)
    )
    delta_rotation = so3_log(
        np.asarray(gt_pose[:3, :3]) @ np.asarray(candidate_pose[:3, :3]).T
    )
    return (
        delta_uv.astype(np.float32),
        float(delta_log_depth),
        delta_rotation.astype(np.float32),
    )


def apply_recovery_delta(
    pose_m: np.ndarray,
    K: np.ndarray,
    delta_uv_px: np.ndarray,
    delta_log_depth: float,
    delta_rotation_rad: np.ndarray,
) -> np.ndarray:
    """Apply translation in image/depth coordinates and left rotation."""

    from tracking.geometry import rotate_pose

    output = shift_projected_center(
        pose_m,
        K,
        delta_uv_px=np.asarray(delta_uv_px, dtype=float),
        delta_log_depth=float(delta_log_depth),
    )
    return rotate_pose(
        output,
        np.asarray(delta_rotation_rad, dtype=float),
        side="left",
    )
