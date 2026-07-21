"""Checkpoint loading and batched iterative RGB/render pose recovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from tracking.rgb_self_recovery.model import (
    RGBRenderRecoveryNet,
    decode_outputs,
)
from tracking.rgb_self_recovery.render_inputs import (
    CropSpec,
    apply_recovery_delta,
    crop_spec_from_bbox,
    decode_render_channels,
    hide_rendered_occluders,
    render_candidate_channels,
    warp_mask,
    warp_rgb,
)
from tracking.scoring import mask_iou
from tracking.types import Detection, FrameData


@dataclass
class RecoveryProposal:
    pose: np.ndarray
    source: str
    measurement_score: float = 0.5
    prior_error: float = 0.0


@dataclass
class RecoveryResult:
    pose: np.ndarray
    source: str
    confidence: float
    quality: float
    silhouette_iou: float
    total_error: float
    delta_center_crop_px: np.ndarray
    delta_log_depth: float
    delta_rotation_rad: np.ndarray
    rendered_mask_crop: np.ndarray


class RGBSelfRecoveryPredictor:
    def __init__(
        self,
        model: RGBRenderRecoveryNet,
        inference_config: dict,
        device: str,
    ):
        self.model = model.to(device).eval()
        self.config = dict(inference_config)
        if self.config.get("uses_observed_depth") is not False:
            raise ValueError("RGB self-recovery checkpoint must be no-depth.")
        self.device = device

    @classmethod
    def load(cls, path: Path, device: str = "cuda") -> "RGBSelfRecoveryPredictor":
        payload = torch.load(Path(path), map_location=device)
        if payload.get("format") != "rgb_render_self_recovery_v1":
            raise ValueError(f"Unsupported self-recovery checkpoint: {path}")
        model_config = payload["model_config"]
        model = RGBRenderRecoveryNet(
            width=int(model_config["width"]),
            hidden_dim=int(model_config["hidden_dim"]),
        )
        model.load_state_dict(payload["model_state"])
        return cls(model, payload["inference_config"], device)

    def observation(
        self,
        frame: FrameData,
        detection: Detection,
    ) -> tuple[torch.Tensor, torch.Tensor, CropSpec, np.ndarray]:
        crop = crop_spec_from_bbox(
            detection.bbox_xywh,
            output_size=int(self.config["crop_size"]),
            crop_scale=float(self.config["crop_scale"]),
        )
        rgb = warp_rgb(frame.image, crop).astype(np.float32) / 255.0
        observed = warp_mask(detection.mask, crop)
        rgb_tensor = torch.from_numpy(np.moveaxis(rgb, -1, 0)).unsqueeze(0)
        mask_tensor = torch.from_numpy(observed.astype(np.float32))[None, None]
        return (
            rgb_tensor.to(self.device),
            mask_tensor.to(self.device),
            crop,
            observed,
        )

    def _forward(
        self,
        rgb: torch.Tensor,
        observed: torch.Tensor,
        encoded_renders: np.ndarray,
    ) -> dict[str, np.ndarray]:
        rendered = torch.from_numpy(
            decode_render_channels(encoded_renders)
        ).to(self.device)
        count = rendered.shape[0]
        with torch.no_grad():
            raw = self.model(
                rgb.expand(count, -1, -1, -1),
                observed.expand(count, -1, -1, -1),
                rendered,
            )
            decoded = decode_outputs(
                raw,
                max_center_px=float(self.config["max_center_crop_px"]),
                max_log_depth=float(self.config["max_log_depth"]),
                max_rotation_deg=float(self.config["max_rotation_deg"]),
            )
        return {
            name: value.detach().cpu().numpy()
            for name, value in decoded.items()
        }

    def refine_and_score(
        self,
        renderer,
        frame: FrameData,
        detection: Detection,
        proposals: Sequence[RecoveryProposal],
        *,
        occluder_mask: np.ndarray | None = None,
        iterations: int = 2,
        quality_weight: float = 1.0,
        confidence_weight: float = 0.5,
        silhouette_weight: float = 1.5,
        measurement_weight: float = 0.05,
        prior_weight: float = 0.20,
    ) -> list[RecoveryResult]:
        if not proposals:
            return []
        rgb, observed_tensor, crop, observed_mask = self.observation(
            frame, detection
        )
        K_crop = crop.transform_intrinsics(frame.K)
        occluder_crop = (
            warp_mask(occluder_mask, crop)
            if occluder_mask is not None
            else None
        )
        poses = [np.asarray(item.pose, dtype=float).copy() for item in proposals]
        accumulated_center = np.zeros((len(poses), 2), dtype=np.float32)
        accumulated_log_depth = np.zeros(len(poses), dtype=np.float32)
        accumulated_rotation = np.zeros((len(poses), 3), dtype=np.float32)

        for _ in range(max(0, int(iterations))):
            encoded = np.stack(
                [
                    hide_rendered_occluders(
                        render_candidate_channels(
                            renderer, pose, K_crop, crop.output_size
                        )[0],
                        occluder_crop,
                    )
                    for pose in poses
                ]
            )
            predicted = self._forward(rgb, observed_tensor, encoded)
            for index, pose in enumerate(poses):
                center_crop = predicted["center_px"][index]
                center_original = center_crop / max(crop.scale, 1e-6)
                poses[index] = apply_recovery_delta(
                    pose,
                    frame.K,
                    center_original,
                    float(predicted["log_depth"][index]),
                    predicted["rotation_rad"][index],
                )
                accumulated_center[index] += center_crop
                accumulated_log_depth[index] += predicted["log_depth"][index]
                accumulated_rotation[index] += predicted["rotation_rad"][index]

        encoded = []
        rendered_masks = []
        for pose in poses:
            channels, mask, _ = render_candidate_channels(
                renderer, pose, K_crop, crop.output_size
            )
            channels = hide_rendered_occluders(channels, occluder_crop)
            visible_mask = (
                mask & ~occluder_crop
                if occluder_crop is not None
                else mask
            )
            encoded.append(channels)
            rendered_masks.append(visible_mask)
        encoded_array = np.stack(encoded)
        predicted = self._forward(rgb, observed_tensor, encoded_array)
        results = []
        for index, (proposal, pose, rendered_mask) in enumerate(
            zip(proposals, poses, rendered_masks)
        ):
            overlap = mask_iou(rendered_mask, observed_mask)
            confidence = float(predicted["confidence"][index])
            quality = float(predicted["quality"][index])
            total_error = (
                quality_weight * quality
                + confidence_weight * (1.0 - confidence)
                + silhouette_weight * (1.0 - overlap)
                + measurement_weight
                * (1.0 - float(np.clip(proposal.measurement_score, 0.0, 1.0)))
                + prior_weight * max(float(proposal.prior_error), 0.0)
            )
            results.append(
                RecoveryResult(
                    pose=pose,
                    source=f"{proposal.source}|rgb_recovery",
                    confidence=confidence,
                    quality=quality,
                    silhouette_iou=overlap,
                    total_error=float(total_error),
                    delta_center_crop_px=accumulated_center[index],
                    delta_log_depth=float(accumulated_log_depth[index]),
                    delta_rotation_rad=accumulated_rotation[index],
                    rendered_mask_crop=rendered_mask,
                )
            )
        return sorted(results, key=lambda item: item.total_error)
