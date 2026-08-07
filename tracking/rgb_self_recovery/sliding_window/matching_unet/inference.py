"""Checkpoint loader and cached predictor for the matching U-Net."""

from __future__ import annotations

from pathlib import Path

import torch

from tracking.rgb_self_recovery.inference import RGBSelfRecoveryPredictor
from tracking.rgb_self_recovery.model import decode_outputs
from tracking.rgb_self_recovery.render_inputs import decode_render_channels

from .model import LightweightMatchingUNet


class MatchingUNetPredictor(RGBSelfRecoveryPredictor):
    def __init__(self, model, inference_config, device):
        super().__init__(model, inference_config, device)
        self._image_pyramid = None

    @classmethod
    def from_payload(cls, payload: dict, device: str):
        config = payload["model_config"]
        model = LightweightMatchingUNet(
            width=int(config["width"]),
            hidden_dim=int(config["hidden_dim"]),
        )
        model.load_state_dict(payload["model_state"])
        return cls(model, payload["inference_config"], device)

    def observation(self, frame, detection):
        result = super().observation(frame, detection)
        with torch.no_grad():
            self._image_pyramid = self.model.encode_image(result[0], result[1])
        return result

    def _forward(self, rgb, observed, encoded_renders):
        rendered = torch.from_numpy(
            decode_render_channels(encoded_renders)
        ).to(self.device)
        with torch.no_grad():
            pyramid = self._image_pyramid
            if pyramid is None:
                pyramid = self.model.encode_image(rgb, observed)
            raw = self.model.forward_encoded(pyramid, rendered)
            decoded = decode_outputs(
                raw,
                max_center_px=float(self.config["max_center_crop_px"]),
                max_log_depth=float(self.config["max_log_depth"]),
                max_rotation_deg=float(self.config["max_rotation_deg"]),
            )
        return {
            name: value.detach().cpu().numpy() for name, value in decoded.items()
        }


def load_predictor(path: Path, device: str = "cuda") -> MatchingUNetPredictor:
    payload = torch.load(Path(path), map_location=device)
    if payload.get("format") != "rgb_render_self_recovery_matching_unet_v1":
        raise ValueError(f"Unsupported matching U-Net checkpoint: {path}")
    return MatchingUNetPredictor.from_payload(payload, device)
