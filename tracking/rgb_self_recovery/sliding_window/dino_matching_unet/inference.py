"""Cached inference and checkpoint loading for DINO matching U-Net."""

from __future__ import annotations

from pathlib import Path
import importlib

import torch

from tracking.rgb_self_recovery.inference import RGBSelfRecoveryPredictor
from tracking.rgb_self_recovery.model import decode_outputs
from tracking.rgb_self_recovery.render_inputs import decode_render_channels

from .model import DINOPatchMatchingUNet


def _configure_attention_backend(device: str) -> None:
    """Avoid selecting CUDA-only xFormers kernels for CPU fallback inference."""
    if torch.device(device).type != "cpu":
        return
    for name in ("dinov2.layers.attention", "dinov2.layers.block"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(module, "XFORMERS_AVAILABLE"):
            module.XFORMERS_AVAILABLE = False


class DINOMatchingUNetPredictor(RGBSelfRecoveryPredictor):
    def __init__(self, model, inference_config, device):
        super().__init__(model, inference_config, device)
        self._image_encoding = None

    @classmethod
    def from_payload(cls, payload: dict, device: str):
        config = payload["model_config"]
        model = DINOPatchMatchingUNet(
            width=int(config["width"]),
            hidden_dim=int(config["hidden_dim"]),
            dino_mode=config["dino_mode"],
            dino_model=config["dino_model"],
            dino_input_size=int(config["dino_input_size"]),
            dino_pretrained=False,
        )
        model.load_state_dict(payload["model_state"])
        return cls(model, payload["inference_config"], device)

    def observation(self, frame, detection):
        result = super().observation(frame, detection)
        with torch.no_grad():
            self._image_encoding = self.model.encode_image(result[0], result[1])
        return result

    def _forward(self, rgb, observed, encoded_renders):
        rendered = torch.from_numpy(
            decode_render_channels(encoded_renders)
        ).to(self.device)
        with torch.no_grad():
            encoding = self._image_encoding
            if encoding is None:
                encoding = self.model.encode_image(rgb, observed)
            raw = self.model.forward_encoded(encoding, rendered)
            decoded = decode_outputs(
                raw,
                max_center_px=float(self.config["max_center_crop_px"]),
                max_log_depth=float(self.config["max_log_depth"]),
                max_rotation_deg=float(self.config["max_rotation_deg"]),
            )
        return {
            name: value.detach().cpu().numpy() for name, value in decoded.items()
        }


def load_predictor(path: Path, device: str = "cuda") -> DINOMatchingUNetPredictor:
    payload = torch.load(Path(path), map_location=device)
    if payload.get("format") != "rgb_render_self_recovery_dino_matching_unet_v1":
        raise ValueError(f"Unsupported DINO matching U-Net checkpoint: {path}")
    predictor = DINOMatchingUNetPredictor.from_payload(payload, device)
    _configure_attention_backend(device)
    return predictor
