"""Checkpoint-compatible predictor loader for isolated CNN and DINO variants."""

from pathlib import Path

import numpy as np
import torch

from tracking.rgb_self_recovery.inference import RGBSelfRecoveryPredictor
from tracking.rgb_self_recovery.model import decode_outputs
from tracking.rgb_self_recovery.render_inputs import decode_render_channels

from .dino_model import DINORGBRenderRecoveryNet


class DINOSelfRecoveryPredictor(RGBSelfRecoveryPredictor):
    def __init__(self, model, inference_config, device):
        super().__init__(model, inference_config, device)
        self._image_encoding = None

    @classmethod
    def from_payload(cls, payload, device):
        config = payload["model_config"]
        model = DINORGBRenderRecoveryNet(
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
            raw = self.model.forward_encoded(encoding[0], encoding[1], rendered)
            decoded = decode_outputs(
                raw,
                max_center_px=float(self.config["max_center_crop_px"]),
                max_log_depth=float(self.config["max_log_depth"]),
                max_rotation_deg=float(self.config["max_rotation_deg"]),
            )
        return {
            name: np.asarray(value.detach().cpu().numpy())
            for name, value in decoded.items()
        }


def load_predictor(path: Path, device: str):
    payload = torch.load(Path(path), map_location=device)
    if payload.get("format") == "rgb_render_self_recovery_v1":
        return RGBSelfRecoveryPredictor.load(path, device)
    if payload.get("format") == "rgb_render_self_recovery_dino_v1":
        return DINOSelfRecoveryPredictor.from_payload(payload, device)
    raise ValueError(f"Unsupported RGB self-recovery checkpoint: {path}")
