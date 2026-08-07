"""DINOv2-assisted RGB/mask/CAD recovery model isolated from the baseline."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from src.models.network.dinov2_hub import load_dinov2_model
from tracking.rgb_self_recovery.model import ConvBlock, Encoder


DINO_DIMENSIONS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


class DINORGBRenderRecoveryNet(nn.Module):
    """Preserve CNN/CAD branches and add pooled pretrained DINO RGB evidence."""

    def __init__(
        self,
        width: int = 32,
        hidden_dim: int = 384,
        *,
        dino_mode: str = "frozen",
        dino_model: str = "dinov2_vits14",
        dino_input_size: int = 224,
        dino_pretrained: bool = True,
    ):
        super().__init__()
        if dino_mode not in {"frozen", "last_block"}:
            raise ValueError("DINO mode must be frozen or last_block")
        if dino_model not in DINO_DIMENSIONS:
            raise ValueError(f"Unknown DINO model: {dino_model}")
        if dino_input_size < 14 or dino_input_size % 14:
            raise ValueError("DINO input size must be a positive multiple of 14")
        self.dino_mode = dino_mode
        self.dino_model_name = dino_model
        self.dino_input_size = int(dino_input_size)
        self.image_encoder = Encoder(4, width)
        self.render_encoder = Encoder(5, width)
        channels = self.image_encoder.output_channels
        self.fusion = nn.Sequential(
            ConvBlock(channels * 4, channels),
            ConvBlock(channels, channels),
        )
        self.dino = load_dinov2_model(
            "facebookresearch/dinov2", dino_model, pretrained=dino_pretrained
        )
        for parameter in self.dino.parameters():
            parameter.requires_grad = False
        if dino_mode == "last_block":
            if not hasattr(self.dino, "blocks") or not self.dino.blocks:
                raise AttributeError("DINOv2 model has no transformer blocks")
            for parameter in self.dino.blocks[-1].parameters():
                parameter.requires_grad = True
            for parameter in self.dino.norm.parameters():
                parameter.requires_grad = True
        self.dino_projection = nn.Sequential(
            nn.Linear(DINO_DIMENSIONS[dino_model] * 2, channels),
            nn.LayerNorm(channels),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(channels * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 8),
        )
        self.register_buffer(
            "dino_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "dino_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.dino.eval()
        if mode and self.dino_mode == "last_block":
            self.dino.blocks[-1].train(True)
            self.dino.norm.train(True)
        return self

    def trainable_parameter_counts(self) -> dict[str, int]:
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "dino_trainable": sum(
                p.numel() for p in self.dino.parameters() if p.requires_grad
            ),
        }

    def encode_image(self, rgb, observed_mask):
        if observed_mask.ndim == 3:
            observed_mask = observed_mask[:, None]
        cnn = self.image_encoder(torch.cat([rgb, observed_mask], dim=1))
        value = F.interpolate(
            rgb,
            size=(self.dino_input_size, self.dino_input_size),
            mode="bilinear",
            align_corners=False,
        )
        value = (value - self.dino_mean) / self.dino_std
        if self.dino_mode == "frozen":
            with torch.no_grad():
                output = self.dino.forward_features(value)
        else:
            output = self.dino.forward_features(value)
        pooled = torch.cat(
            [
                output["x_norm_clstoken"],
                output["x_norm_patchtokens"].mean(dim=1),
            ],
            dim=1,
        )
        return cnn, self.dino_projection(pooled)

    def forward_encoded(self, image_features, dino_features, rendered):
        render_features = self.render_encoder(rendered)
        if image_features.shape[0] == 1 and rendered.shape[0] > 1:
            image_features = image_features.expand(rendered.shape[0], -1, -1, -1)
            dino_features = dino_features.expand(rendered.shape[0], -1)
        if image_features.shape[0] != rendered.shape[0]:
            raise ValueError("Image and rendered feature batch sizes differ")
        fused = self.fusion(
            torch.cat(
                [
                    image_features,
                    render_features,
                    torch.abs(image_features - render_features),
                    image_features * render_features,
                ],
                dim=1,
            )
        )
        pooled = torch.cat(
            [
                F.adaptive_avg_pool2d(fused, 1).flatten(1),
                F.adaptive_max_pool2d(fused, 1).flatten(1),
                dino_features,
            ],
            dim=1,
        )
        output = self.head(pooled)
        return {
            "center_raw": output[:, :2],
            "log_depth_raw": output[:, 2],
            "rotation_raw": output[:, 3:6],
            "confidence_logit": output[:, 6],
            "quality_raw": output[:, 7],
        }

    def forward(self, rgb, observed_mask, rendered):
        image, dino = self.encode_image(rgb, observed_mask)
        return self.forward_encoded(image, dino, rendered)
