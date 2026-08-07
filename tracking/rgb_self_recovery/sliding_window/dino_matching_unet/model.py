"""Fuse spatial DINOv2 tokens into the lightweight CAD matching U-Net."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from src.models.network.dinov2_hub import load_dinov2_model
from tracking.rgb_self_recovery.model import ConvBlock
from tracking.rgb_self_recovery.sliding_window.dino_model import DINO_DIMENSIONS
from tracking.rgb_self_recovery.sliding_window.matching_unet.model import (
    DecoderBlock,
    LightweightMatchingUNet,
    MatchBlock,
    PyramidEncoder,
)


def _group_count(channels: int) -> int:
    groups = min(16, channels)
    while channels % groups:
        groups -= 1
    return groups


class DINOPatchMatchingUNet(nn.Module):
    """Combine semantic DINO patch features with spatial RGB/CAD matching."""

    def __init__(
        self,
        width: int = 24,
        hidden_dim: int = 256,
        *,
        dino_mode: str = "frozen",
        dino_model: str = "dinov2_vits14",
        dino_input_size: int = 224,
        dino_pretrained: bool = True,
    ):
        super().__init__()
        if dino_mode not in {"frozen", "last_block"}:
            raise ValueError("DINO mode must be frozen or last_block.")
        if dino_model not in DINO_DIMENSIONS:
            raise ValueError(f"Unknown DINO model: {dino_model}")
        if dino_input_size < 14 or dino_input_size % 14:
            raise ValueError("DINO input size must be a positive multiple of 14.")
        self.width = int(width)
        self.hidden_dim = int(hidden_dim)
        self.dino_mode = dino_mode
        self.dino_model_name = dino_model
        self.dino_input_size = int(dino_input_size)

        self.image_encoder = PyramidEncoder(4, self.width)
        self.render_encoder = PyramidEncoder(5, self.width)
        channels = self.image_encoder.channels
        self.matches = nn.ModuleList([MatchBlock(channel) for channel in channels])

        self.dino = load_dinov2_model(
            "facebookresearch/dinov2", dino_model, pretrained=dino_pretrained
        )
        for parameter in self.dino.parameters():
            parameter.requires_grad = False
        if dino_mode == "last_block":
            if not hasattr(self.dino, "blocks") or not self.dino.blocks:
                raise AttributeError("DINOv2 model has no transformer blocks.")
            for parameter in self.dino.blocks[-1].parameters():
                parameter.requires_grad = True
            for parameter in self.dino.norm.parameters():
                parameter.requires_grad = True

        dino_dim = DINO_DIMENSIONS[dino_model]
        middle_channels = channels[2]
        self.dino_spatial_projection = nn.Sequential(
            nn.Conv2d(dino_dim, middle_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(middle_channels), middle_channels),
            nn.SiLU(),
        )
        self.dino_global_projection = nn.Sequential(
            nn.Linear(dino_dim * 2, middle_channels),
            nn.LayerNorm(middle_channels),
            nn.SiLU(),
        )
        self.decode_2 = nn.Sequential(
            ConvBlock(channels[3] + middle_channels * 2, middle_channels),
            ConvBlock(middle_channels, middle_channels),
        )
        self.decode_1 = DecoderBlock(middle_channels, channels[1], channels[1])
        self.decode_0 = DecoderBlock(channels[1], channels[0], channels[0])
        self.center_heatmap = nn.Conv2d(channels[0], 1, kernel_size=1)

        pooled_dim = 2 * (channels[3] + channels[0]) + middle_channels
        self.global_head = nn.Sequential(
            nn.Linear(pooled_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 6),
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

    def _dino_features(
        self, rgb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        tokens = output["x_norm_patchtokens"]
        side = math.isqrt(tokens.shape[1])
        if side * side != tokens.shape[1]:
            raise ValueError(
                f"DINO patch count {tokens.shape[1]} is not a square feature map."
            )
        spatial = tokens.reshape(tokens.shape[0], side, side, tokens.shape[2])
        spatial = spatial.permute(0, 3, 1, 2).contiguous()
        spatial = self.dino_spatial_projection(spatial)
        pooled = torch.cat(
            [output["x_norm_clstoken"], tokens.mean(dim=1)], dim=1
        )
        return spatial, self.dino_global_projection(pooled)[:, :, None, None]

    def encode_image(
        self, rgb: torch.Tensor, observed_mask: torch.Tensor
    ) -> list[torch.Tensor]:
        if observed_mask.ndim == 3:
            observed_mask = observed_mask[:, None]
        pyramid = self.image_encoder(torch.cat([rgb, observed_mask], dim=1))
        spatial, pooled = self._dino_features(rgb)
        # Keeping every cached value four-dimensional lets the common training
        # loss expand one image encoding across all CAD candidates efficiently.
        return [*pyramid, spatial, pooled]

    def forward_encoded(
        self, image_encoding: list[torch.Tensor], rendered: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        features = self.extract_visual_embeddings_encoded(image_encoding, rendered)
        decoded = features.pop("decoded")
        heatmap_logits = self.center_heatmap(decoded).squeeze(1)
        center_raw = LightweightMatchingUNet._soft_argmax(heatmap_logits)
        global_output = self.global_head(features["candidate_visual_embedding"])
        return {
            "center_raw": center_raw,
            "center_heatmap_logits": heatmap_logits,
            "log_depth_raw": global_output[:, 0],
            "rotation_raw": global_output[:, 1:4],
            "confidence_logit": global_output[:, 4],
            "quality_raw": global_output[:, 5],
        }

    def extract_visual_embeddings_encoded(
        self, image_encoding: list[torch.Tensor], rendered: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return frozen frame and candidate RGB/mask/CAD evidence vectors.

        The frame vector combines pooled observed-image CNN features with the
        global DINO feature.  The candidate vector is exactly the feature fed
        to the trained matching U-Net global head, before pose prediction.
        """
        if len(image_encoding) != 6:
            raise ValueError("Expected four CNN levels plus spatial/global DINO features.")
        if image_encoding[0].shape[0] == 1 and rendered.shape[0] > 1:
            image_encoding = [
                feature.expand(rendered.shape[0], -1, -1, -1)
                for feature in image_encoding
            ]
        if image_encoding[0].shape[0] != rendered.shape[0]:
            raise ValueError("Image and rendered feature batch sizes differ.")
        image_pyramid = image_encoding[:4]
        dino_spatial, dino_global = image_encoding[4:]
        render_pyramid = self.render_encoder(rendered)
        matched = [
            block(image, cad)
            for block, image, cad in zip(
                self.matches, image_pyramid, render_pyramid
            )
        ]
        deep = F.interpolate(
            matched[3], size=matched[2].shape[-2:], mode="bilinear", align_corners=False
        )
        dino_spatial = F.interpolate(
            dino_spatial,
            size=matched[2].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded = self.decode_2(torch.cat([deep, matched[2], dino_spatial], dim=1))
        decoded = self.decode_1(decoded, matched[1])
        decoded = self.decode_0(decoded, matched[0])
        candidate_visual = torch.cat(
            [
                LightweightMatchingUNet._pool(matched[3]),
                LightweightMatchingUNet._pool(decoded),
                dino_global.flatten(1),
            ],
            dim=1,
        )
        frame_visual = torch.cat(
            [
                LightweightMatchingUNet._pool(image_pyramid[3]),
                dino_global.flatten(1),
            ],
            dim=1,
        )
        return {
            "frame_visual_embedding": frame_visual,
            "candidate_visual_embedding": candidate_visual,
            "decoded": decoded,
        }

    def forward(self, rgb, observed_mask, rendered):
        return self.forward_encoded(self.encode_image(rgb, observed_mask), rendered)

    def parameter_counts(self) -> dict[str, int]:
        return {
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "trainable": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
            "dino_trainable": sum(
                parameter.numel()
                for parameter in self.dino.parameters()
                if parameter.requires_grad
            ),
        }
