"""Dual-encoder lightweight matching U-Net for residual pose recovery."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from tracking.rgb_self_recovery.model import ConvBlock


class PyramidEncoder(nn.Module):
    """Build a four-resolution feature pyramid, starting at half resolution."""

    def __init__(self, input_channels: int, width: int):
        super().__init__()
        channels = [width, width * 2, width * 4, width * 6]
        stages = []
        previous = input_channels
        for channel in channels:
            stages.append(
                nn.Sequential(
                    ConvBlock(previous, channel, stride=2),
                    ConvBlock(channel, channel),
                )
            )
            previous = channel
        self.stages = nn.ModuleList(stages)
        self.channels = channels

    def forward(self, value: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for stage in self.stages:
            value = stage(value)
            features.append(value)
        return features


class MatchBlock(nn.Module):
    """Compare image and rendered-CAD features at one spatial scale."""

    def __init__(self, channels: int):
        super().__init__()
        self.network = nn.Sequential(
            ConvBlock(channels * 4, channels),
            ConvBlock(channels, channels),
        )

    def forward(
        self, image: torch.Tensor, rendered: torch.Tensor
    ) -> torch.Tensor:
        return self.network(
            torch.cat(
                [image, rendered, torch.abs(image - rendered), image * rendered],
                dim=1,
            )
        )


class DecoderBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int):
        super().__init__()
        self.network = nn.Sequential(
            ConvBlock(input_channels + skip_channels, output_channels),
            ConvBlock(output_channels, output_channels),
        )

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(
            value, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.network(torch.cat([value, skip], dim=1))


class LightweightMatchingUNet(nn.Module):
    """Match RGB/mask and CAD geometry while preserving spatial disagreement.

    The center residual is obtained from a spatial probability map. Remaining
    pose and ranking outputs use pooled deep and decoded matching features.
    """

    def __init__(self, width: int = 24, hidden_dim: int = 256):
        super().__init__()
        if width <= 0 or hidden_dim <= 0:
            raise ValueError("Model dimensions must be positive.")
        self.width = int(width)
        self.hidden_dim = int(hidden_dim)
        self.image_encoder = PyramidEncoder(4, self.width)
        self.render_encoder = PyramidEncoder(5, self.width)
        channels = self.image_encoder.channels
        self.matches = nn.ModuleList([MatchBlock(channel) for channel in channels])
        self.decode_2 = DecoderBlock(channels[3], channels[2], channels[2])
        self.decode_1 = DecoderBlock(channels[2], channels[1], channels[1])
        self.decode_0 = DecoderBlock(channels[1], channels[0], channels[0])
        self.center_heatmap = nn.Conv2d(channels[0], 1, kernel_size=1)

        pooled_dim = 2 * (channels[3] + channels[0])
        self.global_head = nn.Sequential(
            nn.Linear(pooled_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 6),
        )

    @staticmethod
    def _soft_argmax(logits: torch.Tensor) -> torch.Tensor:
        batch, height, width = logits.shape
        probability = torch.softmax(logits.reshape(batch, -1), dim=-1).reshape(
            batch, height, width
        )
        y = torch.linspace(-1.0, 1.0, height, dtype=logits.dtype, device=logits.device)
        x = torch.linspace(-1.0, 1.0, width, dtype=logits.dtype, device=logits.device)
        center_x = (probability * x.view(1, 1, width)).sum(dim=(1, 2))
        center_y = (probability * y.view(1, height, 1)).sum(dim=(1, 2))
        normalized = torch.stack([center_x, center_y], dim=-1)
        # decode_outputs applies tanh to center_raw. Inverse tanh preserves the
        # heatmap expectation while retaining the common checkpoint interface.
        return torch.atanh(torch.clamp(normalized, -0.999, 0.999))

    @staticmethod
    def _pool(value: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                F.adaptive_avg_pool2d(value, 1).flatten(1),
                F.adaptive_max_pool2d(value, 1).flatten(1),
            ],
            dim=1,
        )

    def encode_image(
        self,
        rgb: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        if observed_mask.ndim == 3:
            observed_mask = observed_mask[:, None]
        return self.image_encoder(torch.cat([rgb, observed_mask], dim=1))

    def forward_encoded(
        self,
        image_pyramid: list[torch.Tensor],
        rendered: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        render_pyramid = self.render_encoder(rendered)
        if image_pyramid[0].shape[0] == 1 and rendered.shape[0] > 1:
            image_pyramid = [
                feature.expand(rendered.shape[0], -1, -1, -1)
                for feature in image_pyramid
            ]
        if image_pyramid[0].shape[0] != rendered.shape[0]:
            raise ValueError("Image and rendered feature batch sizes differ.")
        matched = [
            block(image, cad)
            for block, image, cad in zip(
                self.matches, image_pyramid, render_pyramid
            )
        ]
        decoded = self.decode_2(matched[3], matched[2])
        decoded = self.decode_1(decoded, matched[1])
        decoded = self.decode_0(decoded, matched[0])
        heatmap_logits = self.center_heatmap(decoded).squeeze(1)
        center_raw = self._soft_argmax(heatmap_logits)
        global_output = self.global_head(
            torch.cat([self._pool(matched[3]), self._pool(decoded)], dim=1)
        )
        return {
            "center_raw": center_raw,
            "center_heatmap_logits": heatmap_logits,
            "log_depth_raw": global_output[:, 0],
            "rotation_raw": global_output[:, 1:4],
            "confidence_logit": global_output[:, 4],
            "quality_raw": global_output[:, 5],
        }

    def forward(
        self,
        rgb: torch.Tensor,
        observed_mask: torch.Tensor,
        rendered: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.forward_encoded(self.encode_image(rgb, observed_mask), rendered)

    def parameter_counts(self) -> dict[str, int]:
        return {
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "trainable": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
        }
