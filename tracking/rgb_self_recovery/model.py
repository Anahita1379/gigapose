"""RGB/render-aware residual pose verifier and refiner."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1):
        super().__init__()
        groups = min(16, output_channels)
        while output_channels % groups:
            groups -= 1
        self.conv = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm = nn.GroupNorm(groups, output_channels)
        self.activation = nn.SiLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(inputs)))


class Encoder(nn.Module):
    def __init__(self, input_channels: int, width: int = 32):
        super().__init__()
        channels = [width, width * 2, width * 4, width * 6]
        layers = []
        previous = input_channels
        for channel in channels:
            layers.extend(
                [
                    ConvBlock(previous, channel, stride=2),
                    ConvBlock(channel, channel),
                ]
            )
            previous = channel
        self.network = nn.Sequential(*layers)
        self.output_channels = channels[-1]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class RGBRenderRecoveryNet(nn.Module):
    """Compare an RGB/mask crop with a candidate CAD geometry rendering.

    Output residuals use image-center/log-depth coordinates because these are
    better conditioned than raw XYZ across a large distance range. Rotation is
    a camera-frame left SO(3) tangent correction.
    """

    def __init__(self, width: int = 32, hidden_dim: int = 384):
        super().__init__()
        self.image_encoder = Encoder(4, width)
        self.render_encoder = Encoder(5, width)
        feature_channels = self.image_encoder.output_channels
        self.fusion = nn.Sequential(
            ConvBlock(feature_channels * 4, feature_channels),
            ConvBlock(feature_channels, feature_channels),
        )
        pooled_dim = feature_channels * 2
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 8),
        )

    def forward(
        self,
        rgb: torch.Tensor,
        observed_mask: torch.Tensor,
        rendered: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if observed_mask.ndim == 3:
            observed_mask = observed_mask[:, None]
        image_input = torch.cat([rgb, observed_mask], dim=1)
        image_features = self.image_encoder(image_input)
        render_features = self.render_encoder(rendered)
        fused = torch.cat(
            [
                image_features,
                render_features,
                torch.abs(image_features - render_features),
                image_features * render_features,
            ],
            dim=1,
        )
        fused = self.fusion(fused)
        pooled = torch.cat(
            [
                F.adaptive_avg_pool2d(fused, 1).flatten(1),
                F.adaptive_max_pool2d(fused, 1).flatten(1),
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


def bounded_vector(raw: torch.Tensor, maximum_norm: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    direction = raw / torch.clamp(norm, min=1e-8)
    return direction * torch.tanh(norm) * float(maximum_norm)


def decode_outputs(
    output: dict[str, torch.Tensor],
    *,
    max_center_px: float,
    max_log_depth: float,
    max_rotation_deg: float,
) -> dict[str, torch.Tensor]:
    return {
        "center_px": torch.tanh(output["center_raw"]) * float(max_center_px),
        "log_depth": torch.tanh(output["log_depth_raw"]) * float(max_log_depth),
        "rotation_rad": bounded_vector(
            output["rotation_raw"], math.radians(float(max_rotation_deg))
        ),
        "confidence": torch.sigmoid(output["confidence_logit"]),
        "quality": F.softplus(output["quality_raw"]),
    }


def skew_torch(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], dim=-1
    ).reshape(*vector.shape[:-1], 3, 3)


def so3_exp_torch(rotvec: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    theta_sq = theta.square()
    A = torch.where(
        theta > 1e-4,
        torch.sin(theta) / torch.clamp(theta, min=1e-8),
        1.0 - theta_sq / 6.0 + theta_sq.square() / 120.0,
    )
    B = torch.where(
        theta > 1e-4,
        (1.0 - torch.cos(theta)) / torch.clamp(theta_sq, min=1e-8),
        0.5 - theta_sq / 24.0 + theta_sq.square() / 720.0,
    )
    K = skew_torch(rotvec)
    identity = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    identity = identity.expand(*rotvec.shape[:-1], 3, 3)
    return identity + A[..., None] * K + B[..., None] * (K @ K)


def rotation_geodesic_atan2(
    predicted_rotvec: torch.Tensor,
    target_rotvec: torch.Tensor,
) -> torch.Tensor:
    predicted = so3_exp_torch(predicted_rotvec)
    target = so3_exp_torch(target_rotvec)
    relative = predicted @ target.transpose(-1, -2)
    vector = torch.stack(
        [
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ],
        dim=-1,
    )
    sine = 0.5 * torch.linalg.vector_norm(vector, dim=-1)
    cosine = 0.5 * (
        relative[..., 0, 0]
        + relative[..., 1, 1]
        + relative[..., 2, 2]
        - 1.0
    )
    return torch.atan2(sine, torch.clamp(cosine, -1.0, 1.0))
