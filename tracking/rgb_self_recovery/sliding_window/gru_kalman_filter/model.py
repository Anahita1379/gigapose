"""Adaptive-gain causal GRU filter over SE(3) GigaPose measurements."""

from __future__ import annotations

import torch
from torch import nn

from .geometry import make_pose, so3_exp, so3_log


class GRUKalmanFilter(nn.Module):
    """A compact KalmanNet-style error-state filter with learned diagonal gains."""

    def __init__(
        self,
        context_dim: int,
        *,
        hidden_dim: int = 128,
        context_hidden: int = 32,
        translation_innovation_scale_m: float = 5.0,
        rotation_innovation_scale_rad: float = 0.785398,
        maximum_speed_mps: float = 80.0,
        maximum_angular_speed_radps: float = 6.283185,
        preserve_measurement_rotation: bool = False,
        use_fallback_heads: bool = False,
        fallback_hidden: int = 64,
        statistics: dict[str, torch.Tensor] | None = None,
    ):
        super().__init__()
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.context_hidden = int(context_hidden)
        self.translation_innovation_scale_m = float(translation_innovation_scale_m)
        self.rotation_innovation_scale_rad = float(rotation_innovation_scale_rad)
        self.maximum_speed_mps = float(maximum_speed_mps)
        self.maximum_angular_speed_radps = float(maximum_angular_speed_radps)
        self.preserve_measurement_rotation = bool(preserve_measurement_rotation)
        self.use_fallback_heads = bool(use_fallback_heads)
        self.fallback_hidden = int(fallback_hidden)
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, context_hidden),
            nn.LayerNorm(context_hidden),
            nn.SiLU(),
        )
        # innovation position/rotation, velocity/angular velocity, dt, context
        dynamic_dim = 6 + 6 + 1 + context_hidden
        self.gru_cell = nn.GRUCell(dynamic_dim, hidden_dim)
        self.gain_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 12),
        )
        # Hidden state, encoded image/measurement context, Kalman gains, and
        # two SE(3) discrepancies: filtered-vs-GigaPose and
        # selector-measurement-vs-GigaPose.
        fallback_input_dim = hidden_dim + context_hidden + 12 + 6 + 6
        self.translation_fallback_head = nn.Sequential(
            nn.Linear(fallback_input_dim, fallback_hidden),
            nn.SiLU(),
            nn.Linear(fallback_hidden, 1),
        )
        self.rotation_fallback_head = nn.Sequential(
            nn.Linear(fallback_input_dim, fallback_hidden),
            nn.SiLU(),
            nn.Linear(fallback_hidden, 1),
        )
        statistics = statistics or {}
        self.register_buffer(
            "context_mean",
            torch.as_tensor(statistics.get("context_mean", torch.zeros(context_dim))).float(),
        )
        self.register_buffer(
            "context_std",
            torch.as_tensor(statistics.get("context_std", torch.ones(context_dim))).float(),
        )

    def config(self) -> dict[str, int | float]:
        return {
            "context_dim": self.context_dim,
            "hidden_dim": self.hidden_dim,
            "context_hidden": self.context_hidden,
            "translation_innovation_scale_m": self.translation_innovation_scale_m,
            "rotation_innovation_scale_rad": self.rotation_innovation_scale_rad,
            "maximum_speed_mps": self.maximum_speed_mps,
            "maximum_angular_speed_radps": self.maximum_angular_speed_radps,
            "preserve_measurement_rotation": self.preserve_measurement_rotation,
            "use_fallback_heads": self.use_fallback_heads,
            "fallback_hidden": self.fallback_hidden,
        }

    def _fallback_logits(
        self,
        hidden,
        context,
        gain,
        filtered_position,
        filtered_rotation,
        measured_position,
        measured_rotation,
        baseline_position,
        baseline_rotation,
    ):
        filtered_delta = torch.cat(
            (
                (filtered_position - baseline_position)
                / self.translation_innovation_scale_m,
                so3_log(baseline_rotation.transpose(-1, -2) @ filtered_rotation)
                / self.rotation_innovation_scale_rad,
            ),
            dim=-1,
        )
        measurement_delta = torch.cat(
            (
                (measured_position - baseline_position)
                / self.translation_innovation_scale_m,
                so3_log(baseline_rotation.transpose(-1, -2) @ measured_rotation)
                / self.rotation_innovation_scale_rad,
            ),
            dim=-1,
        )
        features = torch.cat(
            (hidden, context, gain, filtered_delta, measurement_delta), dim=-1
        )
        return torch.cat(
            (
                self.translation_fallback_head(features),
                self.rotation_fallback_head(features),
            ),
            dim=-1,
        )

    def forward(
        self,
        measurement_pose,
        context_features,
        delta_time_s,
        frame_valid,
        baseline_pose=None,
    ):
        batch, steps = measurement_pose.shape[:2]
        dtype, device = measurement_pose.dtype, measurement_pose.device
        context = (context_features - self.context_mean) / self.context_std
        context = self.context_encoder(context)
        position = measurement_pose[:, 0, :3, 3]
        rotation = measurement_pose[:, 0, :3, :3]
        velocity = torch.zeros(batch, 3, dtype=dtype, device=device)
        angular_velocity = torch.zeros_like(velocity)
        hidden = torch.zeros(batch, self.hidden_dim, dtype=dtype, device=device)
        outputs = [make_pose(rotation, position)]
        zero_gain = torch.zeros(batch, 12, dtype=dtype, device=device)
        gains = [zero_gain]
        fallback_logits = []
        if self.use_fallback_heads:
            if baseline_pose is None:
                raise ValueError("baseline_pose is required when fallback heads are enabled")
            fallback_logits.append(self._fallback_logits(
                hidden,
                context[:, 0],
                zero_gain,
                position,
                rotation,
                measurement_pose[:, 0, :3, 3],
                measurement_pose[:, 0, :3, :3],
                baseline_pose[:, 0, :3, 3],
                baseline_pose[:, 0, :3, :3],
            ))
        for step in range(1, steps):
            valid = frame_valid[:, step]
            dt = delta_time_s[:, step].clamp(0.02, 1.0)
            predicted_position = position + velocity * dt[:, None]
            predicted_rotation = rotation @ so3_exp(angular_velocity * dt[:, None])
            measured_position = measurement_pose[:, step, :3, 3]
            measured_rotation = measurement_pose[:, step, :3, :3]
            position_innovation = measured_position - predicted_position
            rotation_innovation = so3_log(
                predicted_rotation.transpose(-1, -2) @ measured_rotation
            )
            dynamic = torch.cat(
                (
                    position_innovation / self.translation_innovation_scale_m,
                    rotation_innovation / self.rotation_innovation_scale_rad,
                    velocity / self.maximum_speed_mps,
                    angular_velocity / self.maximum_angular_speed_radps,
                    dt[:, None],
                    context[:, step],
                ),
                dim=-1,
            )
            proposed_hidden = self.gru_cell(dynamic, hidden)
            gain = torch.sigmoid(self.gain_head(proposed_hidden))
            pose_position_gain, pose_rotation_gain, velocity_gain, angular_gain = gain.split(3, dim=-1)
            updated_position = predicted_position + pose_position_gain * position_innovation
            learned_rotation = predicted_rotation @ so3_exp(
                pose_rotation_gain * rotation_innovation
            )
            updated_rotation = (
                measured_rotation
                if self.preserve_measurement_rotation
                else learned_rotation
            )
            updated_velocity = velocity + velocity_gain * position_innovation / dt[:, None]
            updated_angular = angular_velocity + angular_gain * rotation_innovation / dt[:, None]
            updated_velocity = self.maximum_speed_mps * torch.tanh(
                updated_velocity / self.maximum_speed_mps
            )
            updated_angular = self.maximum_angular_speed_radps * torch.tanh(
                updated_angular / self.maximum_angular_speed_radps
            )
            position = torch.where(valid[:, None], updated_position, position)
            rotation = torch.where(valid[:, None, None], updated_rotation, rotation)
            velocity = torch.where(valid[:, None], updated_velocity, velocity)
            angular_velocity = torch.where(valid[:, None], updated_angular, angular_velocity)
            hidden = torch.where(valid[:, None], proposed_hidden, hidden)
            gains.append(torch.where(valid[:, None], gain, torch.zeros_like(gain)))
            outputs.append(make_pose(rotation, position))
            if self.use_fallback_heads:
                fallback_logits.append(self._fallback_logits(
                    hidden,
                    context[:, step],
                    gains[-1],
                    position,
                    rotation,
                    measured_position,
                    measured_rotation,
                    baseline_pose[:, step, :3, 3],
                    baseline_pose[:, step, :3, :3],
                ))
        output = (torch.stack(outputs, dim=1), torch.stack(gains, dim=1))
        if self.use_fallback_heads:
            return (*output, torch.stack(fallback_logits, dim=1))
        return output
