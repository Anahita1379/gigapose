"""Training-only depth heads and a depth-conditioned teacher."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from tracking.rgb_self_recovery.model import ConvBlock
from tracking.rgb_self_recovery.sliding_window.dino_matching_unet.model import (
    DINOPatchMatchingUNet,
)
from tracking.rgb_self_recovery.sliding_window.matching_unet.model import DecoderBlock


class AuxiliaryDepthStudent(DINOPatchMatchingUNet):
    """RGB-only inference model with a training-only dense depth head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        channels = self.image_encoder.channels
        self.depth_head_deep = nn.Sequential(
            ConvBlock(channels[3] + channels[2], channels[2]),
            ConvBlock(channels[2], channels[2]),
        )
        self.depth_head_1 = DecoderBlock(channels[2], channels[1], channels[1])
        self.depth_head_0 = DecoderBlock(channels[1], channels[0], channels[0])
        self.depth_head_output = nn.Conv2d(channels[0], 1, 1)

    def predict_auxiliary_depth(self, image_encoding):
        pyramid = image_encoding[:4]
        dino_spatial = F.interpolate(
            image_encoding[4], size=pyramid[3].shape[-2:],
            mode="bilinear", align_corners=False,
        )
        value = self.depth_head_deep(torch.cat([pyramid[3], dino_spatial], dim=1))
        value = self.depth_head_1(value, pyramid[1])
        value = self.depth_head_0(value, pyramid[0])
        return self.depth_head_output(value)

    def inference_state_dict(self):
        return {
            name: value
            for name, value in self.state_dict().items()
            if not name.startswith("depth_head_")
        }


class DepthConditionedTeacher(DINOPatchMatchingUNet):
    """Teacher that receives metric depth; never used for deployment."""

    def __init__(self, *args, maximum_depth_m: float = 200.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.maximum_depth_m = float(maximum_depth_m)
        channels = self.image_encoder.channels
        stages = []
        previous = 2
        for channel in channels:
            stages.append(nn.Sequential(
                ConvBlock(previous, channel, stride=2),
                ConvBlock(channel, channel),
            ))
            previous = channel
        self.depth_encoder = nn.ModuleList(stages)
        self.depth_fusion = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(2 * channel, channel, 1), nn.SiLU()) for channel in channels]
        )

    def encode_image_with_depth(self, rgb, observed_mask, depth_m, depth_valid):
        image_encoding = super().encode_image(rgb, observed_mask)
        normalized = torch.log1p(torch.clamp(depth_m, min=0.0)) / math.log1p(
            self.maximum_depth_m
        )
        value = torch.cat([normalized, depth_valid.float()], dim=1)
        depth_features = []
        for stage in self.depth_encoder:
            value = stage(value)
            depth_features.append(value)
        fused = [
            block(torch.cat([image, depth], dim=1))
            for block, image, depth in zip(
                self.depth_fusion, image_encoding[:4], depth_features
            )
        ]
        return [*fused, *image_encoding[4:]]

