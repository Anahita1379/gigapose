"""Checkpoint routing for candidate export."""

from __future__ import annotations

from pathlib import Path

import torch

from tracking.rgb_self_recovery.inference import RGBSelfRecoveryPredictor
from tracking.rgb_self_recovery.sliding_window.dino_inference import (
    load_predictor as load_dino_predictor,
)
from tracking.rgb_self_recovery.sliding_window.dino_matching_unet.inference import (
    load_predictor as load_dino_matching_predictor,
)
from tracking.rgb_self_recovery.sliding_window.matching_unet.inference import (
    load_predictor as load_matching_predictor,
)


def load_candidate_predictor(path: Path, device: str):
    payload = torch.load(Path(path), map_location="cpu")
    format_name = payload.get("format")
    if format_name == "rgb_render_self_recovery_v1":
        return RGBSelfRecoveryPredictor.load(path, device)
    if format_name == "rgb_render_self_recovery_dino_v1":
        return load_dino_predictor(path, device)
    if format_name == "rgb_render_self_recovery_matching_unet_v1":
        return load_matching_predictor(path, device)
    if format_name == "rgb_render_self_recovery_dino_matching_unet_v1":
        return load_dino_matching_predictor(path, device)
    raise ValueError(f"Unsupported candidate checkpoint format {format_name!r}: {path}")

