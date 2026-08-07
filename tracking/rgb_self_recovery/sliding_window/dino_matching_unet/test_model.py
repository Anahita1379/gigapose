"""Tests for the isolated DINO patch matching U-Net."""

from __future__ import annotations

import torch
from torch import nn

from . import model as model_module
from .inference import load_predictor
from .train import initialize_from_checkpoint


class TinyDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(3, 384)
        self.blocks = nn.ModuleList([nn.Linear(384, 384) for _ in range(2)])
        self.norm = nn.LayerNorm(384)

    def forward_features(self, value):
        pooled = value.mean(dim=(-2, -1))
        feature = self.stem(pooled)
        for block in self.blocks:
            feature = torch.tanh(block(feature))
        feature = self.norm(feature)
        return {
            "x_norm_clstoken": feature,
            "x_norm_patchtokens": feature[:, None].expand(-1, 4, -1),
        }


def make_model(monkeypatch, mode="frozen"):
    monkeypatch.setattr(
        model_module, "load_dinov2_model", lambda *args, **kwargs: TinyDINO()
    )
    return model_module.DINOPatchMatchingUNet(
        width=4,
        hidden_dim=16,
        dino_mode=mode,
        dino_model="dinov2_vits14",
        dino_input_size=28,
    )


def test_frozen_dino_patch_model_shapes_and_cache(monkeypatch):
    torch.manual_seed(2)
    model = make_model(monkeypatch).eval()
    assert model.parameter_counts()["dino_trainable"] == 0
    rgb = torch.rand(1, 3, 32, 32)
    mask = torch.ones(1, 1, 32, 32)
    renders = torch.rand(3, 5, 32, 32)
    with torch.no_grad():
        encoding = model.encode_image(rgb, mask)
        cached = model.forward_encoded(encoding, renders)
        repeated = model(
            rgb.expand(3, -1, -1, -1),
            mask.expand(3, -1, -1, -1),
            renders,
        )
    assert len(encoding) == 6
    assert encoding[4].shape == (1, 16, 2, 2)
    assert cached["center_heatmap_logits"].shape == (3, 16, 16)
    assert cached["rotation_raw"].shape == (3, 3)
    for key in repeated:
        assert torch.allclose(cached[key], repeated[key], atol=1e-5)


def test_last_block_mode_only_unfreezes_expected_dino_layers(monkeypatch):
    model = make_model(monkeypatch, "last_block")
    assert model.parameter_counts()["dino_trainable"] > 0
    assert not any(parameter.requires_grad for parameter in model.dino.stem.parameters())
    assert not any(
        parameter.requires_grad for parameter in model.dino.blocks[0].parameters()
    )
    assert all(
        parameter.requires_grad for parameter in model.dino.blocks[-1].parameters()
    )


def test_checkpoint_round_trip(monkeypatch, tmp_path):
    model = make_model(monkeypatch)
    path = tmp_path / "dino_matching.ckpt"
    torch.save(
        {
            "format": "rgb_render_self_recovery_dino_matching_unet_v1",
            "model_state": model.state_dict(),
            "model_config": {
                "width": 4,
                "hidden_dim": 16,
                "dino_mode": "frozen",
                "dino_model": "dinov2_vits14",
                "dino_input_size": 28,
            },
            "inference_config": {
                "crop_size": 32,
                "crop_scale": 1.5,
                "max_center_crop_px": 56.0,
                "max_log_depth": 0.6,
                "max_rotation_deg": 70.0,
                "uses_observed_depth": False,
            },
        },
        path,
    )
    predictor = load_predictor(path, "cpu")
    assert isinstance(predictor.model, model_module.DINOPatchMatchingUNet)
    assert predictor.config["crop_size"] == 32


def test_last_block_initializes_from_frozen_checkpoint(monkeypatch, tmp_path):
    frozen = make_model(monkeypatch, "frozen")
    path = tmp_path / "frozen.ckpt"
    torch.save(
        {
            "format": "rgb_render_self_recovery_dino_matching_unet_v1",
            "model_state": frozen.state_dict(),
            "model_config": {
                "width": 4,
                "hidden_dim": 16,
                "dino_mode": "frozen",
                "dino_model": "dinov2_vits14",
                "dino_input_size": 28,
            },
        },
        path,
    )
    target = make_model(monkeypatch, "last_block")
    payload = initialize_from_checkpoint(
        target,
        path,
        width=4,
        hidden_dim=16,
        dino_model="dinov2_vits14",
        dino_input_size=28,
    )
    assert payload["model_config"]["dino_mode"] == "frozen"
    for key, value in frozen.state_dict().items():
        assert torch.equal(value, target.state_dict()[key])
    assert target.parameter_counts()["dino_trainable"] > 0
