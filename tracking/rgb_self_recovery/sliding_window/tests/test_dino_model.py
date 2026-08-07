import torch
from torch import nn

from tracking.rgb_self_recovery.sliding_window import dino_model


class _TinyDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(3, 384)
        self.blocks = nn.ModuleList([nn.Linear(384, 384) for _ in range(2)])
        self.norm = nn.LayerNorm(384)

    def forward_features(self, value):
        pooled = value.mean(dim=(-2, -1))
        features = self.stem(pooled)
        for block in self.blocks:
            features = torch.tanh(block(features))
        features = self.norm(features)
        return {
            "x_norm_clstoken": features,
            "x_norm_patchtokens": features[:, None].expand(-1, 4, -1),
        }


def _model(monkeypatch, mode):
    monkeypatch.setattr(
        dino_model, "load_dinov2_model", lambda *args, **kwargs: _TinyDINO()
    )
    return dino_model.DINORGBRenderRecoveryNet(
        width=4,
        hidden_dim=16,
        dino_mode=mode,
        dino_model="dinov2_vits14",
        dino_input_size=28,
    )


def test_frozen_dino_has_no_trainable_backbone_parameters(monkeypatch):
    model = _model(monkeypatch, "frozen")
    assert model.trainable_parameter_counts()["dino_trainable"] == 0
    output = model(
        torch.rand(2, 3, 32, 32),
        torch.ones(2, 1, 32, 32),
        torch.rand(2, 5, 32, 32),
    )
    assert output["rotation_raw"].shape == (2, 3)


def test_last_block_only_unfreezes_final_block_and_norm(monkeypatch):
    model = _model(monkeypatch, "last_block")
    assert model.trainable_parameter_counts()["dino_trainable"] > 0
    assert not any(parameter.requires_grad for parameter in model.dino.stem.parameters())
    assert not any(
        parameter.requires_grad for parameter in model.dino.blocks[0].parameters()
    )
    assert all(
        parameter.requires_grad for parameter in model.dino.blocks[-1].parameters()
    )
