"""Focused tests for the isolated matching U-Net package."""

from __future__ import annotations

import torch

from tracking.rgb_self_recovery.model import decode_outputs

from .inference import load_predictor
from .model import LightweightMatchingUNet
from .train import heatmap_kl_loss


def test_model_shapes_bounds_and_lightweight_parameter_count():
    model = LightweightMatchingUNet().eval()
    with torch.no_grad():
        output = model(
            torch.rand(2, 3, 64, 64),
            torch.rand(2, 1, 64, 64),
            torch.rand(2, 5, 64, 64),
        )
        decoded = decode_outputs(
            output,
            max_center_px=56.0,
            max_log_depth=0.6,
            max_rotation_deg=70.0,
        )
    assert output["center_heatmap_logits"].shape == (2, 32, 32)
    assert output["rotation_raw"].shape == (2, 3)
    assert decoded["center_px"].shape == (2, 2)
    assert torch.all(torch.abs(decoded["center_px"]) <= 56.0)
    assert model.parameter_counts()["total"] < 3_642_408


def test_cached_image_pyramid_matches_regular_forward():
    torch.manual_seed(4)
    model = LightweightMatchingUNet().eval()
    rgb = torch.rand(1, 3, 64, 64)
    mask = torch.rand(1, 1, 64, 64)
    rendered = torch.rand(3, 5, 64, 64)
    with torch.no_grad():
        cached = model.forward_encoded(model.encode_image(rgb, mask), rendered)
        repeated = model(
            rgb.expand(3, -1, -1, -1),
            mask.expand(3, -1, -1, -1),
            rendered,
        )
    for key in repeated:
        # Expanded and physically repeated batches can select slightly different
        # CPU convolution kernels; their discrepancy remains at float32 noise.
        assert torch.allclose(cached[key], repeated[key], atol=1e-5)


def test_heatmap_loss_is_differentiable():
    logits = torch.zeros(2, 16, 16, requires_grad=True)
    loss = heatmap_kl_loss(
        logits,
        torch.tensor([[0.0, 0.0], [20.0, -10.0]]),
        torch.tensor([True, True]),
        max_center_px=56.0,
        sigma_px=4.0,
    )
    assert torch.isfinite(loss)
    assert loss > 0
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_checkpoint_round_trip(tmp_path):
    model = LightweightMatchingUNet(width=8, hidden_dim=32)
    path = tmp_path / "model.ckpt"
    torch.save(
        {
            "format": "rgb_render_self_recovery_matching_unet_v1",
            "model_state": model.state_dict(),
            "model_config": {"width": 8, "hidden_dim": 32},
            "inference_config": {
                "crop_size": 64,
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
    assert isinstance(predictor.model, LightweightMatchingUNet)
    assert predictor.config["crop_size"] == 64
