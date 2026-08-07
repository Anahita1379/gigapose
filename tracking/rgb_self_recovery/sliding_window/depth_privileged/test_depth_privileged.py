"""Tests for training-only depth and RGB-only deployment."""

from __future__ import annotations

from types import SimpleNamespace
import json
import sys

import numpy as np
import torch
from torch import nn

from tracking.rgb_self_recovery.sliding_window.dino_matching_unet import model as dino_model
from tracking.rgb_self_recovery.sliding_window.dino_matching_unet.inference import load_predictor
from tracking.rgb_self_recovery.generate_dataset import (
    RecoveryShardWriter,
    encode_jpeg,
    encode_npz,
)

from .losses import (
    auxiliary_depth_objective,
    distillation_objective,
    forward_grouped,
    supervised_objective,
)
from .model import AuxiliaryDepthStudent, DepthConditionedTeacher
from .prepare_data import crop_depth, main as prepare_main, source_fields
from .dataset import PrivilegedDepthRecoveryDataset
from .run_student import build_parser as build_student_run_parser


class TinyDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(3, 384)
        self.blocks = nn.ModuleList([nn.Linear(384, 384)])
        self.norm = nn.LayerNorm(384)

    def forward_features(self, value):
        feature = self.norm(torch.tanh(self.blocks[0](self.stem(value.mean((-2, -1))))))
        return {
            "x_norm_clstoken": feature,
            "x_norm_patchtokens": feature[:, None].expand(-1, 4, -1),
        }


def patch_dino(monkeypatch):
    monkeypatch.setattr(
        dino_model, "load_dinov2_model", lambda *args, **kwargs: TinyDINO()
    )


def kwargs():
    return dict(
        width=4, hidden_dim=16, dino_mode="frozen",
        dino_model="dinov2_vits14", dino_input_size=28,
    )


def test_auxiliary_student_depth_head_is_training_only(monkeypatch, tmp_path):
    patch_dino(monkeypatch)
    model = AuxiliaryDepthStudent(**kwargs(), dino_pretrained=False).eval()
    rgb, mask = torch.rand(1, 3, 32, 32), torch.ones(1, 1, 32, 32)
    encoding = model.encode_image(rgb, mask)
    assert model.predict_auxiliary_depth(encoding).shape == (1, 1, 16, 16)
    state = model.inference_state_dict()
    assert not any(name.startswith("depth_head_") for name in state)
    path = tmp_path / "student.ckpt"
    torch.save({
        "format": "rgb_render_self_recovery_dino_matching_unet_v1",
        "model_state": state,
        "model_config": {**kwargs(), "dino_pretrained": False},
        "inference_config": {
            "crop_size": 32, "crop_scale": 2.5,
            "max_center_crop_px": 56.0, "max_log_depth": 0.6,
            "max_rotation_deg": 70.0, "uses_observed_depth": False,
        },
    }, path)
    predictor = load_predictor(path, "cpu")
    assert predictor.config["uses_observed_depth"] is False
    assert not hasattr(predictor.model, "depth_head_output")


def test_teacher_encoding_responds_to_privileged_depth(monkeypatch):
    patch_dino(monkeypatch)
    teacher = DepthConditionedTeacher(
        **kwargs(), dino_pretrained=False, maximum_depth_m=100.0
    ).eval()
    rgb, mask = torch.rand(1, 3, 32, 32), torch.ones(1, 1, 32, 32)
    valid = torch.ones(1, 1, 32, 32, dtype=torch.bool)
    first = teacher.encode_image_with_depth(rgb, mask, torch.ones_like(mask), valid)
    second = teacher.encode_image_with_depth(rgb, mask, 20 * torch.ones_like(mask), valid)
    assert not torch.allclose(first[0], second[0])


def test_auxiliary_and_distillation_losses_are_differentiable(monkeypatch):
    patch_dino(monkeypatch)
    student = AuxiliaryDepthStudent(**kwargs(), dino_pretrained=False)
    rgb, mask = torch.rand(1, 3, 32, 32), torch.ones(1, 1, 32, 32)
    encoding = student.encode_image(rgb, mask)
    batch = {
        "observed_depth_m": torch.full((1, 1, 32, 32), 10.0),
        "observed_depth_valid": torch.ones(1, 1, 32, 32, dtype=torch.bool),
    }
    auxiliary = auxiliary_depth_objective(
        student, encoding, batch,
        SimpleNamespace(maximum_depth_m=100.0), "cpu"
    )
    raw = {
        "center_raw": torch.zeros(1, 2, requires_grad=True),
        "log_depth_raw": torch.zeros(1, requires_grad=True),
        "rotation_raw": torch.zeros(1, 3, requires_grad=True),
        "confidence_logit": torch.zeros(1, requires_grad=True),
        "quality_raw": torch.zeros(1, requires_grad=True),
    }
    teacher_raw = {name: value.detach() + 0.5 for name, value in raw.items()}
    distill, _, _ = distillation_objective(
        raw, encoding, teacher_raw, [value.detach() + 0.1 for value in encoding],
        SimpleNamespace(distillation_output_weight=0.5, distillation_feature_weight=0.1),
    )
    (auxiliary + distill).backward()
    assert any(parameter.grad is not None for parameter in student.parameters())


def test_depth_path_fields_and_crop():
    fields = source_fields({"source_key": "run_a__rear__000123"})
    assert fields == {
        "source_key": "run_a__rear__000123", "source_run": "run_a",
        "camera_id": "rear", "frame_id": 123,
    }
    depth = np.arange(64, dtype=np.float32).reshape(8, 8)
    cropped, valid = crop_depth(depth, np.asarray([2, 2, 4, 4]), depth > 0)
    assert cropped.shape == (4, 4)
    assert valid.shape == (4, 4)


def test_full_student_and_teacher_objectives(monkeypatch):
    patch_dino(monkeypatch)
    student = AuxiliaryDepthStudent(**kwargs(), dino_pretrained=False)
    teacher = DepthConditionedTeacher(
        **kwargs(), dino_pretrained=False, maximum_depth_m=100.0
    )
    batch_size, candidates, side = 1, 3, 32
    batch = {
        "rgb": torch.rand(batch_size, 3, side, side),
        "observed_mask": torch.ones(batch_size, 1, side, side),
        "rendered": torch.rand(batch_size, candidates, 5, side, side),
        "center_targets": torch.zeros(batch_size, candidates, 2),
        "log_depth_targets": torch.zeros(batch_size, candidates),
        "rotation_targets": torch.zeros(batch_size, candidates, 3),
        "correctable_targets": torch.ones(batch_size, candidates, dtype=torch.bool),
        "confidence_targets": torch.ones(batch_size, candidates),
        "quality_targets": torch.arange(candidates).float()[None],
        "observed_depth_m": torch.full((batch_size, 1, side, side), 10.0),
        "observed_depth_valid": torch.ones(batch_size, 1, side, side, dtype=torch.bool),
    }
    args = SimpleNamespace(
        max_center_crop_px=56.0, max_log_depth=0.6, max_rotation_deg=70.0,
        heatmap_sigma_px=4.0, ranking_margin=0.1,
        anti_flip_margin=0.25, anti_flip_min_angle_deg=150.0,
        anti_flip_training=True, anti_flip_weight=1.0,
        center_weight=1.0, log_depth_weight=1.0, rotation_weight=1.0,
        confidence_weight=0.5, quality_weight=0.5, ranking_weight=0.5,
        heatmap_weight=0.1,
    )
    student_raw, _, b, c = forward_grouped(student, batch, "cpu")
    loss, metrics = supervised_objective(student_raw, batch, args, "cpu", b, c)
    teacher_raw, _, _, _ = forward_grouped(
        teacher, batch, "cpu", privileged_depth=True
    )
    assert torch.isfinite(loss)
    assert metrics["rotation_error_deg"] >= 0
    assert teacher_raw["rotation_raw"].shape == (candidates, 3)


def test_teacher_student_inference_inherits_sequence_controls():
    parser = build_student_run_parser("teacher_student")
    required = [
        "--predictions", "predictions.csv",
        "--dataset-dir", "dataset",
        "--checkpoint", "student.ckpt",
        "--output-dir", "output",
    ]
    defaults = parser.parse_args(required)
    assert defaults.sequence_aware is True
    assert defaults.sequence_max_frame_gap == 1
    assert defaults.sequence_max_time_gap_s == 0.5
    assert defaults.window_size == 5
    assert defaults.per_frame is False

    per_frame = parser.parse_args([*required, "--window-size", "1", "--per-frame"])
    assert per_frame.window_size == 1
    assert per_frame.per_frame is True


def test_prepare_data_round_trip(monkeypatch, tmp_path):
    source, depth_root, output = tmp_path / "source", tmp_path / "depth", tmp_path / "output"
    source.mkdir()
    manifest = {
        "format": "rgb_render_self_recovery_v1", "uses_observed_depth": False,
        "groups": 1, "crop_size": 8, "crop_scale": 2.5,
        "assetto_shaded_side_crop": {"enabled": False, "pixels_per_side": 0},
    }
    (source / "manifest.json").write_text(json.dumps(manifest))
    writer = RecoveryShardWriter(source, 10)
    candidates = 3
    values = dict(
        observed_mask=np.ones((8, 8), np.uint8),
        rendered=np.zeros((candidates, 5, 8, 8), np.uint8),
        center_targets=np.zeros((candidates, 2), np.float32),
        log_depth_targets=np.zeros(candidates, np.float32),
        rotation_targets=np.zeros((candidates, 3), np.float32),
        correctable_targets=np.ones(candidates, bool),
        confidence_targets=np.ones(candidates, np.float32),
        quality_targets=np.arange(candidates, dtype=np.float32),
        crop=np.asarray([0, 0, 8, 8], np.float32),
    )
    writer.write(
        "sample", encode_jpeg(np.zeros((8, 8, 3), np.uint8)), encode_npz(**values),
        {"source_key": "run_a__rear__000123"},
    )
    writer.close()
    path = depth_root / "run_a/rear/depth"
    path.mkdir(parents=True)
    np.save(path / "000123.npy", np.full((8, 8), 10.0, np.float32))
    monkeypatch.setattr(sys, "argv", [
        "prepare_data", "--input-data", str(source), "--depth-root", str(depth_root),
        "--output-dir", str(output),
    ])
    prepare_main()
    dataset = PrivilegedDepthRecoveryDataset(output, training=False)
    sample = next(iter(dataset))
    assert sample["observed_depth_m"].shape == (1, 8, 8)
    assert torch.all(sample["observed_depth_valid"])
