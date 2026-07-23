from __future__ import annotations

import io
import json
import tarfile
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from tracking.geometry import (
    rotate_pose,
    rotation_error_deg,
    shift_projected_center,
    so3_exp,
)
from tracking.rgb_self_recovery.dataset import RGBRenderRecoveryDataset
from tracking.rgb_self_recovery.inference import (
    RGBSelfRecoveryPredictor,
    RecoveryProposal,
    RecoveryResult,
)
from tracking.rgb_self_recovery.model import (
    RGBRenderRecoveryNet,
    decode_outputs,
    rotation_geodesic_atan2,
)
from tracking.rgb_self_recovery.render_inputs import (
    apply_recovery_delta,
    crop_spec_from_bbox,
    decode_render_channels,
    encode_render_channels,
    hide_rendered_occluders,
    recovery_targets,
    render_candidate_channels,
)
from tracking.rgb_self_recovery.run import (
    broad_recovery_proposals,
    select_with_orientation_gates,
)
from tracking.rgb_self_recovery.train import anti_flip_quality_loss
from tracking.types import Detection, FrameData, Track, TrackMode


class SquareRenderer:
    def render(self, pose_m, K, image_shape):
        height, width = image_shape
        translation = np.asarray(pose_m[:3, 3], dtype=float)
        projected = np.asarray(K, dtype=float) @ translation
        center = projected[:2] / projected[2]
        radius = max(2, int(round(18.0 / translation[2])))
        x0 = max(0, int(round(center[0])) - radius)
        x1 = min(width, int(round(center[0])) + radius + 1)
        y0 = max(0, int(round(center[1])) - radius)
        y1 = min(height, int(round(center[1])) + radius + 1)
        mask = np.zeros((height, width), dtype=bool)
        depth = np.zeros((height, width), dtype=np.float32)
        mask[y0:y1, x0:x1] = True
        depth[mask] = float(translation[2])
        return mask, depth


def test_crop_intrinsics_map_projected_points():
    K = np.asarray([[700.0, 0.0, 320.0], [0.0, 710.0, 180.0], [0.0, 0.0, 1.0]])
    crop = crop_spec_from_bbox(np.asarray([100.0, 50.0, 120.0, 80.0]), output_size=128)
    point = np.asarray([0.3, -0.1, 5.0])
    original = K @ point
    original = original[:2] / original[2]
    expected = crop.homography @ np.asarray([*original, 1.0])
    transformed = crop.transform_intrinsics(K) @ point
    transformed = transformed[:2] / transformed[2]
    np.testing.assert_allclose(transformed, expected[:2], atol=1e-6)


def test_recovery_targets_round_trip_pose():
    K = np.asarray([[800.0, 0.0, 320.0], [0.0, 800.0, 180.0], [0.0, 0.0, 1.0]])
    ground_truth = np.eye(4)
    ground_truth[:3, 3] = [0.25, -0.08, 7.0]
    candidate = shift_projected_center(
        ground_truth,
        K,
        np.asarray([31.0, -18.0]),
        0.22,
    )
    candidate = rotate_pose(candidate, np.asarray([0.12, -0.08, 0.18]), side="left")
    delta_uv, delta_log_depth, delta_rotation = recovery_targets(candidate, ground_truth, K)
    recovered = apply_recovery_delta(
        candidate, K, delta_uv, delta_log_depth, delta_rotation
    )
    np.testing.assert_allclose(recovered[:3, 3], ground_truth[:3, 3], atol=1e-7)
    np.testing.assert_allclose(recovered[:3, :3], ground_truth[:3, :3], atol=1e-7)


def test_render_channel_encoding_and_decoding():
    renderer = SquareRenderer()
    pose = np.eye(4)
    pose[2, 3] = 4.0
    K = np.asarray([[90.0, 0.0, 32.0], [0.0, 90.0, 32.0], [0.0, 0.0, 1.0]])
    encoded, mask, depth = render_candidate_channels(renderer, pose, K, 64)
    decoded = decode_render_channels(encoded[None])[0]
    assert encoded.shape == (5, 64, 64)
    assert decoded.shape == (5, 64, 64)
    assert mask.any()
    assert np.all(depth[mask] == 4.0)
    np.testing.assert_array_equal(decoded[0] > 0.5, mask)
    assert np.all(decoded[1:, ~mask] == 0.0)


def test_known_occluders_are_removed_from_render_input():
    mask = np.ones((8, 8), dtype=bool)
    depth = np.full((8, 8), 4.0, dtype=np.float32)
    K = np.asarray([[40.0, 0.0, 4.0], [0.0, 40.0, 4.0], [0.0, 0.0, 1.0]])
    encoded = encode_render_channels(mask, depth, K, 4.0)
    occluder = np.zeros((8, 8), dtype=bool)
    occluder[:, :3] = True
    hidden = hide_rendered_occluders(encoded, occluder)
    decoded = decode_render_channels(hidden)
    assert np.all(decoded[:, :, :3] == 0.0)
    assert np.all(decoded[0, :, 3:] == 1.0)


def test_model_outputs_are_bounded_and_differentiable():
    model = RGBRenderRecoveryNet(width=8, hidden_dim=32)
    rgb = torch.rand(3, 3, 64, 64)
    observed = torch.rand(3, 1, 64, 64)
    rendered = torch.rand(3, 5, 64, 64)
    raw = model(rgb, observed, rendered)
    decoded = decode_outputs(
        raw,
        max_center_px=56.0,
        max_log_depth=0.6,
        max_rotation_deg=70.0,
    )
    assert decoded["center_px"].shape == (3, 2)
    assert decoded["rotation_rad"].shape == (3, 3)
    assert torch.all(torch.linalg.vector_norm(decoded["rotation_rad"], dim=-1) <= np.deg2rad(70.0) + 1e-6)
    loss = sum(value.mean() for value in raw.values())
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_atan2_rotation_error_handles_pi():
    predicted = torch.tensor([[0.0, 0.0, np.pi]], dtype=torch.float64)
    target = torch.zeros_like(predicted)
    error = rotation_geodesic_atan2(predicted, target)
    torch.testing.assert_close(error, torch.tensor([np.pi], dtype=torch.float64), atol=1e-7, rtol=1e-7)


def test_anti_flip_quality_loss_enforces_explicit_margin():
    predicted = torch.tensor([[0.40, 0.50, 0.80]], requires_grad=True)
    target = torch.tensor([[0.0, 3.0, 1.0]])
    rotations = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, np.pi], [0.2, 0.0, 0.0]]]
    )
    result = anti_flip_quality_loss(
        predicted,
        target,
        rotations,
        margin=0.25,
        minimum_angle_deg=150.0,
    )
    torch.testing.assert_close(result["loss"], torch.tensor(0.15))
    torch.testing.assert_close(result["pair_accuracy"], torch.tensor(1.0))
    torch.testing.assert_close(result["margin_accuracy"], torch.tensor(0.0))
    result["loss"].backward()
    assert predicted.grad is not None
    assert predicted.grad[0, 0] > 0
    assert predicted.grad[0, 1] < 0


def recovery_result(pose: np.ndarray, error: float, source: str) -> RecoveryResult:
    return RecoveryResult(
        pose=pose,
        source=source,
        confidence=0.9,
        quality=error,
        silhouette_iou=0.9,
        total_error=error,
        delta_center_crop_px=np.zeros(2, dtype=np.float32),
        delta_log_depth=0.0,
        delta_rotation_rad=np.zeros(3, dtype=np.float32),
        rendered_mask_crop=np.zeros((8, 8), dtype=bool),
    )


def orientation_gate_args() -> SimpleNamespace:
    return SimpleNamespace(
        orientation_gates=True,
        normal_max_rotation_step_deg=30.0,
        uncertain_max_rotation_step_deg=60.0,
        max_rank0_rotation_disagreement_deg=90.0,
    )


def test_orientation_gate_rejects_lower_error_flip():
    anchor = np.eye(4)
    anchor[2, 3] = 4.0
    flipped = anchor.copy()
    flipped[:3, :3] = so3_exp(np.asarray([0.0, 0.0, np.pi]))
    translated = anchor.copy()
    translated[:3, 3] = [0.1, -0.05, 4.2]
    track = Track(
        track_id=0,
        obj_id=1,
        pose=anchor.copy(),
        previous_pose=None,
        bbox_xywh=np.asarray([10.0, 10.0, 20.0, 20.0]),
        confidence=0.9,
        mode=TrackMode.NORMAL,
        source="previous",
        scene_id=1,
        im_id=1,
    )
    selected, eligible, diagnostics = select_with_orientation_gates(
        [
            recovery_result(flipped, 0.1, "flip"),
            recovery_result(translated, 0.2, "correct"),
        ],
        track=track,
        rank0_pose=anchor,
        args=orientation_gate_args(),
    )
    assert selected.source == "correct"
    assert [item.source for item in eligible] == ["correct"]
    assert diagnostics["orientation_gate_triggered"] == 1
    assert diagnostics["orientation_gate_fallback"] == 0
    assert diagnostics["orientation_candidates_rejected"] == 1


def test_orientation_gate_fallback_keeps_translation_and_anchors_rotation():
    anchor = np.eye(4)
    anchor[2, 3] = 4.0
    flipped = anchor.copy()
    flipped[:3, :3] = so3_exp(np.asarray([0.0, 0.0, np.pi]))
    flipped[:3, 3] = [0.3, -0.2, 5.5]
    selected, eligible, diagnostics = select_with_orientation_gates(
        [recovery_result(flipped, 0.1, "flip")],
        track=None,
        rank0_pose=anchor,
        args=orientation_gate_args(),
    )
    np.testing.assert_allclose(selected.pose[:3, 3], flipped[:3, 3])
    np.testing.assert_allclose(selected.pose[:3, :3], anchor[:3, :3])
    assert len(eligible) == 1
    assert eligible[0] is selected
    assert diagnostics["orientation_gate_triggered"] == 1
    assert diagnostics["orientation_gate_fallback"] == 1
    assert diagnostics["orientation_gate_anchor"] == "gigapose_rank0"


def test_broad_recovery_flip_hypothesis_can_be_disabled():
    seed_pose = np.eye(4)
    seed_pose[2, 3] = 4.0
    seed = RecoveryProposal(seed_pose, "seed", 0.9, 0.0)
    detection = SimpleNamespace(center=np.asarray([32.0, 32.0]))
    K = np.asarray(
        [[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]]
    )
    args = SimpleNamespace(
        rotation_offsets_deg="",
        yaw_offsets_deg="",
        log_depth_offsets="",
        center_offsets_px="",
        broad_seeds=1,
        max_candidates=16,
        allow_flip_hypotheses=False,
    )

    without_flip = broad_recovery_proposals([seed], detection, K, args)
    assert len(without_flip) == 1

    args.allow_flip_hypotheses = True
    with_flip = broad_recovery_proposals([seed], detection, K, args)
    assert len(with_flip) == 2
    assert abs(rotation_error_deg(with_flip[0].pose, with_flip[1].pose) - 180.0) < 1e-4


def test_batched_inference_refines_without_observed_depth():
    K = np.asarray([[90.0, 0.0, 32.0], [0.0, 90.0, 32.0], [0.0, 0.0, 1.0]])
    pose = np.eye(4)
    pose[2, 3] = 4.0
    mask, _ = SquareRenderer().render(pose, K, (64, 64))
    detection = Detection(0, np.asarray([24.0, 24.0, 17.0, 17.0]), mask)
    frame = FrameData(
        scene_id=1,
        im_id=2,
        image=np.full((64, 64, 3), 127, dtype=np.uint8),
        K=K,
        detections=[detection],
        depth_m=None,
    )
    predictor = RGBSelfRecoveryPredictor(
        RGBRenderRecoveryNet(width=8, hidden_dim=32),
        {
            "crop_size": 64,
            "crop_scale": 2.5,
            "max_center_crop_px": 56.0,
            "max_log_depth": 0.6,
            "max_rotation_deg": 70.0,
            "uses_observed_depth": False,
        },
        "cpu",
    )
    results = predictor.refine_and_score(
        SquareRenderer(),
        frame,
        detection,
        [RecoveryProposal(pose, "test")],
        iterations=1,
    )
    assert len(results) == 1
    assert np.isfinite(results[0].pose).all()
    assert 0.0 <= results[0].confidence <= 1.0


def test_streaming_dataset_decodes_group(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": "rgb_render_self_recovery_v1",
                "uses_observed_depth": False,
                "groups": 1,
                "crop_size": 32,
                "crop_scale": 2.5,
            }
        )
    )
    rgb = np.full((32, 32, 3), 127, dtype=np.uint8)
    rgb_buffer = io.BytesIO()
    Image.fromarray(rgb).save(rgb_buffer, format="JPEG")
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    depth = mask.astype(np.float32) * 4.0
    K = np.asarray([[80.0, 0.0, 16.0], [0.0, 80.0, 16.0], [0.0, 0.0, 1.0]])
    rendered = np.stack(
        [encode_render_channels(mask, depth, K, 4.0) for _ in range(3)]
    )
    data_buffer = io.BytesIO()
    np.savez_compressed(
        data_buffer,
        observed_mask=mask.astype(np.uint8),
        rendered=rendered,
        center_targets=np.zeros((3, 2), dtype=np.float32),
        log_depth_targets=np.zeros(3, dtype=np.float32),
        rotation_targets=np.zeros((3, 3), dtype=np.float32),
        correctable_targets=np.ones(3, dtype=np.bool_),
        confidence_targets=np.ones(3, dtype=np.float32),
        quality_targets=np.zeros(3, dtype=np.float32),
    )
    with tarfile.open(root / "shard-000000.tar", "w") as tar:
        for name, content in (
            ("sample.rgb.jpg", rgb_buffer.getvalue()),
            ("sample.data.npz", data_buffer.getvalue()),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    sample = next(iter(RGBRenderRecoveryDataset(root, training=False)))
    assert sample["rgb"].shape == (3, 32, 32)
    assert sample["observed_mask"].shape == (1, 32, 32)
    assert sample["rendered"].shape == (3, 5, 32, 32)
