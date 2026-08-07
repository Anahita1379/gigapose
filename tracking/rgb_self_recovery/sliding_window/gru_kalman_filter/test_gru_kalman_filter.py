"""Tests for the learned SE(3) GRU filter and boundary-safe clips."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from .calibrate_fallback import calibrate_component
from .dataset import FilterClipDataset, context_statistics
from .features import CONTEXT_FEATURE_NAMES
from .geometry import rotation_angle, so3_exp, so3_log
from .model import GRUKalmanFilter


def write_bundle(root: Path, run: str = "train_run") -> None:
    root.mkdir(parents=True)
    count = 10
    gt = np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0)
    gt[:, 0, 3] = np.arange(count, dtype=np.float32) * 0.2
    measurement = gt.copy()
    measurement[:, 0, 3] += np.linspace(0.2, -0.2, count, dtype=np.float32)
    baseline = gt.copy()
    baseline[:, 1, 3] += 0.1
    arrays = {
        "scene_id": np.ones(count, dtype=np.int64),
        "im_id": np.arange(count),
        "source_frame": np.arange(count),
        "time_s": np.arange(count) * 0.1,
        "delta_time_s": np.asarray([0.0] + [0.1] * (count - 1), dtype=np.float32),
        "segment_id": np.asarray([0] * 6 + [1] * 4),
        "measurement_score": np.ones(count, dtype=np.float32),
        "raw_translation_error_m": np.abs(measurement[:, 0, 3] - gt[:, 0, 3]),
        "raw_rotation_error_deg": np.zeros(count, dtype=np.float32),
        "measurement_pose": measurement,
        "baseline_pose": baseline,
        "baseline_score": np.ones(count, dtype=np.float32),
        "ground_truth_pose": gt,
        "context_features": np.ones((count, len(CONTEXT_FEATURE_NAMES)), dtype=np.float32),
        "source_run": np.asarray([run] * count, dtype="<U32"),
        "camera_id": np.asarray(["front"] * count, dtype="<U16"),
    }
    np.savez_compressed(root / "measurements.npz", **arrays)
    (root / "manifest.json").write_text(json.dumps({
        "format": "rgb_self_recovery_gru_kalman_measurements_v2",
        "context_feature_names": list(CONTEXT_FEATURE_NAMES),
    }))


class GRUKalmanFilterTests(unittest.TestCase):
    def test_near_pi_rotation_has_finite_backward_gradient(self):
        rotvec = torch.tensor(
            [[math.pi - 1e-6, 0.0, 0.0]], dtype=torch.float64,
            requires_grad=True,
        )
        rotation = so3_exp(rotvec)
        recovered = so3_log(rotation)
        angle = rotation_angle(
            rotation, torch.eye(3, dtype=torch.float64)[None]
        )
        loss = recovered.square().sum() + angle.sum()
        loss.backward()
        self.assertTrue(torch.isfinite(recovered).all())
        self.assertTrue(torch.isfinite(rotvec.grad).all())

    def test_so3_round_trip(self):
        vector = torch.tensor([[0.1, -0.2, 0.05]], dtype=torch.float32)
        recovered = so3_log(so3_exp(vector))
        self.assertTrue(torch.allclose(vector, recovered, atol=1e-5))

    def test_clips_do_not_cross_segments_and_model_backpropagates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            write_bundle(root)
            dataset = FilterClipDataset(root, clip_length=5, stride=2)
            segments = dataset.bundle.arrays["segment_id"]
            for item in range(len(dataset)):
                indices = dataset[item]["indices"].numpy()
                indices = indices[indices >= 0]
                self.assertEqual(len(set(segments[indices].tolist())), 1)
            statistics = {
                key: torch.from_numpy(value)
                for key, value in context_statistics(dataset.bundle).items()
            }
            model = GRUKalmanFilter(
                dataset.bundle.context_dim, hidden_dim=24, context_hidden=8,
                statistics=statistics,
            )
            sample = dataset[0]
            filtered, gains = model(
                sample["measurement_pose"][None],
                sample["context_features"][None],
                sample["delta_time_s"][None],
                sample["frame_valid"][None],
            )
            self.assertEqual(tuple(filtered.shape), (1, 5, 4, 4))
            self.assertEqual(tuple(gains.shape), (1, 5, 12))
            loss = filtered[..., :3, 3].square().mean() + rotation_angle(
                filtered[..., :3, :3], sample["target_pose"][None, ..., :3, :3]
            ).mean()
            loss.backward()
            self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_translation_only_filter_preserves_measurement_rotation(self):
        measurement = torch.eye(4, dtype=torch.float32).repeat(1, 5, 1, 1)
        vectors = torch.tensor(
            [[0.0, 0.0, 0.0], [0.1, -0.2, 0.3], [-0.4, 0.1, 0.2],
             [0.2, 0.3, -0.1], [1.0, -0.2, 0.4]],
            dtype=torch.float32,
        )
        measurement[0, :, :3, :3] = so3_exp(vectors)
        measurement[0, :, :3, 3] = torch.arange(5)[:, None] * torch.tensor(
            [0.2, 0.0, 0.1]
        )
        model = GRUKalmanFilter(
            len(CONTEXT_FEATURE_NAMES), hidden_dim=24, context_hidden=8,
            preserve_measurement_rotation=True,
        )
        filtered, _gains = model(
            measurement,
            torch.ones(1, 5, len(CONTEXT_FEATURE_NAMES)),
            torch.tensor([[0.0, 0.1, 0.1, 0.1, 0.1]]),
            torch.ones(1, 5, dtype=torch.bool),
        )
        self.assertTrue(torch.equal(
            filtered[0, :, :3, :3], measurement[0, :, :3, :3]
        ))
        loss = filtered[..., :3, 3].square().mean()
        loss.backward()
        self.assertTrue(any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))

    def test_component_fallback_heads_are_independent_and_backpropagate(self):
        measurement = torch.eye(4, dtype=torch.float32).repeat(2, 5, 1, 1)
        baseline = measurement.clone()
        baseline[:, :, 0, 3] = 0.5
        baseline[:, :, :3, :3] = so3_exp(
            torch.tensor([[0.0, 0.0, 0.2]], dtype=torch.float32)
        )[:, None].expand(2, 5, 3, 3)
        model = GRUKalmanFilter(
            len(CONTEXT_FEATURE_NAMES),
            hidden_dim=24,
            context_hidden=8,
            fallback_hidden=12,
            use_fallback_heads=True,
        )
        filtered, gains, trust_logits = model(
            measurement,
            torch.ones(2, 5, len(CONTEXT_FEATURE_NAMES)),
            torch.tensor([[0.0, 0.1, 0.1, 0.1, 0.1]]).expand(2, 5),
            torch.ones(2, 5, dtype=torch.bool),
            baseline_pose=baseline,
        )
        self.assertEqual(tuple(filtered.shape), (2, 5, 4, 4))
        self.assertEqual(tuple(gains.shape), (2, 5, 12))
        self.assertEqual(tuple(trust_logits.shape), (2, 5, 2))
        trust_logits.sum().backward()
        self.assertTrue(any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for name, parameter in model.named_parameters()
            if "translation_fallback_head" in name
        ))
        self.assertTrue(any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for name, parameter in model.named_parameters()
            if "rotation_fallback_head" in name
        ))

    def test_component_fallback_calibration_is_independent(self):
        rows = []
        for probability, candidate, baseline in [
            (0.1, 5.0, 1.0), (0.2, 4.0, 2.0), (0.9, 1.0, 3.0),
        ]:
            rows.append({
                "translation_trust_probability": str(probability),
                "candidate_translation_error_m": str(candidate),
                "baseline_translation_error_m": str(baseline),
                "raw_translation_error_m": str(candidate),
                "correction_translation_m": "0",
            })
        result = calibrate_component(
            rows, "translation", "m", 100.0, 1.0, 0.5, 0.0,
        )
        self.assertGreater(result["threshold"], 0.2)
        self.assertLessEqual(result["threshold"], 0.9)
        self.assertEqual(result["learned_fallback_count"], 2)
        self.assertAlmostEqual(result["mean_error"], 4 / 3)


if __name__ == "__main__":
    unittest.main()
