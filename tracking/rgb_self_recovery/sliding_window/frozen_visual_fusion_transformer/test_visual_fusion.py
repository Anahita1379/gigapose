"""Focused tests for frozen visual fusion and lag masking."""

import tempfile
from pathlib import Path
import unittest

import torch

from .model import FORMAT, FrozenVisualFusionTransformer, load_transformer
from .evaluate import _trust


class VisualFusionModelTest(unittest.TestCase):
    def _model(self):
        return FrozenVisualFusionTransformer(
            23, 8, 16,
            candidate_visual_dim=32,
            frame_visual_dim=24,
            window_length=8,
            model_dim=32,
            heads=4,
            feedforward_dim=64,
            cross_layers=2,
            dropout=0.0,
            attention_future_lag=4,
        )

    def test_visual_shapes_and_fixed_lag(self):
        model = self._model().eval()
        poses = torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(2, 8, 16, 1, 1)
        output = model(
            torch.randn(2, 8, 16, 23),
            torch.randn(2, 8, 8),
            torch.ones(2, 8, 16, dtype=torch.bool),
            torch.ones(2, 8, dtype=torch.bool),
            candidate_visual_features=torch.randn(2, 8, 16, 32),
            frame_visual_features=torch.randn(2, 8, 24),
            candidate_poses=poses,
            time_s=torch.arange(8, dtype=torch.float32)[None].repeat(2, 1) * 0.1,
        )
        self.assertEqual(output["candidate_logits"].shape, (2, 8, 16))
        self.assertEqual(output["residual"].shape, (2, 8, 16, 6))
        self.assertEqual(output["translation_fixed_lag_delta"].shape, (2, 8, 3))
        self.assertFalse(bool(model.temporal_attention_mask(8, 4, "cpu")[3].any()))

    def test_visual_evidence_changes_predictions(self):
        torch.manual_seed(9)
        model = self._model().eval()
        numeric_candidate = torch.randn(1, 8, 16, 23)
        numeric_frame = torch.randn(1, 8, 8)
        valid_candidate = torch.ones(1, 8, 16, dtype=torch.bool)
        valid_frame = torch.ones(1, 8, dtype=torch.bool)
        frame_visual = torch.randn(1, 8, 24)
        poses = torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(1, 8, 16, 1, 1)
        times = torch.arange(8, dtype=torch.float32)[None] * 0.1
        first = model(
            numeric_candidate, numeric_frame, valid_candidate, valid_frame,
            candidate_visual_features=torch.zeros(1, 8, 16, 32),
            frame_visual_features=frame_visual,
            candidate_poses=poses, time_s=times,
        )
        second = model(
            numeric_candidate, numeric_frame, valid_candidate, valid_frame,
            candidate_visual_features=torch.ones(1, 8, 16, 32) * 3,
            frame_visual_features=frame_visual,
            candidate_poses=poses, time_s=times,
        )
        self.assertFalse(torch.allclose(
            first["candidate_logits"], second["candidate_logits"]
        ))

    def test_checkpoint_round_trip(self):
        model = self._model()
        payload = {
            "format": FORMAT,
            "model_config": model.config(),
            "model_state": model.state_dict(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.ckpt"
            torch.save(payload, path)
            loaded, loaded_payload = load_transformer(path)
        self.assertEqual(loaded_payload["format"], FORMAT)
        self.assertEqual(loaded.candidate_visual_dim, 32)
        self.assertEqual(loaded.frame_visual_dim, 24)

    def test_visual_features_are_required(self):
        model = self._model()
        with self.assertRaisesRegex(ValueError, "requires visual feature"):
            model(
                torch.randn(1, 8, 16, 23),
                torch.randn(1, 8, 8),
                torch.ones(1, 8, 16, dtype=torch.bool),
                torch.ones(1, 8, dtype=torch.bool),
            )

    def test_trust_metrics_measure_fallback_against_hidden_proposal(self):
        rows = [
            {
                "baseline_translation_error_m": 2.0,
                "proposed_translation_error_m": 1.0,
                "translation_trust_probability": 0.9,
                "translation_fallback": False,
            },
            {
                "baseline_translation_error_m": 1.0,
                "proposed_translation_error_m": 3.0,
                "translation_trust_probability": 0.1,
                "translation_fallback": True,
            },
        ]
        metrics = _trust(rows, "translation")
        self.assertEqual(metrics["fallback_precision"], 1.0)
        self.assertEqual(metrics["fallback_recall"], 1.0)
        self.assertAlmostEqual(metrics["brier_score"], 0.01)


if __name__ == "__main__":
    unittest.main()
