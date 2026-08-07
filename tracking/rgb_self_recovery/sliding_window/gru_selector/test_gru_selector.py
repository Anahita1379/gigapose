"""Tests for boundary-safe GRU candidate selection."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from .dataset import SequenceClipDataset, feature_statistics
from .features import CANDIDATE_FEATURE_NAMES, FRAME_FEATURE_NAMES
from .model import GRUCandidateSelector


def _bundle(root: Path):
    root.mkdir()
    frames, candidates = 7, 3
    valid = np.ones((frames, candidates), dtype=bool)
    arrays = {
        "scene_id": np.ones(frames, dtype=np.int64),
        "im_id": np.arange(frames),
        "source_frame": np.asarray([0, 1, 2, 10, 11, 20, 21]),
        "time_s": np.arange(frames, dtype=np.float64),
        "segment_id": np.asarray([0, 0, 0, 1, 1, 2, 2]),
        "poses": np.repeat(np.eye(4)[None, None], frames * candidates, axis=0).reshape(frames, candidates, 4, 4),
        "features": np.random.default_rng(3).normal(size=(frames, candidates, len(CANDIDATE_FEATURE_NAMES))).astype(np.float32),
        "frame_features": np.random.default_rng(4).normal(size=(frames, len(FRAME_FEATURE_NAMES))).astype(np.float32),
        "valid": valid,
        "sources": np.full((frames, candidates), "candidate", dtype="<U32"),
        "ground_truth_pose": np.repeat(np.eye(4)[None], frames, axis=0),
        "oracle_cost": np.ones((frames, candidates), dtype=np.float32),
        "translation_error_m": np.ones((frames, candidates), dtype=np.float32),
        "rotation_error_deg": np.ones((frames, candidates), dtype=np.float32),
        "oracle_index": np.asarray([0, 1, 0, 2, 0, 1, 0]),
        "sequence_run": np.asarray(["train_run"] * frames, dtype="<U32"),
        "sequence_camera": np.asarray(["rear"] * frames, dtype="<U16"),
    }
    np.savez_compressed(root / "candidates.npz", **arrays)
    (root / "manifest.json").write_text(json.dumps({
        "format": "rgb_self_recovery_gru_candidates_v1",
        "candidate_feature_names": list(CANDIDATE_FEATURE_NAMES),
        "frame_feature_names": list(FRAME_FEATURE_NAMES),
        "saved_candidates": candidates,
    }))


class GRUSelectorTests(unittest.TestCase):
    def test_clips_never_cross_segments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            _bundle(root)
            dataset = SequenceClipDataset(root, clip_length=4, stride=1)
            segment = dataset.bundle.arrays["segment_id"]
            for item in range(len(dataset)):
                indices = dataset[item]["indices"].numpy()
                indices = indices[indices >= 0]
                self.assertEqual(len(set(segment[indices].tolist())), 1)

    def test_model_masks_candidates_and_is_causal_shape_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bundle"
            _bundle(root)
            dataset = SequenceClipDataset(root, clip_length=4, stride=1)
            stats = {key: torch.from_numpy(value) for key, value in feature_statistics(dataset.bundle).items()}
            model = GRUCandidateSelector(
                dataset.bundle.candidate_dim,
                dataset.bundle.frame_dim,
                candidate_hidden=16,
                gru_hidden=24,
                statistics=stats,
            )
            sample = dataset[0]
            valid = sample["candidate_valid"].clone()
            valid[0, 2] = False
            logits, hidden = model(
                sample["candidate_features"][None],
                sample["frame_features"][None],
                valid[None],
            )
            self.assertEqual(tuple(logits.shape), (1, 4, 3))
            self.assertEqual(tuple(hidden.shape), (1, 1, 24))
            self.assertLess(float(logits[0, 0, 2].detach()), -1e8)


if __name__ == "__main__":
    unittest.main()
