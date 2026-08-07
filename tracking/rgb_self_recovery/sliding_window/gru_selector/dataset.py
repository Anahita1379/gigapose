"""Contiguous, boundary-safe sequence clips for GRU candidate training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def resolve_bundle(path: Path) -> tuple[Path, Path]:
    path = Path(path)
    if path.is_dir():
        return path / "candidates.npz", path / "manifest.json"
    return path, path.with_name("manifest.json")


class CandidateBundle:
    def __init__(self, path: Path):
        data_path, manifest_path = resolve_bundle(path)
        if not data_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Candidate bundle is incomplete: {path}")
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("format") != "rgb_self_recovery_gru_candidates_v1":
            raise ValueError(f"Unsupported candidate bundle: {path}")
        with np.load(data_path, allow_pickle=False) as payload:
            self.arrays = {name: np.asarray(payload[name]).copy() for name in payload.files}
        count = len(self.arrays["scene_id"])
        if any(len(value) != count for value in self.arrays.values()):
            raise ValueError("Candidate bundle arrays have inconsistent frame counts")
        if not np.all(self.arrays["valid"][:, 0]):
            raise ValueError("Candidate index 0 must always be valid GigaPose")

    def __len__(self) -> int:
        return len(self.arrays["scene_id"])

    @property
    def candidate_dim(self) -> int:
        return int(self.arrays["features"].shape[-1])

    @property
    def frame_dim(self) -> int:
        return int(self.arrays["frame_features"].shape[-1])

    @property
    def candidate_count(self) -> int:
        return int(self.arrays["features"].shape[1])


def feature_statistics(bundle: CandidateBundle) -> dict[str, np.ndarray]:
    candidate = bundle.arrays["features"][bundle.arrays["valid"]]
    frame = bundle.arrays["frame_features"]
    return {
        "candidate_mean": candidate.mean(axis=0).astype(np.float32),
        "candidate_std": np.maximum(candidate.std(axis=0), 1e-4).astype(np.float32),
        "frame_mean": frame.mean(axis=0).astype(np.float32),
        "frame_std": np.maximum(frame.std(axis=0), 1e-4).astype(np.float32),
    }


class SequenceClipDataset(Dataset):
    """Fixed-length clips that never cross an exported contiguous segment."""

    def __init__(self, path: Path, *, clip_length: int = 8, stride: int = 1):
        if clip_length < 1 or stride < 1:
            raise ValueError("clip_length and stride must be positive")
        self.bundle = CandidateBundle(path)
        self.clip_length = int(clip_length)
        self.stride = int(stride)
        segments = self.bundle.arrays["segment_id"]
        self.clips: list[np.ndarray] = []
        for segment_id in np.unique(segments):
            indices = np.flatnonzero(segments == segment_id)
            if not len(indices):
                continue
            if len(indices) <= self.clip_length:
                self.clips.append(indices)
                continue
            starts = list(range(0, len(indices) - self.clip_length + 1, self.stride))
            last = len(indices) - self.clip_length
            if starts[-1] != last:
                starts.append(last)
            self.clips.extend(indices[start : start + self.clip_length] for start in starts)
        if not self.clips:
            raise ValueError("Candidate bundle produced no sequence clips")

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        indices = self.clips[item]
        arrays = self.bundle.arrays
        length = len(indices)
        candidates = self.bundle.candidate_count
        candidate_dim = self.bundle.candidate_dim
        frame_dim = self.bundle.frame_dim
        candidate_features = np.zeros(
            (self.clip_length, candidates, candidate_dim), dtype=np.float32
        )
        frame_features = np.zeros((self.clip_length, frame_dim), dtype=np.float32)
        candidate_valid = np.zeros((self.clip_length, candidates), dtype=bool)
        frame_valid = np.zeros(self.clip_length, dtype=bool)
        labels = np.zeros(self.clip_length, dtype=np.int64)
        costs = np.zeros((self.clip_length, candidates), dtype=np.float32)
        candidate_features[:length] = arrays["features"][indices]
        frame_features[:length] = arrays["frame_features"][indices]
        candidate_valid[:length] = arrays["valid"][indices]
        frame_valid[:length] = True
        labels[:length] = arrays["oracle_index"][indices]
        costs[:length] = arrays["oracle_cost"][indices]
        # Keep padded GRU steps numerically well-defined; frame_valid excludes them.
        candidate_valid[length:, 0] = True
        return {
            "candidate_features": torch.from_numpy(candidate_features),
            "frame_features": torch.from_numpy(frame_features),
            "candidate_valid": torch.from_numpy(candidate_valid),
            "frame_valid": torch.from_numpy(frame_valid),
            "labels": torch.from_numpy(labels),
            "oracle_costs": torch.from_numpy(costs),
            "indices": torch.from_numpy(
                np.pad(indices, (0, self.clip_length - length), constant_values=-1)
            ),
        }

