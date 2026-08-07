"""Boundary-safe clips of raw GigaPose measurements and pose targets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MeasurementBundle:
    def __init__(self, path: Path):
        root = Path(path)
        data_path = root / "measurements.npz" if root.is_dir() else root
        manifest_path = root / "manifest.json" if root.is_dir() else root.with_name("manifest.json")
        if not data_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Incomplete measurement bundle: {path}")
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("format") not in {
            "rgb_self_recovery_gru_kalman_measurements_v1",
            "rgb_self_recovery_gru_kalman_measurements_v2",
        }:
            raise ValueError(f"Unsupported measurement bundle: {path}")
        with np.load(data_path, allow_pickle=False) as payload:
            self.arrays = {name: np.asarray(payload[name]).copy() for name in payload.files}
        count = len(self.arrays["scene_id"])
        if any(len(value) != count for value in self.arrays.values()):
            raise ValueError("Measurement arrays have inconsistent lengths")
        if self.arrays["measurement_pose"].shape[1:] != (4, 4):
            raise ValueError("Measurement poses must be 4x4")
        # Version-one bundles predate learned component-wise GigaPose fallback.
        # Treating the measurement as its own baseline preserves their behavior.
        if "baseline_pose" not in self.arrays:
            self.arrays["baseline_pose"] = self.arrays["measurement_pose"].copy()
        if self.arrays["baseline_pose"].shape[1:] != (4, 4):
            raise ValueError("Fallback baseline poses must be 4x4")
        if "baseline_score" not in self.arrays:
            self.arrays["baseline_score"] = self.arrays["measurement_score"].copy()

    def __len__(self) -> int:
        return len(self.arrays["scene_id"])

    @property
    def context_dim(self) -> int:
        return int(self.arrays["context_features"].shape[-1])


def context_statistics(bundle: MeasurementBundle) -> dict[str, np.ndarray]:
    context = bundle.arrays["context_features"].astype(np.float32)
    return {
        "context_mean": context.mean(axis=0).astype(np.float32),
        "context_std": np.maximum(context.std(axis=0), 1e-4).astype(np.float32),
    }


class FilterClipDataset(Dataset):
    def __init__(self, path: Path, *, clip_length: int = 16, stride: int = 4):
        if clip_length < 2 or stride < 1:
            raise ValueError("clip_length must be >=2 and stride positive")
        self.bundle = MeasurementBundle(path)
        self.clip_length = int(clip_length)
        self.clips: list[np.ndarray] = []
        segments = self.bundle.arrays["segment_id"]
        for segment_id in np.unique(segments):
            indices = np.flatnonzero(segments == segment_id)
            if len(indices) < 2:
                continue
            if len(indices) <= clip_length:
                self.clips.append(indices)
                continue
            starts = list(range(0, len(indices) - clip_length + 1, stride))
            final = len(indices) - clip_length
            if starts[-1] != final:
                starts.append(final)
            self.clips.extend(indices[start : start + clip_length] for start in starts)
        if not self.clips:
            raise ValueError("No temporal segment contains at least two measurements")

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        indices = self.clips[item]
        arrays = self.bundle.arrays
        length = len(indices)
        measurement = np.repeat(np.eye(4, dtype=np.float32)[None], self.clip_length, axis=0)
        baseline = measurement.copy()
        target = measurement.copy()
        context = np.zeros((self.clip_length, self.bundle.context_dim), dtype=np.float32)
        delta_time = np.zeros(self.clip_length, dtype=np.float32)
        valid = np.zeros(self.clip_length, dtype=bool)
        measurement[:length] = arrays["measurement_pose"][indices]
        baseline[:length] = arrays["baseline_pose"][indices]
        target[:length] = arrays["ground_truth_pose"][indices]
        context[:length] = arrays["context_features"][indices]
        delta_time[:length] = arrays["delta_time_s"][indices]
        valid[:length] = True
        # Clip starts are filter resets even when sampled from a longer segment.
        delta_time[0] = 0.0
        return {
            "measurement_pose": torch.from_numpy(measurement),
            "baseline_pose": torch.from_numpy(baseline),
            "target_pose": torch.from_numpy(target),
            "context_features": torch.from_numpy(context),
            "delta_time_s": torch.from_numpy(delta_time),
            "frame_valid": torch.from_numpy(valid),
            "indices": torch.from_numpy(np.pad(indices, (0, self.clip_length - length), constant_values=-1)),
        }
