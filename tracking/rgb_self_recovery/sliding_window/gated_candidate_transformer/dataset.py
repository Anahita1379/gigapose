"""Boundary-safe windows over one or more existing candidate bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from tracking.rgb_self_recovery.sliding_window.gru_selector.dataset import (
    CandidateBundle,
)


SCHEMA_KEYS = ("candidate_feature_names", "frame_feature_names", "saved_candidates")


class BundleCollection:
    def __init__(self, paths: list[Path] | tuple[Path, ...]):
        if not paths:
            raise ValueError("At least one candidate bundle is required")
        self.bundles = [CandidateBundle(Path(path)) for path in paths]
        reference = self.bundles[0].manifest
        for bundle in self.bundles[1:]:
            for key in SCHEMA_KEYS:
                if bundle.manifest.get(key) != reference.get(key):
                    raise ValueError(f"Candidate bundle schema differs for {key}")

    @property
    def candidate_dim(self) -> int:
        return self.bundles[0].candidate_dim

    @property
    def frame_dim(self) -> int:
        return self.bundles[0].frame_dim

    @property
    def candidate_count(self) -> int:
        return self.bundles[0].candidate_count

    @property
    def candidate_feature_names(self) -> list[str]:
        return list(self.bundles[0].manifest["candidate_feature_names"])

    @property
    def frame_feature_names(self) -> list[str]:
        return list(self.bundles[0].manifest["frame_feature_names"])

    @property
    def runs(self) -> set[str]:
        return {
            str(value)
            for bundle in self.bundles
            for value in bundle.arrays["sequence_run"].tolist()
        }


def collection_statistics(collection: BundleCollection) -> dict[str, np.ndarray]:
    candidate = np.concatenate(
        [bundle.arrays["features"][bundle.arrays["valid"]] for bundle in collection.bundles],
        axis=0,
    )
    frame = np.concatenate(
        [bundle.arrays["frame_features"] for bundle in collection.bundles], axis=0
    )
    return {
        "candidate_mean": candidate.mean(axis=0).astype(np.float32),
        "candidate_std": np.maximum(candidate.std(axis=0), 1e-4).astype(np.float32),
        "frame_mean": frame.mean(axis=0).astype(np.float32),
        "frame_std": np.maximum(frame.std(axis=0), 1e-4).astype(np.float32),
    }


@dataclass(frozen=True)
class ClipRef:
    bundle_index: int
    indices: np.ndarray


class TrainingWindowDataset(Dataset):
    """Consecutive clips; no clip crosses an exported segment boundary."""

    def __init__(
        self,
        paths: list[Path] | tuple[Path, ...],
        *,
        window_length: int = 8,
        stride: int = 1,
    ):
        if window_length < 2 or stride < 1:
            raise ValueError("Window length must be >=2 and stride must be positive")
        self.collection = BundleCollection(paths)
        self.window_length = int(window_length)
        self.clips: list[ClipRef] = []
        for bundle_index, bundle in enumerate(self.collection.bundles):
            segments = bundle.arrays["segment_id"]
            for segment_id in np.unique(segments):
                indices = np.flatnonzero(segments == segment_id)
                if len(indices) <= self.window_length:
                    self.clips.append(ClipRef(bundle_index, indices))
                    continue
                starts = list(range(0, len(indices) - self.window_length + 1, stride))
                final = len(indices) - self.window_length
                if starts[-1] != final:
                    starts.append(final)
                self.clips.extend(
                    ClipRef(bundle_index, indices[start : start + self.window_length])
                    for start in starts
                )
        if not self.clips:
            raise ValueError("Candidate data produced no training windows")

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        clip = self.clips[item]
        return _window_tensors(
            self.collection.bundles[clip.bundle_index],
            clip.indices,
            self.window_length,
        )


@dataclass(frozen=True)
class FixedLagRef:
    bundle_index: int
    window_indices: np.ndarray
    window_valid: np.ndarray
    target_global_index: int


class FixedLagWindowDataset(Dataset):
    """One centered/padded context window for every frame in chronological order."""

    def __init__(
        self,
        path: Path,
        *,
        window_length: int = 8,
        fixed_lag: int = 4,
    ):
        if not 0 <= fixed_lag < window_length:
            raise ValueError("Fixed lag must be in [0, window_length)")
        self.collection = BundleCollection([Path(path)])
        self.bundle = self.collection.bundles[0]
        self.window_length = int(window_length)
        self.fixed_lag = int(fixed_lag)
        self.target_position = window_length - fixed_lag - 1
        self.samples: list[FixedLagRef] = []
        segments = self.bundle.arrays["segment_id"]
        for segment_id in np.unique(segments):
            segment = np.flatnonzero(segments == segment_id)
            for local_target, global_target in enumerate(segment):
                local_positions = (
                    local_target - self.target_position + np.arange(window_length)
                )
                valid = (local_positions >= 0) & (local_positions < len(segment))
                safe = np.clip(local_positions, 0, len(segment) - 1)
                self.samples.append(
                    FixedLagRef(0, segment[safe], valid, int(global_target))
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        sample = self.samples[item]
        result = _window_tensors(
            self.bundle,
            sample.window_indices,
            self.window_length,
            frame_valid_override=sample.window_valid,
        )
        result["target_global_index"] = torch.tensor(
            sample.target_global_index, dtype=torch.long
        )
        return result


def _window_tensors(
    bundle: CandidateBundle,
    indices: np.ndarray,
    window_length: int,
    *,
    frame_valid_override: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    arrays = bundle.arrays
    length = min(len(indices), window_length)
    candidates = bundle.candidate_count
    candidate_features = np.zeros(
        (window_length, candidates, bundle.candidate_dim), dtype=np.float32
    )
    frame_features = np.zeros((window_length, bundle.frame_dim), dtype=np.float32)
    candidate_valid = np.zeros((window_length, candidates), dtype=bool)
    frame_valid = np.zeros(window_length, dtype=bool)
    poses = np.repeat(
        np.eye(4, dtype=np.float32)[None, None], window_length * candidates, axis=0
    ).reshape(window_length, candidates, 4, 4)
    ground_truth = np.repeat(
        np.eye(4, dtype=np.float32)[None], window_length, axis=0
    )
    costs = np.zeros((window_length, candidates), dtype=np.float32)
    translation_error = np.zeros_like(costs)
    rotation_error = np.zeros_like(costs)
    labels = np.zeros(window_length, dtype=np.int64)
    time_s = np.zeros(window_length, dtype=np.float32)
    global_indices = np.full(window_length, -1, dtype=np.int64)

    selected = np.asarray(indices[:length], dtype=np.int64)
    candidate_features[:length] = arrays["features"][selected]
    frame_features[:length] = arrays["frame_features"][selected]
    candidate_valid[:length] = arrays["valid"][selected]
    frame_valid[:length] = True
    poses[:length] = arrays["poses"][selected]
    ground_truth[:length] = arrays["ground_truth_pose"][selected]
    costs[:length] = arrays["oracle_cost"][selected]
    translation_error[:length] = arrays["translation_error_m"][selected]
    rotation_error[:length] = arrays["rotation_error_deg"][selected]
    labels[:length] = arrays["oracle_index"][selected]
    time_values = np.asarray(arrays["time_s"][selected], dtype=np.float32)
    time_s[:length] = np.where(np.isfinite(time_values), time_values, 0.0)
    global_indices[:length] = selected

    if frame_valid_override is not None:
        override = np.asarray(frame_valid_override, dtype=bool)
        if override.shape != (window_length,):
            raise ValueError("Frame-valid override has the wrong shape")
        frame_valid &= override
        candidate_valid &= override[:, None]
    # MultiheadAttention cannot receive an all-masked key row. Candidate zero
    # is a numerical placeholder on padded frames; frame_valid excludes it.
    candidate_valid[~frame_valid, 0] = True

    return {
        "candidate_features": torch.from_numpy(candidate_features),
        "frame_features": torch.from_numpy(frame_features),
        "candidate_valid": torch.from_numpy(candidate_valid),
        "frame_valid": torch.from_numpy(frame_valid),
        "poses": torch.from_numpy(poses),
        "ground_truth_pose": torch.from_numpy(ground_truth),
        "oracle_costs": torch.from_numpy(costs),
        "translation_error_m": torch.from_numpy(translation_error),
        "rotation_error_deg": torch.from_numpy(rotation_error),
        "labels": torch.from_numpy(labels),
        "time_s": torch.from_numpy(time_s),
        "global_indices": torch.from_numpy(global_indices),
    }
