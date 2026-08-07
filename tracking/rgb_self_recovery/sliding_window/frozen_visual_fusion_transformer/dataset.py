"""Candidate windows augmented by precomputed frozen visual embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tracking.rgb_self_recovery.sliding_window.gated_candidate_transformer.dataset import (
    BundleCollection,
    FixedLagWindowDataset,
    TrainingWindowDataset,
)
from tracking.rgb_self_recovery.sliding_window.gru_selector.dataset import resolve_bundle


def _sidecar(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    data_path, _ = resolve_bundle(path)
    visual_path = data_path.with_name("visual_features.npz")
    manifest_path = data_path.with_name("visual_manifest.json")
    if not visual_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing visual sidecar beside {data_path}; run prepare_visual_features"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "rgb_self_recovery_frozen_visual_features_v1":
        raise ValueError(f"Unsupported visual feature sidecar: {manifest_path}")
    with np.load(visual_path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]).copy() for name in payload.files}
    return arrays, manifest


class VisualBundleCollection(BundleCollection):
    def __init__(self, paths):
        super().__init__(paths)
        self.visual = []
        self.visual_manifests = []
        for path, bundle in zip(paths, self.bundles):
            arrays, manifest = _sidecar(Path(path))
            if not np.array_equal(arrays["scene_id"], bundle.arrays["scene_id"]):
                raise ValueError("Visual sidecar scene IDs do not align with candidates")
            if not np.array_equal(arrays["im_id"], bundle.arrays["im_id"]):
                raise ValueError("Visual sidecar image IDs do not align with candidates")
            if arrays["candidate_visual_features"].shape[:2] != (
                len(bundle), bundle.candidate_count
            ):
                raise ValueError("Visual candidate dimensions do not align")
            self.visual.append(arrays)
            self.visual_manifests.append(manifest)
        dimensions = {
            (self.frame_visual_dim_at(i), self.candidate_visual_dim_at(i))
            for i in range(len(self.visual))
        }
        if len(dimensions) != 1:
            raise ValueError(f"Visual feature dimensions differ: {dimensions}")
        signatures = {
            (
                manifest.get("checkpoint"),
                manifest.get("mesh_scale"),
                manifest.get("center_mesh"),
            )
            for manifest in self.visual_manifests
        }
        if len(signatures) != 1:
            raise ValueError("Visual sidecars were not extracted with one frozen encoder/CAD setup")

    def frame_visual_dim_at(self, index: int) -> int:
        return int(self.visual[index]["frame_visual_features"].shape[-1])

    def candidate_visual_dim_at(self, index: int) -> int:
        return int(self.visual[index]["candidate_visual_features"].shape[-1])

    @property
    def frame_visual_dim(self) -> int:
        return self.frame_visual_dim_at(0)

    @property
    def candidate_visual_dim(self) -> int:
        return self.candidate_visual_dim_at(0)


def visual_statistics(collection: VisualBundleCollection) -> dict[str, np.ndarray]:
    candidate = np.concatenate([
        values["candidate_visual_features"][bundle.arrays["valid"]]
        for bundle, values in zip(collection.bundles, collection.visual)
    ], axis=0).astype(np.float32)
    frame = np.concatenate([
        values["frame_visual_features"] for values in collection.visual
    ], axis=0).astype(np.float32)
    return {
        "candidate_visual_mean": candidate.mean(axis=0).astype(np.float32),
        "candidate_visual_std": np.maximum(candidate.std(axis=0), 1e-4).astype(np.float32),
        "frame_visual_mean": frame.mean(axis=0).astype(np.float32),
        "frame_visual_std": np.maximum(frame.std(axis=0), 1e-4).astype(np.float32),
    }


class VisualTrainingWindowDataset(TrainingWindowDataset):
    def __init__(self, paths, *, window_length=8, stride=1):
        super().__init__(paths, window_length=window_length, stride=stride)
        self.collection = VisualBundleCollection(paths)

    def __getitem__(self, item):
        result = super().__getitem__(item)
        clip = self.clips[item]
        indices = clip.indices[: self.window_length]
        values = self.collection.visual[clip.bundle_index]
        frame = np.zeros((self.window_length, self.collection.frame_visual_dim), np.float32)
        candidate = np.zeros((
            self.window_length,
            self.collection.candidate_count,
            self.collection.candidate_visual_dim,
        ), np.float32)
        frame[:len(indices)] = values["frame_visual_features"][indices]
        candidate[:len(indices)] = values["candidate_visual_features"][indices]
        result["frame_visual_features"] = torch.from_numpy(frame)
        result["candidate_visual_features"] = torch.from_numpy(candidate)
        return result


class VisualFixedLagWindowDataset(FixedLagWindowDataset):
    def __init__(self, path, *, window_length=8, fixed_lag=4):
        super().__init__(path, window_length=window_length, fixed_lag=fixed_lag)
        self.visual_collection = VisualBundleCollection([path])

    def __getitem__(self, item):
        result = super().__getitem__(item)
        sample = self.samples[item]
        values = self.visual_collection.visual[0]
        frame = values["frame_visual_features"][sample.window_indices].astype(np.float32)
        candidate = values["candidate_visual_features"][sample.window_indices].astype(np.float32)
        frame[~sample.window_valid] = 0
        candidate[~sample.window_valid] = 0
        result["frame_visual_features"] = torch.from_numpy(frame)
        result["candidate_visual_features"] = torch.from_numpy(candidate)
        return result
