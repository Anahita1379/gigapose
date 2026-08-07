"""Streaming recovery shards augmented with training-only metric depth."""

from __future__ import annotations

import io
import json
from pathlib import Path
import random
import tarfile

import numpy as np
from PIL import Image
import torch
from torch.utils.data import IterableDataset, get_worker_info

from tracking.rgb_self_recovery.dataset import _augment_mask, _augment_rgb
from tracking.rgb_self_recovery.render_inputs import decode_render_channels


class PrivilegedDepthRecoveryDataset(IterableDataset):
    def __init__(self, root: Path, *, training: bool, seed: int = 20260720,
                 shuffle_buffer: int = 128):
        super().__init__()
        self.root = Path(root)
        self.training = bool(training)
        self.seed = int(seed)
        self.shuffle_buffer = int(shuffle_buffer)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("format") != "rgb_render_self_recovery_privileged_depth_v1":
            raise ValueError(f"Unsupported privileged-depth data: {self.root}")
        self.shards = sorted(self.root.glob("shard-*.tar"))
        if not self.shards:
            raise FileNotFoundError(f"No privileged-depth shards in {self.root}")

    def __len__(self):
        return int(self.manifest["groups"])

    @staticmethod
    def _decode(rgb_bytes: bytes, data_bytes: bytes):
        rgb = np.asarray(Image.open(io.BytesIO(rgb_bytes)).convert("RGB"))
        with np.load(io.BytesIO(data_bytes)) as payload:
            values = {name: np.asarray(payload[name]).copy() for name in payload.files}
        required = {"observed_depth_m", "observed_depth_valid"}
        if not required <= values.keys():
            raise ValueError(f"Privileged shard lacks {sorted(required - values.keys())}")
        return {
            "rgb_uint8": torch.from_numpy(rgb.copy()),
            "observed_mask_uint8": torch.from_numpy(values["observed_mask"].astype(np.uint8)),
            "rendered": torch.from_numpy(decode_render_channels(values["rendered"])),
            "center_targets": torch.from_numpy(values["center_targets"].astype(np.float32)),
            "log_depth_targets": torch.from_numpy(values["log_depth_targets"].astype(np.float32)),
            "rotation_targets": torch.from_numpy(values["rotation_targets"].astype(np.float32)),
            "correctable_targets": torch.from_numpy(values["correctable_targets"].astype(bool)),
            "confidence_targets": torch.from_numpy(values["confidence_targets"].astype(np.float32)),
            "quality_targets": torch.from_numpy(values["quality_targets"].astype(np.float32)),
            "observed_depth_m": torch.from_numpy(values["observed_depth_m"].astype(np.float32)),
            "observed_depth_valid": torch.from_numpy(values["observed_depth_valid"].astype(bool)),
        }

    def _iter_shard(self, path: Path):
        pending = {}
        with tarfile.open(path) as archive:
            for member in archive:
                suffix = None
                if member.isfile() and member.name.endswith(".rgb.jpg"):
                    key, suffix = member.name.removesuffix(".rgb.jpg"), "rgb"
                elif member.isfile() and member.name.endswith(".data.npz"):
                    key, suffix = member.name.removesuffix(".data.npz"), "data"
                if suffix is None:
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                pending.setdefault(key, {})[suffix] = handle.read()
                if {"rgb", "data"} <= pending[key].keys():
                    value = pending.pop(key)
                    yield self._decode(value["rgb"], value["data"])

    def _prepare(self, sample, rng):
        rgb = sample.pop("rgb_uint8").numpy()
        mask = sample.pop("observed_mask_uint8").numpy()
        if self.training:
            rgb = _augment_rgb(rgb, rng)
            mask = _augment_mask(mask, rng)
        else:
            rgb = rgb.astype(np.float32) / 255.0
        sample["rgb"] = torch.from_numpy(np.moveaxis(rgb.astype(np.float32), -1, 0))
        sample["observed_mask"] = torch.from_numpy(mask.astype(np.float32)[None])
        sample["observed_depth_m"] = sample["observed_depth_m"][None]
        sample["observed_depth_valid"] = sample["observed_depth_valid"][None]
        return sample

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        shards = self.shards[worker_id::worker_count]
        rng = np.random.default_rng(self.seed + 1009 * worker_id)
        if self.training:
            shards = list(shards)
            random.Random(self.seed + worker_id).shuffle(shards)
        buffer = []
        for shard in shards:
            for sample in self._iter_shard(shard):
                value = self._prepare(sample, rng)
                if not self.training or self.shuffle_buffer <= 1:
                    yield value
                else:
                    buffer.append(value)
                    if len(buffer) >= self.shuffle_buffer:
                        yield buffer.pop(int(rng.integers(len(buffer))))
        while buffer:
            yield buffer.pop(int(rng.integers(len(buffer))))

