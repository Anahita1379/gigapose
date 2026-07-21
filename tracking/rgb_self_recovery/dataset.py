"""Streaming grouped training data for RGB/render self recovery."""

from __future__ import annotations

import io
import json
from pathlib import Path
import random
import tarfile
from typing import Iterator

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import IterableDataset, get_worker_info

from tracking.rgb_self_recovery.render_inputs import decode_render_channels


def _augment_rgb(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    output = np.asarray(image, dtype=np.float32) / 255.0
    output = (output - 0.5) * float(rng.uniform(0.70, 1.35)) + 0.5
    output *= float(rng.uniform(0.65, 1.35))
    gamma = float(rng.uniform(0.75, 1.35))
    output = np.clip(output, 0.0, 1.0) ** gamma
    if rng.random() < 0.65:
        hsv = cv2.cvtColor(
            np.round(output * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV
        ).astype(np.float32)
        hsv[..., 0] = np.mod(hsv[..., 0] + rng.uniform(-8.0, 8.0), 180.0)
        hsv[..., 1] *= float(rng.uniform(0.60, 1.45))
        output = cv2.cvtColor(
            np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB
        ).astype(np.float32) / 255.0
    if rng.random() < 0.30:
        sigma = float(rng.uniform(0.25, 1.20))
        output = cv2.GaussianBlur(output, (0, 0), sigma)
    if rng.random() < 0.60:
        output += rng.normal(0.0, rng.uniform(0.0, 0.035), output.shape)
    return np.clip(output, 0.0, 1.0)


def _augment_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    output = np.asarray(mask, dtype=np.uint8)
    if rng.random() < 0.35:
        kernel_size = int(rng.choice([3, 5]))
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        operation = cv2.erode if rng.random() < 0.5 else cv2.dilate
        output = operation(output, kernel, iterations=1)
    if rng.random() < 0.25 and output.any():
        ys, xs = np.where(output > 0)
        if xs.size:
            width = max(1, int(rng.uniform(0.05, 0.25) * (np.ptp(xs) + 1)))
            height = max(1, int(rng.uniform(0.05, 0.25) * (np.ptp(ys) + 1)))
            x0 = int(rng.integers(xs.min(), max(xs.max() - width + 2, xs.min() + 1)))
            y0 = int(rng.integers(ys.min(), max(ys.max() - height + 2, ys.min() + 1)))
            output[y0 : y0 + height, x0 : x0 + width] = 0
    return output


class RGBRenderRecoveryDataset(IterableDataset):
    def __init__(
        self,
        root: Path,
        *,
        training: bool,
        seed: int = 20260720,
        shuffle_buffer: int = 128,
    ):
        super().__init__()
        self.root = Path(root)
        self.training = bool(training)
        self.seed = int(seed)
        self.shuffle_buffer = int(shuffle_buffer)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("format") != "rgb_render_self_recovery_v1":
            raise ValueError(f"Unsupported recovery format in {manifest_path}")
        if self.manifest.get("uses_observed_depth") is not False:
            raise ValueError("RGB self-recovery data must not use observed depth.")
        self.shards = sorted(self.root.glob("shard-*.tar"))
        if not self.shards:
            raise FileNotFoundError(f"No recovery shards in {self.root}")

    def __len__(self) -> int:
        return int(self.manifest["groups"])

    @staticmethod
    def _decode(rgb_bytes: bytes, data_bytes: bytes) -> dict[str, torch.Tensor]:
        rgb = np.asarray(Image.open(io.BytesIO(rgb_bytes)).convert("RGB"))
        with np.load(io.BytesIO(data_bytes)) as payload:
            values = {name: np.asarray(payload[name]).copy() for name in payload.files}
        rendered = decode_render_channels(values["rendered"])
        return {
            "rgb_uint8": torch.from_numpy(rgb.copy()),
            "observed_mask_uint8": torch.from_numpy(
                values["observed_mask"].astype(np.uint8)
            ),
            "rendered": torch.from_numpy(rendered),
            "center_targets": torch.from_numpy(values["center_targets"].astype(np.float32)),
            "log_depth_targets": torch.from_numpy(
                values["log_depth_targets"].astype(np.float32)
            ),
            "rotation_targets": torch.from_numpy(
                values["rotation_targets"].astype(np.float32)
            ),
            "correctable_targets": torch.from_numpy(
                values["correctable_targets"].astype(bool)
            ),
            "confidence_targets": torch.from_numpy(
                values["confidence_targets"].astype(np.float32)
            ),
            "quality_targets": torch.from_numpy(
                values["quality_targets"].astype(np.float32)
            ),
        }

    def _iter_shard(self, path: Path) -> Iterator[dict[str, torch.Tensor]]:
        pending: dict[str, dict[str, bytes]] = {}
        with tarfile.open(path) as tar:
            for member in tar:
                if not member.isfile():
                    continue
                if member.name.endswith(".rgb.jpg"):
                    key = member.name.removesuffix(".rgb.jpg")
                    suffix = "rgb"
                elif member.name.endswith(".data.npz"):
                    key = member.name.removesuffix(".data.npz")
                    suffix = "data"
                else:
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                pending.setdefault(key, {})[suffix] = handle.read()
                if {"rgb", "data"} <= pending[key].keys():
                    values = pending.pop(key)
                    yield self._decode(values["rgb"], values["data"])

    def _prepare(self, sample: dict[str, torch.Tensor], rng: np.random.Generator):
        rgb = sample.pop("rgb_uint8").numpy()
        mask = sample.pop("observed_mask_uint8").numpy()
        if self.training:
            rgb = _augment_rgb(rgb, rng)
            mask = _augment_mask(mask, rng)
        else:
            rgb = np.asarray(rgb, dtype=np.float32) / 255.0
        sample["rgb"] = torch.from_numpy(
            np.moveaxis(rgb.astype(np.float32), -1, 0)
        )
        sample["observed_mask"] = torch.from_numpy(
            mask.astype(np.float32)[None]
        )
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
                prepared = self._prepare(sample, rng)
                if not self.training or self.shuffle_buffer <= 1:
                    yield prepared
                    continue
                buffer.append(prepared)
                if len(buffer) >= self.shuffle_buffer:
                    index = int(rng.integers(0, len(buffer)))
                    yield buffer.pop(index)
        while buffer:
            index = int(rng.integers(0, len(buffer)))
            yield buffer.pop(index)
