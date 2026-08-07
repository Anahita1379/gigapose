"""Reusable whole-segment GRU inference for first-pass and replay runs."""

from __future__ import annotations

import numpy as np
import torch


def run_filter_segments(model, arrays, device, *, measurement_pose=None):
    measurements = (
        arrays["measurement_pose"] if measurement_pose is None else measurement_pose
    )
    if measurements.shape != arrays["measurement_pose"].shape:
        raise ValueError("Replacement measurements have the wrong shape")
    count = len(measurements)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0)
    gains = np.zeros((count, 12), dtype=np.float32)
    with torch.no_grad():
        for segment_id in np.unique(arrays["segment_id"]):
            indices = np.flatnonzero(arrays["segment_id"] == segment_id)
            measurement = torch.from_numpy(
                np.asarray(measurements[indices], dtype=np.float32)
            )[None].to(device)
            context = torch.from_numpy(
                arrays["context_features"][indices]
            )[None].to(device)
            delta_time = torch.from_numpy(
                arrays["delta_time_s"][indices].astype(np.float32)
            )[None].to(device)
            delta_time[:, 0] = 0.0
            valid = torch.ones(1, len(indices), dtype=torch.bool, device=device)
            baseline = torch.from_numpy(
                arrays["baseline_pose"][indices]
            )[None].to(device)
            result = model(
                measurement,
                context,
                delta_time,
                valid,
                baseline_pose=baseline if model.use_fallback_heads else None,
            )
            poses[indices] = result[0][0].cpu().numpy()
            gains[indices] = result[1][0].cpu().numpy()
    return poses, gains
