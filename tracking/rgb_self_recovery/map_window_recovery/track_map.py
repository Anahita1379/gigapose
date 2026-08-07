"""Local, read-only Frenet projection for prepared teacher track maps."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def _unit(value):
    value = np.asarray(value, dtype=float)
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-12)


class TrackMap:
    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as data:
            self.center = np.asarray(data["center_xyz"], dtype=float)
            self.s = np.asarray(data["s"], dtype=float)
            self.tangent = _unit(data["tangent_xyz"])
            self.normal = _unit(data["normal_xyz"])
            self.lateral = _unit(data["lateral_xyz"])
            count = len(self.s)
            self.left = np.asarray(data["left_width"] if "left_width" in data else np.full(count, 6.), dtype=float)
            self.right = np.asarray(data["right_width"] if "right_width" in data else np.full(count, 6.), dtype=float)
            self.closed = bool(np.asarray(data["closed"]).item()) if "closed" in data else True
            self.length = float(np.asarray(data["track_length_m"]).item()) if "track_length_m" in data else float(self.s[-1])
        if len(self.s) < 4 or not np.all(np.diff(self.s) > 0):
            raise ValueError("Track map needs >=4 samples with increasing s")
        self.tree = cKDTree(self.center)

    def signed_delta(self, first, second):
        value = float(first) - float(second)
        if self.closed:
            value = (value + .5 * self.length) % self.length - .5 * self.length
        return value

    def project(self, point, reference_s=None):
        distances, indices = self.tree.query(np.asarray(point, float), k=min(12, len(self.s)))
        distances, indices = np.atleast_1d(distances), np.atleast_1d(indices)
        costs = distances.copy()
        if reference_s is not None:
            costs += .05 * np.abs([self.signed_delta(self.s[i], reference_s) for i in indices])
        index = int(indices[int(np.argmin(costs))])
        delta = np.asarray(point, float) - self.center[index]
        return {
            "s": float(self.s[index]), "d": float(delta @ self.lateral[index]),
            "h": float(delta @ self.normal[index]), "index": index,
            "distance": float(np.linalg.norm(delta)), "tangent": self.tangent[index],
            "left_width": float(self.left[index]), "right_width": float(self.right[index]),
        }
