"""Searchable closed-track coordinates and pose reconstruction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def _unit(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _rz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class TrackProjection:
    s: float
    d: float
    h: float
    index: int
    distance: float


class TrackMap:
    def __init__(self, path: str | Path):
        data = np.load(path, allow_pickle=False)
        self.center = np.asarray(data["center_xyz"], dtype=float)
        self.s = np.asarray(data["s"], dtype=float).reshape(-1)
        self.tangent = _unit(np.asarray(data["tangent_xyz"], dtype=float))
        self.normal = _unit(np.asarray(data["normal_xyz"], dtype=float))
        self.lateral = _unit(np.asarray(data["lateral_xyz"], dtype=float))
        count = len(self.center)
        self.left_width = np.asarray(
            data["left_width"] if "left_width" in data else np.full(count, 6.0),
            dtype=float,
        )
        self.right_width = np.asarray(
            data["right_width"] if "right_width" in data else np.full(count, 6.0),
            dtype=float,
        )
        self.closed = bool(
            np.asarray(data["closed"]).item() if "closed" in data else True
        )
        self.length = float(
            np.asarray(data["track_length_m"]).item()
            if "track_length_m" in data
            else self.s[-1]
        )
        if count < 4 or not np.all(np.diff(self.s) > 0):
            raise ValueError("Track map requires >=4 samples with increasing s")
        self.tree = cKDTree(self.center)

    def wrap_s(self, value):
        return np.mod(value, self.length) if self.closed else np.clip(value, 0, self.length)

    def signed_s_delta(self, first, second):
        delta = np.asarray(first) - np.asarray(second)
        if self.closed:
            delta = (delta + 0.5 * self.length) % self.length - 0.5 * self.length
        return delta

    def project(self, point, reference_s=None, candidates=12) -> TrackProjection:
        point = np.asarray(point, dtype=float).reshape(3)
        distances, indices = self.tree.query(point, k=min(candidates, len(self.s)))
        distances, indices = np.atleast_1d(distances), np.atleast_1d(indices)
        costs = distances.copy()
        if reference_s is not None:
            costs += 0.05 * np.abs(self.signed_s_delta(self.s[indices], reference_s))
        chosen = int(indices[int(np.argmin(costs))])
        delta = point - self.center[chosen]
        return TrackProjection(
            s=float(self.s[chosen]),
            d=float(delta @ self.lateral[chosen]),
            h=float(delta @ self.normal[chosen]),
            index=chosen,
            distance=float(np.linalg.norm(delta)),
        )

    def sample(self, s_value):
        value = float(self.wrap_s(s_value))
        query_s = self.s
        def interp(array):
            if self.closed:
                xp = np.r_[query_s, self.length]
                values = np.concatenate([array, array[:1]], axis=0)
            else:
                xp, values = query_s, array
            if values.ndim == 1:
                return float(np.interp(value, xp, values))
            return np.array([np.interp(value, xp, values[:, j]) for j in range(values.shape[1])])
        tangent = _unit(interp(self.tangent)[None])[0]
        normal = _unit(interp(self.normal)[None])[0]
        lateral = _unit(np.cross(normal, tangent)[None])[0]
        normal = _unit(np.cross(tangent, lateral)[None])[0]
        return {
            "center": interp(self.center), "tangent": tangent,
            "lateral": lateral, "normal": normal,
            "left_width": interp(self.left_width),
            "right_width": interp(self.right_width),
        }

    def pose(self, s_value, d, delta_yaw, alignment):
        sample = self.sample(s_value)
        frame = np.column_stack(
            [sample["tangent"], sample["lateral"], sample["normal"]]
        )
        output = np.eye(4)
        output[:3, :3] = frame @ _rz(float(delta_yaw)) @ alignment
        output[:3, 3] = sample["center"] + float(d) * sample["lateral"]
        return output


def unwrap_progress(values: np.ndarray, length: float) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    for index in range(1, len(values)):
        delta = values[index] - values[index - 1]
        if delta > 0.5 * length:
            values[index:] -= length
        elif delta < -0.5 * length:
            values[index:] += length
    return values
