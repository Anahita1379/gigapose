"""Load per-frame map/LiDAR and fixed LiDAR/camera transforms without changing core I/O."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def _pose(value, unit="m"):
    if isinstance(value, dict):
        for key in ("matrix", "data", "T", "transform"):
            if key in value:
                value = value[key]
                break
    if isinstance(value, str):
        value = np.fromstring(value, sep=" ")
    result = np.asarray(value, dtype=float)
    if result.size == 12:
        output = np.eye(4)
        output[:3] = result.reshape(3, 4)
    elif result.size == 16:
        output = result.reshape(4, 4).copy()
    else:
        raise ValueError(f"Expected a 3x4/4x4 transform, got shape {result.shape}")
    if unit == "mm":
        output[:3, 3] *= .001
    elif unit != "m":
        raise ValueError(f"Unsupported transform unit {unit}")
    return output


def _yaml(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Map metadata needs PyYAML") from exc
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Metadata is not a mapping: {path}")
    return value


class MapTransformProvider:
    def __init__(self, args):
        self.unit = args.map_transform_unit
        self.metadata_root = args.map_metadata_root
        self.fixed_extrinsic = None
        if args.map_extrinsics is not None:
            data = json.loads(args.map_extrinsics.read_text())
            value = data.get("T_lidar_camera_optimized", data.get("T_lidar_camera"))
            if value is None:
                raise KeyError(f"{args.map_extrinsics} has no LiDAR-camera transform")
            self.fixed_extrinsic = _pose(value, "m")
        self.cache = {}

    def _metadata(self, frame):
        row = frame.metadata
        path_text = row.get("sample_metadata_path") or row.get("metadata_path")
        if not path_text:
            return row
        path = Path(path_text)
        if self.metadata_root is not None:
            path = self.metadata_root / path.name
        if path not in self.cache:
            self.cache[path] = _yaml(path)
        return self.cache[path]

    def map_camera(self, frame):
        metadata = self._metadata(frame)
        map_value = metadata.get("T_map_lidar", metadata.get("t_map_lidar"))
        if map_value is None:
            return None, "missing_T_map_lidar"
        extrinsic = self.fixed_extrinsic
        if extrinsic is None:
            value = metadata.get(
                "T_lidar_camera", metadata.get("t_lidar_camera_prior")
            )
            if value is None:
                return None, "missing_T_lidar_camera"
            extrinsic = _pose(value, self.unit)
        return _pose(map_value, self.unit) @ extrinsic, "available"
