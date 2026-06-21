"""Shared Assetto Corsa geometry, pose conversion, and rendering utilities."""

from __future__ import annotations

import os

# pyrender must see this before importing PyOpenGL in a headless session.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pyrender
import trimesh


CAMERA_SCENE_IDS = {
    "front": 1,
    "rear": 2,
    "stereo_left": 3,
    "stereo_right": 4,
}
AC_CAMERA_TO_OPENCV = np.diag([1.0, -1.0, 1.0])

# The shared CAD is X-longitudinal, Y-lateral, Z-up. Visual projection against
# the recorded front view confirms the mapping AC=(CAD-Y, CAD-Z, CAD-X) below.
# It is a proper rotation (det=+1), as required because the CSV body basis is
# right-handed even though its camera basis is left-handed.
DEFAULT_CAD_TO_AC_BODY = np.array(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float64
)


def parse_matrix(value: str | None) -> np.ndarray:
    """Parse nine comma-separated row-major values, or return the default."""
    if value is None:
        return DEFAULT_CAD_TO_AC_BODY.copy()
    numbers = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(numbers) != 9:
        raise ValueError("A coordinate matrix must contain nine comma-separated values.")
    matrix = np.asarray(numbers, dtype=np.float64).reshape(3, 3)
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5):
        raise ValueError("The CAD-to-AC matrix must be orthonormal.")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-5):
        raise ValueError("The CAD-to-AC matrix must be a proper rotation (det=+1).")
    return matrix


def load_centered_mesh(cad_path: Path, scale: float = 1.0) -> trimesh.Trimesh:
    mesh = trimesh.load(cad_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError(f"Could not load a non-empty mesh from {cad_path}")
    mesh = mesh.copy()
    mesh.apply_scale(scale)
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    return mesh


def read_grouped_rows(
    source_root: Path,
    cameras: Iterable[str],
    opponent_ids: set[int] | None = None,
) -> dict[tuple[str, int], list[dict[str, str]]]:
    csv_path = source_root / "csv" / "bboxes_3d.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing annotation CSV: {csv_path}")
    camera_set = set(cameras)
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    with csv_path.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            if row["camera_id"] not in camera_set:
                continue
            if opponent_ids is not None and int(row["opp_id"]) not in opponent_ids:
                continue
            grouped[(row["camera_id"], int(row["frame"]))].append(row)
    return grouped


def image_stem(row: dict[str, str]) -> str:
    return f"{int(row['sim_time_ms']):010d}_{int(row['frame']):06d}"


def intrinsics(row: dict[str, str]) -> np.ndarray:
    return np.array(
        [
            [float(row["fx"]), 0.0, float(row["cx"])],
            [0.0, float(row["fy"]), float(row["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def csv_bbox_xywh(row: dict[str, str]) -> list[int]:
    x0, y0 = int(row["xmin"]), int(row["ymin"])
    x1, y1 = int(row["xmax"]), int(row["ymax"])
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def _closest_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def cad_to_camera_pose(
    row: dict[str, str], cad_to_ac_body: np.ndarray
) -> np.ndarray:
    """Return a right-handed OpenCV CAD-to-camera transform in meters."""
    axis_matrix = np.column_stack(
        [
            [float(row[f"axis_{axis}_camera_{coord}"]) for coord in "xyz"]
            for axis in "xyz"
        ]
    )
    center_ac_camera = np.array(
        [float(row[f"center_camera_{coord}"]) for coord in "xyz"],
        dtype=np.float64,
    )
    rotation = AC_CAMERA_TO_OPENCV @ axis_matrix @ cad_to_ac_body
    rotation = _closest_rotation(rotation)
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError("Pose conversion produced an improper rotation.")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = AC_CAMERA_TO_OPENCV @ center_ac_camera
    return pose


def bbox_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def bbox_iou(first: list[int], second: list[int]) -> float:
    ax0, ay0, aw, ah = first
    bx0, by0, bw, bh = second
    ax1, ay1, bx1, by1 = ax0 + aw, ay0 + ah, bx0 + bw, by0 + bh
    intersection = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(
        0, min(ay1, by1) - max(ay0, by0)
    )
    union = aw * ah + bw * bh - intersection
    return float(intersection / union) if union else 0.0


class InstanceRenderer:
    """Render visible instance IDs and metric depth with pyrender/EGL."""

    def __init__(self, mesh: trimesh.Trimesh):
        self.mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        self.renderers: dict[tuple[int, int], pyrender.OffscreenRenderer] = {}

    def _renderer(self, width: int, height: int) -> pyrender.OffscreenRenderer:
        key = (width, height)
        if key not in self.renderers:
            self.renderers[key] = pyrender.OffscreenRenderer(width, height)
        return self.renderers[key]

    def render(
        self,
        poses_cv_m: list[np.ndarray],
        K: np.ndarray,
        width: int,
        height: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        scene = pyrender.Scene(bg_color=np.zeros(4), ambient_light=np.ones(4))
        camera = pyrender.IntrinsicsCamera(
            fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
            znear=0.05, zfar=1000.0,
        )
        # The world stores OpenCV coordinates; this camera pose converts the
        # OpenGL viewing convention (Y-up, -Z-forward) to OpenCV.
        camera_pose = np.diag([1.0, -1.0, -1.0, 1.0])
        scene.add(camera, pose=camera_pose)
        node_colors = {}
        for instance_index, pose in enumerate(poses_cv_m, start=1):
            if instance_index > 255:
                raise ValueError("The renderer supports at most 255 instances per image.")
            node = scene.add(self.mesh, pose=pose)
            node_colors[node] = np.array([instance_index, 0, 0], dtype=np.uint8)
        segmentation, depth_m = self._renderer(width, height).render(
            scene, flags=pyrender.RenderFlags.SEG, seg_node_map=node_colors
        )
        return segmentation[:, :, 0], depth_m

    def close(self) -> None:
        for renderer in self.renderers.values():
            renderer.delete()
        self.renderers.clear()
