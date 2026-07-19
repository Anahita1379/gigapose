"""Lazy CAD renderer used by tracking and recovery-data generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class CADRenderer:
    """Render one OpenCV camera-from-object pose into a mask and metric depth.

    Heavy EGL/pyrender imports are deliberately delayed until construction so
    geometry, association, configuration, and unit tests remain usable on
    machines without a rendering context.
    """

    def __init__(
        self,
        mesh_path: Path,
        *,
        mesh_scale: float = 1.0,
        center_mesh: bool = False,
    ):
        try:
            import trimesh
            from fine_tuning.ac_geometry import InstanceRenderer
        except ImportError as exc:
            raise ImportError(
                "CAD tracking requires trimesh, pyrender, and PyOpenGL in the "
                "GigaPose environment."
            ) from exc
        mesh = trimesh.load(Path(mesh_path), force="mesh")
        if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
            raise ValueError(f"Could not load a non-empty mesh from {mesh_path}")
        mesh = mesh.copy()
        mesh.apply_scale(float(mesh_scale))
        if center_mesh:
            mesh.apply_translation(-mesh.bounds.mean(axis=0))
        self.mesh_path = Path(mesh_path)
        self.mesh_scale = float(mesh_scale)
        self.center_mesh = bool(center_mesh)
        self.mesh_extents_m = np.asarray(mesh.extents, dtype=float)
        self._renderer = InstanceRenderer(mesh)

    def render(
        self,
        pose_m: np.ndarray,
        K: np.ndarray,
        image_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = (int(value) for value in image_shape)
        segmentation, depth_m = self._renderer.render(
            [np.asarray(pose_m, dtype=float)],
            np.asarray(K, dtype=float),
            width=width,
            height=height,
        )
        return segmentation == 1, np.asarray(depth_m, dtype=np.float32)

    def close(self) -> None:
        self._renderer.close()

    def __enter__(self) -> "CADRenderer":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
