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
        segmentation, depth_m = self.render_instances(
            [pose_m], K, image_shape
        )
        return segmentation == 1, depth_m

    def render_instances(
        self,
        poses_m: list[np.ndarray],
        K: np.ndarray,
        image_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Jointly render instances so nearer cars occlude farther ones."""

        height, width = (int(value) for value in image_shape)
        segmentation, depth_m = self._renderer.render(
            [np.asarray(pose, dtype=float) for pose in poses_m],
            np.asarray(K, dtype=float),
            width=width,
            height=height,
        )
        return np.asarray(segmentation), np.asarray(depth_m, dtype=np.float32)

    def close(self) -> None:
        self._renderer.close()

    def __enter__(self) -> "CADRenderer":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


class CPUSilhouetteRenderer:
    """Render CAD silhouettes with OpenCV, without EGL or a GPU context.

    This intentionally returns a silhouette and an empty depth image. It is
    suitable for diagnostic overlays where only the projected CAD footprint is
    needed.
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
        except ImportError as exc:
            raise ImportError(
                "CPU CAD overlays require trimesh in the GigaPose environment."
            ) from exc
        mesh = trimesh.load(Path(mesh_path), force="mesh")
        if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
            raise ValueError(f"Could not load a non-empty mesh from {mesh_path}")
        vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
        vertices *= float(mesh_scale)
        if center_mesh:
            vertices -= 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
        self.vertices = vertices
        self.faces = np.asarray(mesh.faces, dtype=np.int64)

    def render(
        self,
        pose_m: np.ndarray,
        K: np.ndarray,
        image_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        import cv2

        height, width = (int(value) for value in image_shape)
        pose = np.asarray(pose_m, dtype=np.float64).reshape(4, 4)
        intrinsics = np.asarray(K, dtype=np.float64).reshape(3, 3)
        camera_vertices = (
            pose[:3, :3] @ self.vertices.T
        ).T + pose[:3, 3]
        z = camera_vertices[:, 2]
        projected = np.zeros((len(camera_vertices), 2), dtype=np.float64)
        positive = z > 1e-5
        projected[positive, 0] = (
            intrinsics[0, 0] * camera_vertices[positive, 0] / z[positive]
            + intrinsics[0, 2]
        )
        projected[positive, 1] = (
            intrinsics[1, 1] * camera_vertices[positive, 1] / z[positive]
            + intrinsics[1, 2]
        )
        valid_faces = self.faces[np.all(positive[self.faces], axis=1)]
        mask = np.zeros((height, width), dtype=np.uint8)
        if len(valid_faces):
            # Bounding projected coordinates avoids int32 overflow for
            # near-camera triangles while preserving anything visible.
            limit = 8 * max(height, width)
            points = np.clip(
                np.rint(projected[valid_faces]), -limit, limit
            ).astype(np.int32)
            for start in range(0, len(points), 10000):
                cv2.fillPoly(mask, points[start : start + 10000], 1)
        return mask.astype(bool), np.zeros((height, width), dtype=np.float32)

    def close(self) -> None:
        pass
