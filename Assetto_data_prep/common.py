"""Shared parsing, geometry, rendering, and WebDataset helpers.

The 20260623 recorder uses the convention ``T_A_B`` maps coordinates from B
to A. Its camera frame is X-right, Y-up, Z-forward and is therefore
left-handed. GigaPose uses a right-handed OpenCV camera frame, so every pose is
premultiplied by ``diag(1, -1, 1)``.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from PIL import Image


CAMERA_SCENE_IDS = {
    "front": 1,
    "rear": 2,
    "stereo_left": 3,
    "stereo_right": 4,
}
AC_CAMERA_TO_OPENCV = np.diag([1.0, -1.0, 1.0])


@dataclass(frozen=True)
class FrameGroup:
    session_index: int
    source_root: Path
    camera_id: str
    frame_id: int
    camera_row: dict[str, str]
    rows: tuple[dict[str, str], ...]

    @property
    def scene_id(self) -> int:
        return self.session_index * 100 + CAMERA_SCENE_IDS[self.camera_id]

    @property
    def sim_time_ms(self) -> int:
        return int(self.camera_row["sim_time_ms"])

    @property
    def stem(self) -> str:
        return f"{self.sim_time_ms:010d}_{self.frame_id:06d}"

    @property
    def key(self) -> str:
        return f"{self.scene_id:06d}_{self.frame_id:06d}"

    @property
    def image_path(self) -> Path:
        return self.source_root / "images" / self.camera_id / f"{self.stem}.jpg"

    def mask_path(self, mask_dir_name: str = "generated_masks") -> Path:
        return self.source_root / mask_dir_name / self.camera_id / f"{self.stem}.png"


@dataclass(frozen=True)
class ModelAlignment:
    """A centered mesh plus the center expressed in opponent-local CSP axes."""

    mesh: object
    local_center: np.ndarray
    aabb_center: np.ndarray
    aabb_size: np.ndarray
    axis_convention: str
    fit_aabb: str
    cad_scale: float


def parse_matrix4(value: str, name: str = "matrix") -> np.ndarray:
    if not value:
        raise ValueError(f"Matrix field {name} is empty")
    cleaned = value.strip().strip('"').strip("[]").replace(";", " ")
    numbers = np.fromstring(cleaned, sep=" ", dtype=np.float64)
    if numbers.size != 16:
        raise ValueError(f"Matrix field {name} has {numbers.size} values; expected 16")
    return numbers.reshape(4, 4)


def parse_cameras(value: str) -> list[str]:
    cameras = list(CAMERA_SCENE_IDS) if value.strip().lower() == "all" else [
        item.strip() for item in value.split(",") if item.strip()
    ]
    unknown = sorted(set(cameras) - set(CAMERA_SCENE_IDS))
    if unknown:
        raise ValueError(f"Unknown cameras {unknown}; valid values: {list(CAMERA_SCENE_IDS)}")
    if not cameras:
        raise ValueError("At least one camera must be selected")
    return cameras


def parse_ids(value: str | None) -> set[int] | None:
    if value is None:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def _required_columns(path: Path, actual: Iterable[str] | None, required: set[str]) -> None:
    missing = required - set(actual or [])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")


def read_session_frames(
    source_root: Path,
    session_index: int,
    cameras: Iterable[str],
    opponent_ids: set[int] | None = None,
    exclude_truncated: bool = False,
) -> list[FrameGroup]:
    """Join camera_frames.csv and visible transforms.csv annotations."""
    camera_set = set(cameras)
    camera_path = source_root / "csv" / "camera_frames.csv"
    transforms_path = source_root / "csv" / "transforms.csv"
    for path in (camera_path, transforms_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing new-schema annotation file: {path}")

    camera_rows: dict[tuple[str, int], dict[str, str]] = {}
    with camera_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _required_columns(
            camera_path,
            reader.fieldnames,
            {"frame", "sim_time_ms", "camera_id", "image_width", "image_height",
             "fx", "fy", "cx", "cy"},
        )
        for row in reader:
            if row["camera_id"] in camera_set:
                camera_rows[(row["camera_id"], int(row["frame"]))] = row

    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    with transforms_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _required_columns(
            transforms_path,
            reader.fieldnames,
            {"frame", "sim_time_ms", "camera_id", "opp_id", "visible", "truncated",
             "xmin", "ymin", "xmax", "ymax", "T_camera_opponent_visual",
             "aabb_center_local_x", "aabb_center_local_y", "aabb_center_local_z",
             "aabb_size_x", "aabb_size_y", "aabb_size_z"},
        )
        for row in reader:
            if row["camera_id"] not in camera_set or row["visible"] != "1":
                continue
            if not row["xmin"]:
                continue
            if opponent_ids is not None and int(row["opp_id"]) not in opponent_ids:
                continue
            if exclude_truncated and int(row["truncated"]):
                continue
            grouped[(row["camera_id"], int(row["frame"]))].append(row)

    frames: list[FrameGroup] = []
    for (camera_id, frame_id), rows in grouped.items():
        camera_row = camera_rows.get((camera_id, frame_id))
        if camera_row is None:
            raise ValueError(
                f"No camera_frames.csv row for {source_root}, {camera_id}, frame {frame_id}"
            )
        if int(camera_row["sim_time_ms"]) != int(rows[0]["sim_time_ms"]):
            raise ValueError(f"Timestamp mismatch for {source_root}, {camera_id}, {frame_id}")
        frames.append(
            FrameGroup(
                session_index=session_index,
                source_root=source_root,
                camera_id=camera_id,
                frame_id=frame_id,
                camera_row=camera_row,
                rows=tuple(sorted(rows, key=lambda item: int(item["opp_id"]))),
            )
        )
    frames.sort(key=lambda item: (item.frame_id, CAMERA_SCENE_IDS[item.camera_id]))
    return frames


def select_session_frames(
    frames: list[FrameGroup], frame_stride: int, max_frames: int | None
) -> list[FrameGroup]:
    if frame_stride < 1:
        raise ValueError("--frame-stride must be at least one")
    frame_ids = sorted({frame.frame_id for frame in frames})[::frame_stride]
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError("--max-frames-per-session must be positive")
        frame_ids = frame_ids[:max_frames]
    selected = set(frame_ids)
    return [frame for frame in frames if frame.frame_id in selected]


def collect_sessions(
    source_roots: Iterable[Path],
    cameras: Iterable[str],
    opponent_ids: set[int] | None,
    exclude_truncated: bool,
    frame_stride: int,
    max_frames_per_session: int | None,
) -> list[list[FrameGroup]]:
    sessions = []
    for index, root in enumerate(source_roots):
        frames = read_session_frames(
            root, index, cameras, opponent_ids, exclude_truncated
        )
        frames = select_session_frames(frames, frame_stride, max_frames_per_session)
        if not frames:
            raise RuntimeError(f"No visible annotated frames selected from {root}")
        sessions.append(frames)
    return sessions


def intrinsics(frame: FrameGroup) -> np.ndarray:
    row = frame.camera_row
    return np.array(
        [[float(row["fx"]), 0.0, float(row["cx"])],
         [0.0, float(row["fy"]), float(row["cy"])],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def image_size(frame: FrameGroup) -> tuple[int, int]:
    return int(frame.camera_row["image_width"]), int(frame.camera_row["image_height"])


def csv_bbox_xywh(row: dict[str, str]) -> list[int]:
    x0, y0 = int(float(row["xmin"])), int(float(row["ymin"]))
    x1, y1 = int(float(row["xmax"])), int(float(row["ymax"]))
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def bbox_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if not len(xs):
        return [0, 0, 0, 0]
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def aabb(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    center = np.array([float(row[f"aabb_center_local_{c}"]) for c in "xyz"])
    size = np.array([float(row[f"aabb_size_{c}"]) for c in "xyz"])
    return center, size


def build_aligned_mesh(
    cad_path: Path,
    reference_row: dict[str, str],
    cad_scale: float = 1.0,
    axis_convention: str = "x-forward-z-up",
    fit_aabb: str = "nonuniform",
) -> ModelAlignment:
    """Apply projection.py's model alignment, then center the exported mesh."""
    import trimesh

    mesh = trimesh.load(cad_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError(f"Could not load a non-empty triangular mesh from {cad_path}")
    mesh = mesh.copy()
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * cad_scale
    if axis_convention == "x-forward-z-up":
        vertices = vertices[:, [1, 2, 0]]
    elif axis_convention != "csp":
        raise ValueError("axis_convention must be 'x-forward-z-up' or 'csp'")

    target_center, target_size = aabb(reference_row)
    if fit_aabb not in {"none", "uniform", "nonuniform"}:
        raise ValueError("fit_aabb must be none, uniform, or nonuniform")
    if fit_aabb != "none":
        mesh_min, mesh_max = vertices.min(axis=0), vertices.max(axis=0)
        mesh_size = mesh_max - mesh_min
        if np.any(mesh_size < 1e-9):
            raise ValueError("CAD bounds are degenerate")
        scale = target_size / mesh_size
        if fit_aabb == "uniform":
            scale[:] = np.min(scale)
        vertices = (vertices - (mesh_min + mesh_max) * 0.5) * scale + target_center

    local_center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    mesh.vertices = vertices - local_center
    return ModelAlignment(
        mesh=mesh,
        local_center=local_center,
        aabb_center=target_center,
        aabb_size=target_size,
        axis_convention=axis_convention,
        fit_aabb=fit_aabb,
        cad_scale=cad_scale,
    )


def validate_model_row(alignment: ModelAlignment, row: dict[str, str], atol: float = 1e-4) -> None:
    center, size = aabb(row)
    if alignment.fit_aabb != "none" and (
        not np.allclose(center, alignment.aabb_center, atol=atol)
        or not np.allclose(size, alignment.aabb_size, atol=atol)
    ):
        raise ValueError(
            "Opponent AABB differs from the reference CAD alignment. This likely "
            "means multiple car geometries are present and require separate object IDs."
        )


def cad_to_camera_pose(row: dict[str, str], local_center: np.ndarray) -> np.ndarray:
    """Return centered-CAD-to-OpenCV-camera pose in meters."""
    transform = parse_matrix4(
        row["T_camera_opponent_visual"], "T_camera_opponent_visual"
    )
    rotation = AC_CAMERA_TO_OPENCV @ transform[:3, :3]
    translation_ac = transform[:3, :3] @ local_center + transform[:3, 3]
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = AC_CAMERA_TO_OPENCV @ translation_ac
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5):
        raise ValueError("Converted pose rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-5):
        raise ValueError("Converted pose rotation is not right-handed")
    return pose


def encode_instance_ids(instance_ids: np.ndarray) -> np.ndarray:
    values = instance_ids.astype(np.uint32)
    rgb = np.empty((*values.shape, 3), dtype=np.uint8)
    rgb[..., 0] = values & 255
    rgb[..., 1] = (values >> 8) & 255
    rgb[..., 2] = (values >> 16) & 255
    return rgb


def decode_instance_ids(path: Path, expected_size: tuple[int, int]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing generated mask {path}. Run generate_masks.py first."
        )
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint32)
    if (rgb.shape[1], rgb.shape[0]) != expected_size:
        raise ValueError(f"Mask size mismatch: {path}")
    return rgb[..., 0] + (rgb[..., 1] << 8) + (rgb[..., 2] << 16)


def png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def split_sessions(
    sessions: list[list[FrameGroup]],
    validation_sessions: int,
    validation_fraction: float,
    gap_frames: int,
) -> tuple[list[FrameGroup], list[FrameGroup], dict[str, object]]:
    if len(sessions) > 1:
        if not 1 <= validation_sessions < len(sessions):
            raise ValueError("--validation-sessions must leave a training session")
        train = [frame for session in sessions[:-validation_sessions] for frame in session]
        val = [frame for session in sessions[-validation_sessions:] for frame in session]
        policy = {
            "type": "held_out_sessions",
            "training_session_indices": list(range(len(sessions) - validation_sessions)),
            "validation_session_indices": list(
                range(len(sessions) - validation_sessions, len(sessions))
            ),
        }
    else:
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("--validation-fraction must be between zero and one")
        ids = sorted({frame.frame_id for frame in sessions[0]})
        split = max(1, math.floor(len(ids) * (1.0 - validation_fraction)))
        train_ids = set(ids[: max(0, split - gap_frames)])
        val_ids = set(ids[split:])
        train = [frame for frame in sessions[0] if frame.frame_id in train_ids]
        val = [frame for frame in sessions[0] if frame.frame_id in val_ids]
        policy = {
            "type": "contiguous_validation_tail",
            "training_last_frame": max(train_ids) if train_ids else None,
            "validation_first_frame": min(val_ids) if val_ids else None,
            "excluded_gap_frames": gap_frames,
        }
    if not train or not val:
        raise RuntimeError("Training/validation split produced an empty partition")
    return train, val, policy


def model_info(mesh: object) -> dict[str, float]:
    bounds = np.asarray(mesh.bounds)
    size = bounds[1] - bounds[0]
    return {
        "diameter": float(np.linalg.norm(size)),
        "min_x": float(bounds[0, 0]), "min_y": float(bounds[0, 1]),
        "min_z": float(bounds[0, 2]), "size_x": float(size[0]),
        "size_y": float(size[1]), "size_z": float(size[2]),
    }


def write_models(alignment: ModelAlignment, dataset_dir: Path, object_id: int) -> None:
    models = dataset_dir / "models"
    models.mkdir(parents=True, exist_ok=True)
    alignment.mesh.export(models / f"obj_{object_id:06d}.ply")
    alignment.mesh.export(models / f"obj_{object_id:06d}.obj")
    (models / "models_info.json").write_text(
        json.dumps({str(object_id): model_info(alignment.mesh)}, indent=2)
    )
    (dataset_dir / "models_info.json").write_text(
        json.dumps([{"obj_id": object_id}], indent=2)
    )


def key_to_shard_map(split_dir: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for shard in sorted(split_dir.glob("shard-*.tar")):
        shard_id = int(re.findall(r"\d+", shard.name)[0])
        with tarfile.open(shard) as archive:
            for key in {member.name.split(".")[0] for member in archive.getmembers()}:
                mapping[key] = shard_id
    return mapping


def iter_frames(sessions: Iterable[Iterable[FrameGroup]]) -> Iterator[FrameGroup]:
    for session in sessions:
        yield from session
