"""Render supervision and build GigaPose train/validation WebDataset shards."""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import webdataset as wds
from bop_toolkit_lib import pycoco_utils
from PIL import Image

from fine_tuning.ac_geometry import (
    CAMERA_SCENE_IDS,
    InstanceRenderer,
    bbox_from_mask,
    cad_to_camera_pose,
    csv_bbox_xywh,
    image_stem,
    instance_mask_for_row,
    instance_mask_path,
    intrinsics,
    load_centered_mesh,
    load_instance_ids,
    parse_matrix,
    read_grouped_rows,
)


DEFAULT_SOURCE = Path(
    "/mnt/ssd2tb/.local_share_backup/Steam/steamapps/common/assettocorsa/"
    "apps/lua/multi_cam_obs/frames/20260620_haze_3opp_withInstanceMask"
)
DEFAULT_CAD = Path("gigaPose_datasets/datasets/racecar/models/obj_000001.ply")


@dataclass(frozen=True)
class AnnotatedFrame:
    session_index: int
    source_root: Path
    camera_id: str
    frame_id: int
    rows: list[dict[str, str]]

    @property
    def scene_id(self) -> int:
        return self.session_index * 100 + CAMERA_SCENE_IDS[self.camera_id]

    @property
    def key(self) -> str:
        return f"{self.scene_id:06d}_{self.frame_id:06d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        action="append",
        help="Recording root; repeat for sessions. Defaults to the corrected 20260620 session.",
    )
    parser.add_argument("--cad-path", type=Path, default=DEFAULT_CAD)
    parser.add_argument(
        "--output-root", type=Path, default=Path("gigaPose_datasets/datasets")
    )
    parser.add_argument("--dataset-name", default="assettocorsa")
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--cameras", default="front")
    parser.add_argument("--opponent-ids", default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-session", type=int, default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--validation-sessions",
        type=int,
        default=1,
        help="With multiple recordings, hold out this many trailing sessions.",
    )
    parser.add_argument(
        "--gap-frames",
        type=int,
        default=50,
        help="For one recording, exclude this many frames before the validation tail.",
    )
    parser.add_argument("--max-shard-size", type=int, default=250)
    parser.add_argument("--cad-scale", type=float, default=1.0)
    parser.add_argument(
        "--min-mask-overlap",
        type=float,
        default=0.25,
        help=(
            "Skip an instance when the observed/CAD-rendered mask intersection "
            "covers less than this fraction of the observed mask."
        ),
    )
    parser.add_argument(
        "--cad-to-ac-body",
        default=None,
        help="Optional row-major 3x3 override as nine comma-separated values.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_names(value: str, allowed: set[str]) -> list[str]:
    names = list(allowed) if value.strip().lower() == "all" else [
        part.strip() for part in value.split(",") if part.strip()
    ]
    unknown = set(names) - allowed
    if unknown:
        raise ValueError(f"Unknown values: {sorted(unknown)}; allowed: {sorted(allowed)}")
    return names


def parse_ids(value: str | None) -> set[int] | None:
    if value is None:
        return None
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def collect_frames(args: argparse.Namespace) -> list[list[AnnotatedFrame]]:
    source_roots = args.source_root or [DEFAULT_SOURCE]
    cameras = parse_names(args.cameras, set(CAMERA_SCENE_IDS))
    opponent_ids = parse_ids(args.opponent_ids)
    sessions = []
    for session_index, source_root in enumerate(source_roots):
        grouped = read_grouped_rows(source_root, cameras, opponent_ids)
        frame_ids = sorted({frame for _, frame in grouped})
        if args.frame_stride < 1:
            raise ValueError("--frame-stride must be at least one.")
        frame_ids = frame_ids[:: args.frame_stride]
        if args.max_frames_per_session is not None:
            frame_ids = frame_ids[: args.max_frames_per_session]
        selected = set(frame_ids)
        frames = [
            AnnotatedFrame(session_index, source_root, camera, frame, rows)
            for (camera, frame), rows in grouped.items()
            if frame in selected
        ]
        frames.sort(key=lambda item: (item.frame_id, CAMERA_SCENE_IDS[item.camera_id]))
        if not frames:
            raise RuntimeError(f"No annotated frames selected from {source_root}")
        sessions.append(frames)
    return sessions


def split_frames(
    sessions: list[list[AnnotatedFrame]],
    validation_fraction: float,
    validation_sessions: int,
    gap_frames: int,
) -> tuple[list[AnnotatedFrame], list[AnnotatedFrame], dict[str, object]]:
    if len(sessions) > 1:
        if validation_sessions < 1 or validation_sessions >= len(sessions):
            raise ValueError(
                "--validation-sessions must leave at least one training session."
            )
        train_sessions = sessions[:-validation_sessions]
        val_sessions = sessions[-validation_sessions:]
        train = [frame for session in train_sessions for frame in session]
        val = [frame for session in val_sessions for frame in session]
        policy = {
            "type": "held_out_sessions",
            "training_session_indices": list(range(len(train_sessions))),
            "validation_session_indices": list(range(len(train_sessions), len(sessions))),
        }
    else:
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("--validation-fraction must be between zero and one.")
        unique_ids = sorted({frame.frame_id for frame in sessions[0]})
        split_index = max(1, int(math.floor(len(unique_ids) * (1.0 - validation_fraction))))
        validation_ids = set(unique_ids[split_index:])
        train_end = max(0, split_index - gap_frames)
        training_ids = set(unique_ids[:train_end])
        train = [frame for frame in sessions[0] if frame.frame_id in training_ids]
        val = [frame for frame in sessions[0] if frame.frame_id in validation_ids]
        policy = {
            "type": "contiguous_validation_tail",
            "training_last_frame": max(training_ids) if training_ids else None,
            "validation_first_frame": min(validation_ids) if validation_ids else None,
            "excluded_gap_frames": gap_frames,
        }
    if not train or not val:
        raise RuntimeError(
            "The split produced an empty partition. Reduce --gap-frames or provide more data."
        )
    return train, val, policy


def png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def model_info(mesh: trimesh.Trimesh) -> dict[str, float]:
    bounds = mesh.bounds
    size = bounds[1] - bounds[0]
    return {
        "diameter": float(np.linalg.norm(size)),
        "min_x": float(bounds[0, 0]),
        "min_y": float(bounds[0, 1]),
        "min_z": float(bounds[0, 2]),
        "size_x": float(size[0]),
        "size_y": float(size[1]),
        "size_z": float(size[2]),
    }


def write_models(mesh: trimesh.Trimesh, dataset_dir: Path, object_id: int) -> None:
    model_dir = dataset_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    mesh.export(model_dir / f"obj_{object_id:06d}.ply")
    mesh.export(model_dir / f"obj_{object_id:06d}.obj")
    (model_dir / "models_info.json").write_text(
        json.dumps({str(object_id): model_info(mesh)}, indent=2)
    )
    # The training loader uses the MegaPose list-shaped metadata at dataset root.
    (dataset_dir / "models_info.json").write_text(
        json.dumps([{"obj_id": object_id}], indent=2)
    )


def key_to_shard(split_dir: Path) -> dict[str, int]:
    mapping = {}
    for shard_path in sorted(split_dir.glob("shard-*.tar")):
        shard_id = int(re.findall(r"\d+", shard_path.name)[0])
        with tarfile.open(shard_path) as tar:
            for key in {name.split(".")[0] for name in tar.getnames()}:
                mapping[key] = shard_id
    return mapping


def validate_rows(frame: AnnotatedFrame, image: Image.Image) -> None:
    expected = (int(frame.rows[0]["image_width"]), int(frame.rows[0]["image_height"]))
    if image.size != expected:
        raise ValueError(f"Image size mismatch for {frame.key}: {image.size} != {expected}")
    calibration = tuple(frame.rows[0][key] for key in ("fx", "fy", "cx", "cy"))
    for row in frame.rows:
        if tuple(row[key] for key in ("fx", "fy", "cx", "cy")) != calibration:
            raise ValueError(f"Inconsistent intrinsics for {frame.key}")


def write_split(
    frames: list[AnnotatedFrame],
    split_dir: Path,
    mesh_renderer: InstanceRenderer,
    cad_to_ac_body: np.ndarray,
    object_id: int,
    max_shard_size: int,
    min_mask_overlap: float,
) -> tuple[int, int, list[dict[str, object]]]:
    split_dir.mkdir(parents=True)
    writer = wds.ShardWriter(
        pattern=str(split_dir / "shard-%06d.tar"),
        maxcount=max_shard_size,
        encoder=False,
    )
    samples_written = 0
    instances_written = 0
    manifest = []
    try:
        for frame in frames:
            rows = sorted(frame.rows, key=lambda row: int(row["opp_id"]))
            first = rows[0]
            stem = image_stem(first)
            image_path = frame.source_root / "images" / frame.camera_id / f"{stem}.jpg"
            image = Image.open(image_path).convert("RGB")
            validate_rows(frame, image)
            width, height = image.size
            K = intrinsics(first)
            poses_m = [cad_to_camera_pose(row, cad_to_ac_body) for row in rows]
            segmentation, depth_m = mesh_renderer.render(poses_m, K, width, height)
            observed_instance_ids = load_instance_ids(
                instance_mask_path(frame.source_root, frame.camera_id, first),
                image.size,
            )

            gt, gt_info, visible_masks, kept_instances = [], [], {}, []
            for render_id, (row, pose_m) in enumerate(zip(rows, poses_m), start=1):
                observed_mask = instance_mask_for_row(observed_instance_ids, row)
                rendered_mask = segmentation == render_id
                mask = np.logical_and(observed_mask, rendered_mask)
                observed_pixels = int(observed_mask.sum())
                rendered_pixels = int(rendered_mask.sum())
                overlap_fraction = (
                    float(mask.sum() / observed_pixels) if observed_pixels else 0.0
                )
                if overlap_fraction < min_mask_overlap:
                    continue
                visible_bbox = bbox_from_mask(mask)
                gt_index = len(gt)
                gt.append(
                    {
                        "cam_R_m2c": pose_m[:3, :3].reshape(-1).tolist(),
                        "cam_t_m2c": (pose_m[:3, 3] * 1000.0).tolist(),
                        "obj_id": object_id,
                    }
                )
                gt_info.append(
                    {
                        "bbox_obj": csv_bbox_xywh(row),
                        "bbox_visib": visible_bbox,
                        "px_count_all": rendered_pixels,
                        "px_count_valid": int(mask.sum()),
                        "px_count_visib": int(mask.sum()),
                        "visib_fract": (
                            float(mask.sum() / rendered_pixels)
                            if rendered_pixels else 0.0
                        ),
                    }
                )
                visible_masks[str(gt_index)] = pycoco_utils.binary_mask_to_rle(
                    mask.astype(np.uint8)
                )
                kept_instances.append(
                    {
                        "opp_id": int(row["opp_id"]),
                        "instance_id": int(row["instance_id"]),
                        "csv_bbox": csv_bbox_xywh(row),
                        "rendered_visible_bbox": visible_bbox,
                        "observed_pixels": observed_pixels,
                        "rendered_pixels": rendered_pixels,
                        "supervision_pixels": int(mask.sum()),
                        "observed_mask_coverage": overlap_fraction,
                    }
                )
            if not gt:
                continue

            depth_mm = np.rint(depth_m * 1000.0)
            depth_mm = np.clip(depth_mm, 0, np.iinfo(np.uint16).max).astype(np.uint16)
            camera = {"cam_K": K.reshape(-1).tolist(), "depth_scale": 1.0}
            writer.write(
                {
                    "__key__": frame.key,
                    "rgb.jpg": image_path.read_bytes(),
                    "depth.png": png_bytes(depth_mm),
                    "camera.json": json.dumps(camera).encode(),
                    "gt.json": json.dumps(gt).encode(),
                    "gt_info.json": json.dumps(gt_info).encode(),
                    "mask_visib.json": json.dumps(visible_masks).encode(),
                }
            )
            samples_written += 1
            instances_written += len(gt)
            manifest.append(
                {
                    "key": frame.key,
                    "session_index": frame.session_index,
                    "source_root": str(frame.source_root),
                    "camera_id": frame.camera_id,
                    "source_frame": frame.frame_id,
                    "image_path": str(image_path),
                    "instances": kept_instances,
                }
            )
    finally:
        writer.close()
    mapping = key_to_shard(split_dir)
    (split_dir / "key_to_shard.json").write_text(json.dumps(mapping, indent=2))
    return samples_written, instances_written, manifest


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.min_mask_overlap <= 1.0:
        raise ValueError("--min-mask-overlap must be between zero and one.")
    sessions = collect_frames(args)
    train_frames, val_frames, split_policy = split_frames(
        sessions,
        args.validation_fraction,
        args.validation_sessions,
        args.gap_frames,
    )
    dataset_dir = args.output_root / args.dataset_name
    train_dir = dataset_dir / "train_pbr_web"
    val_dir = dataset_dir / "val_pbr_web"
    existing = [path for path in (train_dir, val_dir) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output split exists: {existing}; pass --overwrite.")
    if args.overwrite:
        for path in existing:
            shutil.rmtree(path)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    mesh = load_centered_mesh(args.cad_path, args.cad_scale)
    write_models(mesh, dataset_dir, args.object_id)
    cad_to_ac_body = parse_matrix(args.cad_to_ac_body)
    renderer = InstanceRenderer(mesh)
    try:
        train_count, train_instances, train_manifest = write_split(
            train_frames,
            train_dir,
            renderer,
            cad_to_ac_body,
            args.object_id,
            args.max_shard_size,
            args.min_mask_overlap,
        )
        val_count, val_instances, val_manifest = write_split(
            val_frames,
            val_dir,
            renderer,
            cad_to_ac_body,
            args.object_id,
            args.max_shard_size,
            args.min_mask_overlap,
        )
    finally:
        renderer.close()

    # Keep the standard MegaPose/GigaPose root index for train_pbr_web too.
    shutil.copyfile(train_dir / "key_to_shard.json", dataset_dir / "key_to_shard.json")
    metadata = {
        "dataset_name": args.dataset_name,
        "cad_source": str(args.cad_path.resolve()),
        "cad_centered": True,
        "cad_scale": args.cad_scale,
        "cad_to_ac_body": cad_to_ac_body.tolist(),
        "pose_translation_units": "millimeters",
        "depth_storage_units": "millimeters",
        "training_depth_scale": 1.0,
        "mask_supervision": "observed_instance_mask_intersected_with_rendered_cad",
        "minimum_observed_mask_overlap": args.min_mask_overlap,
        "split_policy": split_policy,
        "train": {
            "samples": train_count,
            "instances": train_instances,
            "manifest": train_manifest,
        },
        "validation": {
            "samples": val_count,
            "instances": val_instances,
            "manifest": val_manifest,
        },
    }
    (dataset_dir / "fine_tuning_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    print(f"Training: {train_count} images / {train_instances} instances")
    print(f"Validation: {val_count} images / {val_instances} instances")
    print(f"Dataset written to {dataset_dir}")


if __name__ == "__main__":
    main()
