#!/usr/bin/env python3
"""Build GigaPose train_pbr_web and val_pbr_web from new AC sessions."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import webdataset as wds
from bop_toolkit_lib import pycoco_utils
from PIL import Image

from Assetto_data_prep.common import (
    bbox_from_mask,
    build_aligned_mesh,
    cad_to_camera_pose,
    collect_sessions,
    csv_bbox_xywh,
    decode_instance_ids,
    image_size,
    intrinsics,
    key_to_shard_map,
    parse_cameras,
    parse_ids,
    png_bytes,
    split_sessions,
    validate_model_row,
    write_models,
)
from fine_tuning.ac_geometry import InstanceRenderer


DEFAULT_CAD = Path("gigaPose_datasets/datasets/racecar/models/obj_000001.ply")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare new AC sessions for GigaPose training.")
    parser.add_argument(
        "--source-root", type=Path, action="append", required=True,
        help="Repeat in the desired train/validation order; trailing sessions are held out.",
    )
    parser.add_argument("--cad-path", type=Path, default=DEFAULT_CAD)
    parser.add_argument("--output-root", type=Path, default=Path("gigaPose_datasets/datasets"))
    parser.add_argument("--dataset-name", default="assettocorsa")
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--cameras", default="front")
    parser.add_argument("--opponent-ids", default=None)
    parser.add_argument("--mask-dir-name", default="generated_masks")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-session", type=int, default=None)
    parser.add_argument("--exclude-truncated", action="store_true")
    parser.add_argument("--cad-scale", type=float, default=1.0)
    parser.add_argument(
        "--cad-axis-convention", choices=("x-forward-z-up", "csp"),
        default="x-forward-z-up",
    )
    parser.add_argument(
        "--fit-aabb", choices=("none", "uniform", "nonuniform"),
        default="nonuniform",
    )
    parser.add_argument("--validation-sessions", type=int, default=1)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--gap-frames", type=int, default=50)
    parser.add_argument("--min-mask-pixels", type=int, default=64)
    parser.add_argument("--max-shard-size", type=int, default=250)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_split(
    frames,
    output_dir: Path,
    alignment,
    mask_dir_name: str,
    object_id: int,
    max_shard_size: int,
    min_mask_pixels: int,
) -> tuple[int, int, list[dict[str, object]]]:
    output_dir.mkdir(parents=True)
    renderer = InstanceRenderer(alignment.mesh)
    writer = wds.ShardWriter(
        pattern=str(output_dir / "shard-%06d.tar"),
        maxcount=max_shard_size,
        encoder=False,
    )
    sample_count = instance_count = 0
    manifest: list[dict[str, object]] = []
    try:
        for progress, frame in enumerate(frames, start=1):
            if not frame.image_path.is_file():
                raise FileNotFoundError(f"Missing image: {frame.image_path}")
            size = image_size(frame)
            with Image.open(frame.image_path) as image:
                if image.size != size:
                    raise ValueError(f"Image size mismatch: {frame.image_path}")
            instance_ids = decode_instance_ids(frame.mask_path(mask_dir_name), size)
            for row in frame.rows:
                validate_model_row(alignment, row)
            poses = [
                cad_to_camera_pose(row, alignment.local_center) for row in frame.rows
            ]
            segmentation, depth_m = renderer.render(
                poses, intrinsics(frame), size[0], size[1]
            )

            gt, gt_info, visible_masks, kept = [], [], {}, []
            for render_id, (row, pose) in enumerate(zip(frame.rows, poses), start=1):
                opponent_id = int(row["opp_id"])
                observed_mask = instance_ids == opponent_id
                rendered_mask = segmentation == render_id
                # Keep only pixels that are in the generated/observed instance
                # mask, the rendered CAD silhouette, and the rendered depth map.
                # GigaPose samples 3D correspondences from this mask, so every
                # supervised pixel needs valid CAD depth.
                mask = np.logical_and.reduce(
                    (observed_mask, rendered_mask, depth_m > 0)
                )
                pixels = int(mask.sum())
                if pixels < min_mask_pixels:
                    continue
                rendered_pixels = int((segmentation == render_id).sum())
                bbox = bbox_from_mask(mask)
                gt_index = len(gt)
                gt.append({
                    "cam_R_m2c": pose[:3, :3].reshape(-1).tolist(),
                    "cam_t_m2c": (pose[:3, 3] * 1000.0).tolist(),
                    "obj_id": object_id,
                })
                gt_info.append({
                    "bbox_obj": csv_bbox_xywh(row),
                    "bbox_visib": bbox,
                    "px_count_all": rendered_pixels,
                    "px_count_valid": pixels,
                    "px_count_visib": pixels,
                    "visib_fract": float(pixels / rendered_pixels) if rendered_pixels else 0.0,
                })
                visible_masks[str(gt_index)] = pycoco_utils.binary_mask_to_rle(
                    mask.astype(np.uint8)
                )
                kept.append({
                    "opp_id": opponent_id,
                    "bbox": bbox,
                    "csv_bbox": csv_bbox_xywh(row),
                    "mask_pixels": pixels,
                    "rendered_pixels": rendered_pixels,
                })
            if not gt:
                continue

            depth_mm = np.clip(
                np.rint(depth_m * 1000.0), 0, np.iinfo(np.uint16).max
            ).astype(np.uint16)
            camera = {"cam_K": intrinsics(frame).reshape(-1).tolist(), "depth_scale": 1.0}
            writer.write({
                "__key__": frame.key,
                "rgb.jpg": frame.image_path.read_bytes(),
                "depth.png": png_bytes(depth_mm),
                "camera.json": json.dumps(camera).encode(),
                "gt.json": json.dumps(gt).encode(),
                "gt_info.json": json.dumps(gt_info).encode(),
                "mask_visib.json": json.dumps(visible_masks).encode(),
            })
            sample_count += 1
            instance_count += len(gt)
            manifest.append({
                "key": frame.key,
                "session_index": frame.session_index,
                "source_root": str(frame.source_root),
                "camera_id": frame.camera_id,
                "source_frame": frame.frame_id,
                "instances": kept,
            })
            if progress % 250 == 0 or progress == len(frames):
                print(f"{output_dir.name}: processed {progress}/{len(frames)} frames")
    finally:
        writer.close()
        renderer.close()
    (output_dir / "key_to_shard.json").write_text(
        json.dumps(key_to_shard_map(output_dir), indent=2)
    )
    return sample_count, instance_count, manifest


def main() -> None:
    args = parse_args()
    if args.min_mask_pixels < 1:
        raise ValueError("--min-mask-pixels must be positive")
    sessions = collect_sessions(
        args.source_root,
        parse_cameras(args.cameras),
        parse_ids(args.opponent_ids),
        args.exclude_truncated,
        args.frame_stride,
        args.max_frames_per_session,
    )
    train_frames, val_frames, split_policy = split_sessions(
        sessions, args.validation_sessions, args.validation_fraction, args.gap_frames
    )
    reference = sessions[0][0].rows[0]
    alignment = build_aligned_mesh(
        args.cad_path, reference, args.cad_scale,
        args.cad_axis_convention, args.fit_aabb,
    )

    dataset_dir = args.output_root / args.dataset_name
    train_dir = dataset_dir / "train_pbr_web"
    val_dir = dataset_dir / "val_pbr_web"
    existing = [path for path in (train_dir, val_dir) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Training output exists: {existing}; pass --overwrite")
    if args.overwrite:
        for path in existing:
            shutil.rmtree(path)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_models(alignment, dataset_dir, args.object_id)

    train_count, train_instances, train_manifest = write_split(
        train_frames, train_dir, alignment, args.mask_dir_name,
        args.object_id, args.max_shard_size, args.min_mask_pixels,
    )
    val_count, val_instances, val_manifest = write_split(
        val_frames, val_dir, alignment, args.mask_dir_name,
        args.object_id, args.max_shard_size, args.min_mask_pixels,
    )
    shutil.copyfile(train_dir / "key_to_shard.json", dataset_dir / "key_to_shard.json")
    metadata = {
        "schema": "assetto_20260623_transforms_v1",
        "dataset_name": args.dataset_name,
        "source_roots": [str(path) for path in args.source_root],
        "cad_source": str(args.cad_path.resolve()),
        "cad_axis_convention": args.cad_axis_convention,
        "fit_aabb": args.fit_aabb,
        "centered_model_local_center": alignment.local_center.tolist(),
        "pose_translation_units": "millimeters",
        "depth_storage_units": "millimeters",
        "mask_dir_name": args.mask_dir_name,
        "mask_supervision": (
            "generated_instance_mask_intersected_with_rendered_cad_and_valid_depth"
        ),
        "mask_limitations": (
            "Opponent-opponent occlusion is rendered. Track, ego-car, and static-scene "
            "occlusion is not represented unless it is implicitly removed by CSV clipping."
        ),
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
    print(f"Dataset: {dataset_dir}")


if __name__ == "__main__":
    main()
