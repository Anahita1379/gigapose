#!/usr/bin/env python3
"""Build GigaPose test shards and detections from 20260623+ AC sessions.

By default, detections use the pose-rendered masks produced by
``generate_masks.py``. This is useful for pipeline debugging and an oracle
upper bound, but not for measuring a real detector+pose pipeline.
"""

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
    validate_model_row,
    write_models,
)


DEFAULT_CAD = Path("gigaPose_datasets/datasets/racecar/models/obj_000001.ply")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare new AC sessions for GigaPose inference.")
    parser.add_argument("--source-root", type=Path, action="append", required=True)
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
    parser.add_argument("--max-shard-size", type=int, default=1000)
    parser.add_argument(
        "--detection-file-name", default="cnos-fastsam_assettocorsa-test.json"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sessions = collect_sessions(
        args.source_root,
        parse_cameras(args.cameras),
        parse_ids(args.opponent_ids),
        args.exclude_truncated,
        args.frame_stride,
        args.max_frames_per_session,
    )
    reference = sessions[0][0].rows[0]
    alignment = build_aligned_mesh(
        args.cad_path, reference, args.cad_scale,
        args.cad_axis_convention, args.fit_aabb,
    )

    dataset_dir = args.output_root / args.dataset_name
    test_dir = dataset_dir / "test"
    detection_dir = args.output_root / "cnos-fastsam"
    detection_path = detection_dir / args.detection_file_name
    existing = [path for path in (test_dir, detection_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Inference output exists: {existing}; pass --overwrite")
    if args.overwrite:
        if test_dir.exists():
            shutil.rmtree(test_dir)
        detection_path.unlink(missing_ok=True)
        for name in ("test_targets_bop19.json", "frame_map.json", "inference_metadata.json"):
            (dataset_dir / name).unlink(missing_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    detection_dir.mkdir(parents=True, exist_ok=True)
    write_models(alignment, dataset_dir, args.object_id)

    detections: list[dict[str, object]] = []
    targets: list[dict[str, int]] = []
    frame_map: list[dict[str, object]] = []
    writer = wds.ShardWriter(
        pattern=str(test_dir / "shard-%06d.tar"),
        maxcount=args.max_shard_size,
        encoder=False,
    )
    try:
        for session in sessions:
            for frame in session:
                if not frame.image_path.is_file():
                    raise FileNotFoundError(f"Missing image: {frame.image_path}")
                size = image_size(frame)
                with Image.open(frame.image_path) as image:
                    if image.size != size:
                        raise ValueError(f"Image size mismatch: {frame.image_path}")
                instance_ids = decode_instance_ids(frame.mask_path(args.mask_dir_name), size)
                gt, gt_info, mask_rles, instances = [], [], {}, []
                for row in frame.rows:
                    validate_model_row(alignment, row)
                    opponent_id = int(row["opp_id"])
                    mask = instance_ids == opponent_id
                    if not mask.any():
                        continue
                    pose = cad_to_camera_pose(row, alignment.local_center)
                    bbox = bbox_from_mask(mask)
                    gt_index = len(gt)
                    gt.append({
                        "cam_R_m2c": pose[:3, :3].reshape(-1).tolist(),
                        "cam_t_m2c": (pose[:3, 3] * 1000.0).tolist(),
                        "obj_id": args.object_id,
                    })
                    gt_info.append({
                        "bbox_obj": csv_bbox_xywh(row),
                        "bbox_visib": bbox,
                        "px_count_all": int(mask.sum()),
                        "px_count_valid": int(mask.sum()),
                        "px_count_visib": int(mask.sum()),
                        "visib_fract": 1.0,
                    })
                    rle = pycoco_utils.binary_mask_to_rle(mask.astype(np.uint8))
                    mask_rles[str(gt_index)] = rle
                    detections.append({
                        "scene_id": frame.scene_id,
                        "image_id": frame.frame_id,
                        "category_id": args.object_id,
                        "score": 1.0,
                        "bbox": bbox,
                        "segmentation": rle,
                        "time": 0.0,
                        "opp_id": opponent_id,
                        "truncated": int(row["truncated"]),
                        "mask_provenance": "ground_truth_pose_rendered_cad",
                    })
                    instances.append({
                        "opp_id": opponent_id,
                        "bbox": bbox,
                        "csv_bbox": csv_bbox_xywh(row),
                    })
                if not gt:
                    continue
                K = intrinsics(frame)
                camera = {"cam_K": K.reshape(-1).tolist(), "depth_scale": 1.0}
                writer.write({
                    "__key__": frame.key,
                    "rgb.jpg": frame.image_path.read_bytes(),
                    "depth.png": png_bytes(np.zeros((size[1], size[0]), dtype=np.uint16)),
                    "camera.json": json.dumps(camera).encode(),
                    "gt.json": json.dumps(gt).encode(),
                    "gt_info.json": json.dumps(gt_info).encode(),
                    "mask_visib.json": json.dumps(mask_rles).encode(),
                })
                targets.append({
                    "scene_id": frame.scene_id,
                    "im_id": frame.frame_id,
                    "obj_id": args.object_id,
                    "inst_count": len(gt),
                })
                frame_map.append({
                    "scene_id": frame.scene_id,
                    "im_id": frame.frame_id,
                    "session_index": frame.session_index,
                    "source_root": str(frame.source_root),
                    "camera_id": frame.camera_id,
                    "sim_time_ms": frame.sim_time_ms,
                    "image_path": str(frame.image_path),
                    "mask_path": str(frame.mask_path(args.mask_dir_name)),
                    "instances": instances,
                })
    finally:
        writer.close()

    (test_dir / "key_to_shard.json").write_text(
        json.dumps(key_to_shard_map(test_dir), indent=2)
    )
    (dataset_dir / "test_targets_bop19.json").write_text(json.dumps(targets, indent=2))
    (dataset_dir / "frame_map.json").write_text(json.dumps(frame_map, indent=2))
    detection_path.write_text(json.dumps(detections, indent=2))
    (dataset_dir / "inference_metadata.json").write_text(json.dumps({
        "dataset_name": args.dataset_name,
        "source_roots": [str(path) for path in args.source_root],
        "mask_provenance": "ground_truth_pose_rendered_cad",
        "oracle_inference_warning": (
            "These detections are pose-derived and must not be reported as an unbiased "
            "GigaPose inference benchmark. Replace the detection JSON with masks from an "
            "independent segmenter for a real inference evaluation."
        ),
        "cad_axis_convention": args.cad_axis_convention,
        "fit_aabb": args.fit_aabb,
    }, indent=2))
    print(f"Prepared {len(targets)} camera frames and {len(detections)} detections.")
    print(f"Dataset: {dataset_dir}")
    print(f"Detections: {detection_path}")


if __name__ == "__main__":
    main()
