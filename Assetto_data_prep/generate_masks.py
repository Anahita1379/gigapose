#!/usr/bin/env python3
"""Render pose-derived, lossless instance-ID masks for 20260623+ recordings.

All visible opponents in a camera frame are rendered together, so nearer cars
occlude farther cars. Pixels contain the opponent ID encoded little-endian as
RGB: ``id = R + 256*G + 65536*B``; zero is background.

These masks use ground-truth poses. They are suitable as training supervision
and geometry diagnostics, but using them as GigaPose inference detections makes
the inference experiment an oracle/upper-bound experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from Assetto_data_prep.common import (
    build_aligned_mesh,
    cad_to_camera_pose,
    collect_sessions,
    csv_bbox_xywh,
    encode_instance_ids,
    image_size,
    intrinsics,
    parse_cameras,
    parse_ids,
    validate_model_row,
)
from fine_tuning.ac_geometry import InstanceRenderer


DEFAULT_CAD = Path("gigaPose_datasets/datasets/racecar/models/obj_000001.ply")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CAD-silhouette instance masks for new AC recordings."
    )
    parser.add_argument("--source-root", type=Path, action="append", required=True)
    parser.add_argument("--cad-path", type=Path, default=DEFAULT_CAD)
    parser.add_argument("--cameras", default="all")
    parser.add_argument("--opponent-ids", default=None)
    parser.add_argument("--mask-dir-name", default="generated_masks")
    parser.add_argument(
        "--visible-mask-dir-name",
        default="generated_masks_visib",
        help=(
            "Directory for one binary visible-mask PNG per opponent. Set to an "
            "empty string to disable these additional files."
        ),
    )
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
    parser.add_argument(
        "--no-clip-to-csv-bbox", dest="clip_to_csv_bbox", action="store_false",
        help="Do not clip each rendered silhouette to its saved projected AABB.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(clip_to_csv_bbox=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cameras = parse_cameras(args.cameras)
    sessions = collect_sessions(
        args.source_root,
        cameras,
        parse_ids(args.opponent_ids),
        args.exclude_truncated,
        args.frame_stride,
        args.max_frames_per_session,
    )
    total_written = total_skipped = empty_instances = 0
    for frames in sessions:
        reference_row = frames[0].rows[0]
        alignment = build_aligned_mesh(
            args.cad_path,
            reference_row,
            args.cad_scale,
            args.cad_axis_convention,
            args.fit_aabb,
        )
        renderer = InstanceRenderer(alignment.mesh)
        session_written = session_skipped = 0
        try:
            for index, frame in enumerate(frames, start=1):
                output = frame.mask_path(args.mask_dir_name)
                visible_outputs = [
                    frame.source_root
                    / args.visible_mask_dir_name
                    / frame.camera_id
                    / f"{frame.stem}_opp{int(row['opp_id']):06d}.png"
                    for row in frame.rows
                ] if args.visible_mask_dir_name else []
                outputs_complete = output.exists() and all(
                    path.exists() for path in visible_outputs
                )
                if outputs_complete and not args.overwrite:
                    session_skipped += 1
                    continue
                if not frame.image_path.is_file():
                    raise FileNotFoundError(f"Missing synchronized image: {frame.image_path}")
                expected_size = image_size(frame)
                with Image.open(frame.image_path) as image:
                    if image.size != expected_size:
                        raise ValueError(
                            f"Image size mismatch for {frame.image_path}: "
                            f"{image.size} != {expected_size}"
                        )
                for row in frame.rows:
                    validate_model_row(alignment, row)
                    opponent_id = int(row["opp_id"])
                    if not 0 < opponent_id < 2**24:
                        raise ValueError(f"Opponent ID cannot be RGB-encoded: {opponent_id}")
                poses = [
                    cad_to_camera_pose(row, alignment.local_center) for row in frame.rows
                ]
                segmentation, _ = renderer.render(
                    poses, intrinsics(frame), expected_size[0], expected_size[1]
                )
                instance_ids = np.zeros(segmentation.shape, dtype=np.uint32)
                visible_masks: list[tuple[int, np.ndarray]] = []
                for render_id, row in enumerate(frame.rows, start=1):
                    mask = segmentation == render_id
                    if args.clip_to_csv_bbox:
                        x, y, width, height = csv_bbox_xywh(row)
                        clipped = np.zeros_like(mask)
                        clipped[y:y + height, x:x + width] = mask[y:y + height, x:x + width]
                        mask = clipped
                    opponent_id = int(row["opp_id"])
                    visible_masks.append((opponent_id, mask))
                    if not mask.any():
                        empty_instances += 1
                        continue
                    instance_ids[mask] = opponent_id
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_name(output.stem + ".tmp.png")
                Image.fromarray(encode_instance_ids(instance_ids)).save(temporary)
                temporary.replace(output)
                if args.visible_mask_dir_name:
                    visible_dir = visible_outputs[0].parent
                    visible_dir.mkdir(parents=True, exist_ok=True)
                    for opponent_id, mask in visible_masks:
                        visible_path = visible_dir / (
                            f"{frame.stem}_opp{opponent_id:06d}.png"
                        )
                        visible_temporary = visible_path.with_name(
                            visible_path.stem + ".tmp.png"
                        )
                        Image.fromarray(mask.astype(np.uint8) * 255).save(
                            visible_temporary
                        )
                        visible_temporary.replace(visible_path)
                session_written += 1
                if index % 250 == 0 or index == len(frames):
                    print(
                        f"{frame.source_root.name}: processed {index}/{len(frames)} "
                        f"camera frames"
                    )
        finally:
            renderer.close()

        metadata = {
            "schema": "assetto_pose_rendered_instance_ids_v1",
            "source_root": str(frames[0].source_root),
            "cad_path": str(args.cad_path.resolve()),
            "cad_scale": args.cad_scale,
            "cad_axis_convention": args.cad_axis_convention,
            "fit_aabb": args.fit_aabb,
            "clip_to_csv_bbox": args.clip_to_csv_bbox,
            "visible_mask_dir_name": args.visible_mask_dir_name or None,
            "encoding": "instance_id = R + 256*G + 65536*B; background = 0",
            "visible_mask_encoding": (
                "one binary uint8 PNG per opponent: background=0, visible=255"
                if args.visible_mask_dir_name else None
            ),
            "ground_truth_pose_derived": True,
            "limitations": [
                "Does not model occlusion by track, ego car, or static scene geometry.",
                "Must not be treated as an independently predicted inference mask.",
            ],
            "written_this_run": session_written,
            "skipped_existing_this_run": session_skipped,
        }
        metadata_path = frames[0].source_root / args.mask_dir_name / "metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2))
        total_written += session_written
        total_skipped += session_skipped

    print(f"Wrote {total_written} masks; skipped {total_skipped} existing masks.")
    if empty_instances:
        print(f"WARNING: {empty_instances} visible CSV instances rendered zero pixels.")


if __name__ == "__main__":
    main()
