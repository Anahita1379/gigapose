"""Generate grouped RGB/render recovery training shards without sensor depth."""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import shutil
import tarfile
import time

import cv2
import numpy as np
from PIL import Image

from tracking.generate_recovery_dataset import iter_gt_frames
from tracking.geometry import (
    flipped_pose,
    pose_from_rt,
    project_center,
    rotate_pose,
    rotation_error_deg,
    shift_projected_center,
    translation_error_m,
)
from tracking.io import decode_uncompressed_rle
from tracking.rendering import CADRenderer
from tracking.rgb_self_recovery.render_inputs import (
    crop_spec_from_bbox,
    hide_rendered_occluders,
    recovery_targets,
    render_candidate_channels,
    warp_mask,
    warp_rgb,
)
from tracking.scoring import bbox_from_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", default="train_pbr_web_gsam_clean")
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--center-mesh", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--candidates-per-instance", type=int, default=12)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--crop-scale", type=float, default=2.5)
    parser.add_argument("--minimum-crop-side-px", type=float, default=48.0)
    parser.add_argument("--groups-per-shard", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--occlusion-probability", type=float, default=0.30)
    parser.add_argument("--moderate-center-sigma-px", type=float, default=18.0)
    parser.add_argument("--large-center-sigma-px", type=float, default=70.0)
    parser.add_argument("--moderate-log-depth-sigma", type=float, default=0.10)
    parser.add_argument("--large-log-depth-sigma", type=float, default=0.40)
    parser.add_argument("--moderate-rotation-sigma-deg", type=float, default=12.0)
    parser.add_argument("--large-rotation-sigma-deg", type=float, default=65.0)
    parser.add_argument("--correctable-center-crop-px", type=float, default=56.0)
    parser.add_argument("--correctable-log-depth", type=float, default=0.60)
    parser.add_argument("--correctable-rotation-deg", type=float, default=70.0)
    parser.add_argument("--positive-translation-m", type=float, default=0.15)
    parser.add_argument("--positive-rotation-deg", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_mesh(args: argparse.Namespace) -> Path:
    if args.mesh is not None:
        return args.mesh
    candidate = args.dataset_dir / "models" / "obj_000001.ply"
    if candidate.is_file():
        return candidate
    fallback = Path("gigaPose_datasets/datasets/racecar/models/obj_000001.ply")
    if fallback.is_file():
        return fallback
    raise FileNotFoundError("Pass --mesh; no default CAD was found.")


def occlude_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    output = np.asarray(mask, dtype=bool).copy()
    ys, xs = np.where(output)
    if xs.size < 16:
        return output
    box_width = xs.max() - xs.min() + 1
    box_height = ys.max() - ys.min() + 1
    width = max(1, int(rng.uniform(0.12, 0.45) * box_width))
    height = max(1, int(rng.uniform(0.12, 0.45) * box_height))
    x0 = int(rng.integers(xs.min(), max(xs.max() - width + 2, xs.min() + 1)))
    y0 = int(rng.integers(ys.min(), max(ys.max() - height + 2, ys.min() + 1)))
    output[y0 : y0 + height, x0 : x0 + width] = False
    return output


def random_candidate(
    gt_pose: np.ndarray,
    K: np.ndarray,
    candidate_index: int,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, str]:
    if candidate_index == 0:
        return gt_pose.copy(), "ground_truth"
    if candidate_index == 1:
        return flipped_pose(gt_pose, "z"), "flip180"
    large = candidate_index % 3 == 0
    center_sigma = (
        args.large_center_sigma_px if large else args.moderate_center_sigma_px
    )
    log_depth_sigma = (
        args.large_log_depth_sigma if large else args.moderate_log_depth_sigma
    )
    rotation_sigma = math.radians(
        args.large_rotation_sigma_deg
        if large
        else args.moderate_rotation_sigma_deg
    )
    candidate = shift_projected_center(
        gt_pose,
        K,
        delta_uv_px=rng.normal(0.0, center_sigma, size=2),
        delta_log_depth=float(rng.normal(0.0, log_depth_sigma)),
    )
    axis = rng.normal(size=3)
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    candidate = rotate_pose(
        candidate,
        axis * float(rng.normal(0.0, rotation_sigma)),
        side="left",
    )
    return candidate, "large_random" if large else "moderate_random"


def encode_jpeg(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(
        buffer, format="JPEG", quality=92, subsampling=0
    )
    return buffer.getvalue()


def encode_npz(**values) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **values)
    return buffer.getvalue()


def synthesize_rgb_occluder(
    rgb_crop: np.ndarray,
    occluder_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Replace artificially hidden target pixels with varied RGB content."""

    output = np.asarray(rgb_crop, dtype=np.uint8).copy()
    occluded = np.asarray(occluder_mask, dtype=bool)
    count = int(occluded.sum())
    if count == 0:
        return output
    base = rng.integers(0, 256, size=3, dtype=np.uint8).astype(np.int16)
    noise = rng.normal(0.0, 28.0, size=(count, 3))
    output[occluded] = np.clip(base + noise, 0, 255).astype(np.uint8)
    return output


class RecoveryShardWriter:
    def __init__(self, output_dir: Path, groups_per_shard: int):
        self.output_dir = output_dir
        self.groups_per_shard = int(groups_per_shard)
        self.shard_index = -1
        self.groups_in_shard = 0
        self.tar: tarfile.TarFile | None = None
        self.shard_counts: dict[str, int] = {}

    def _open_next(self) -> None:
        if self.tar is not None:
            self.tar.close()
        self.shard_index += 1
        name = f"shard-{self.shard_index:06d}.tar"
        self.tar = tarfile.open(self.output_dir / name, "w")
        self.groups_in_shard = 0
        self.shard_counts[name] = 0

    def _add(self, name: str, value: bytes) -> None:
        assert self.tar is not None
        info = tarfile.TarInfo(name)
        info.size = len(value)
        self.tar.addfile(info, io.BytesIO(value))

    def write(self, key: str, rgb_jpeg: bytes, data_npz: bytes, metadata: dict) -> None:
        if self.tar is None or self.groups_in_shard >= self.groups_per_shard:
            self._open_next()
        self._add(f"{key}.rgb.jpg", rgb_jpeg)
        self._add(f"{key}.data.npz", data_npz)
        self._add(f"{key}.json", json.dumps(metadata).encode())
        self.groups_in_shard += 1
        name = f"shard-{self.shard_index:06d}.tar"
        self.shard_counts[name] += 1

    def close(self) -> None:
        if self.tar is not None:
            self.tar.close()
            self.tar = None


def main() -> None:
    args = parse_args()
    if args.candidates_per_instance < 3:
        raise ValueError("Use at least three candidates per instance.")
    if args.crop_size < 32:
        raise ValueError("--crop-size must be at least 32.")
    if args.groups_per_shard <= 0:
        raise ValueError("--groups-per-shard must be positive.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} is not empty; pass --overwrite."
            )
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    renderer = CADRenderer(
        resolve_mesh(args),
        mesh_scale=args.mesh_scale,
        center_mesh=args.center_mesh,
    )
    writer = RecoveryShardWriter(args.output_dir, args.groups_per_shard)
    frame_count = instance_count = group_count = candidate_count = 0
    started = time.perf_counter()
    try:
        for frame_index, (key, image, K, _depth, gt, extra) in enumerate(
            iter_gt_frames(args.dataset_dir, args.split, load_depth=False)
        ):
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
            frame_count += 1
            for instance_index, pose_data in enumerate(gt):
                mask_rle = extra["masks"].get(str(instance_index))
                if mask_rle is None:
                    continue
                full_mask = decode_uncompressed_rle(mask_rle)
                if int(full_mask.sum()) < 16:
                    continue
                observed_mask = (
                    occlude_mask(full_mask, rng)
                    if rng.random() < args.occlusion_probability
                    else full_mask
                )
                if int(observed_mask.sum()) < 8:
                    continue
                bbox = bbox_from_mask(observed_mask)
                crop = crop_spec_from_bbox(
                    bbox,
                    output_size=args.crop_size,
                    crop_scale=args.crop_scale,
                    minimum_side_px=args.minimum_crop_side_px,
                )
                K_crop = crop.transform_intrinsics(K)
                rgb_crop = warp_rgb(image, crop)
                observed_crop = warp_mask(observed_mask, crop)
                known_occluder_crop = warp_mask(
                    np.asarray(full_mask, dtype=bool)
                    & ~np.asarray(observed_mask, dtype=bool),
                    crop,
                )
                rgb_crop = synthesize_rgb_occluder(
                    rgb_crop, known_occluder_crop, rng
                )
                gt_pose = pose_from_rt(
                    np.asarray(pose_data["cam_R_m2c"], dtype=float).reshape(3, 3),
                    np.asarray(pose_data["cam_t_m2c"], dtype=float) * 0.001,
                )

                render_inputs = []
                center_targets = []
                log_depth_targets = []
                rotation_targets = []
                correctable_targets = []
                confidence_targets = []
                quality_targets = []
                candidate_poses = []
                sources = []
                for candidate_index in range(args.candidates_per_instance):
                    candidate, source = random_candidate(
                        gt_pose, K, candidate_index, rng, args
                    )
                    if candidate[2, 3] <= 1e-5:
                        continue
                    encoded, _, _ = render_candidate_channels(
                        renderer, candidate, K_crop, args.crop_size
                    )
                    encoded = hide_rendered_occluders(
                        encoded, known_occluder_crop
                    )
                    delta_uv, delta_log_depth, delta_rotation = recovery_targets(
                        candidate, gt_pose, K_crop
                    )
                    translation_error = translation_error_m(candidate, gt_pose)
                    rotation_error = rotation_error_deg(candidate, gt_pose)
                    quality = math.log1p(
                        translation_error
                        / max(args.positive_translation_m, 1e-6)
                        + rotation_error
                        / max(args.positive_rotation_deg, 1e-6)
                    )
                    correctable = (
                        float(np.linalg.norm(delta_uv))
                        <= args.correctable_center_crop_px
                        and abs(delta_log_depth) <= args.correctable_log_depth
                        and math.degrees(float(np.linalg.norm(delta_rotation)))
                        <= args.correctable_rotation_deg
                    )
                    confidence = (
                        translation_error <= args.positive_translation_m
                        and rotation_error <= args.positive_rotation_deg
                    )
                    render_inputs.append(encoded)
                    center_targets.append(delta_uv)
                    log_depth_targets.append(delta_log_depth)
                    rotation_targets.append(delta_rotation)
                    correctable_targets.append(correctable)
                    confidence_targets.append(confidence)
                    quality_targets.append(quality)
                    candidate_poses.append(candidate.astype(np.float32))
                    sources.append(source)
                if len(render_inputs) < 3:
                    continue

                group_key = (
                    f"{key}_i{instance_index:02d}_g{group_count:08d}"
                )
                data = encode_npz(
                    observed_mask=observed_crop.astype(np.uint8),
                    rendered=np.stack(render_inputs).astype(np.uint8),
                    center_targets=np.asarray(center_targets, dtype=np.float32),
                    log_depth_targets=np.asarray(
                        log_depth_targets, dtype=np.float32
                    ),
                    rotation_targets=np.asarray(rotation_targets, dtype=np.float32),
                    correctable_targets=np.asarray(
                        correctable_targets, dtype=np.bool_
                    ),
                    confidence_targets=np.asarray(
                        confidence_targets, dtype=np.float32
                    ),
                    quality_targets=np.asarray(quality_targets, dtype=np.float32),
                    candidate_poses=np.stack(candidate_poses).astype(np.float32),
                    gt_pose=gt_pose.astype(np.float32),
                    K=np.asarray(K, dtype=np.float32),
                    K_crop=K_crop.astype(np.float32),
                    crop=np.asarray(
                        [crop.x0, crop.y0, crop.side, crop.output_size],
                        dtype=np.float32,
                    ),
                )
                writer.write(
                    group_key,
                    encode_jpeg(rgb_crop),
                    data,
                    {
                        "source_key": key,
                        "instance_index": instance_index,
                        "obj_id": int(pose_data.get("obj_id", 1)),
                        "sources": sources,
                    },
                )
                instance_count += 1
                group_count += 1
                candidate_count += len(render_inputs)
            if frame_count % 25 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"generated {group_count} groups/{candidate_count} candidates "
                    f"from {frame_count} frames in {elapsed:.1f}s"
                )
    finally:
        writer.close()
        renderer.close()

    if group_count == 0:
        raise RuntimeError("No RGB self-recovery groups were generated.")
    manifest = {
        "format": "rgb_render_self_recovery_v1",
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "mesh": str(resolve_mesh(args)),
        "uses_observed_depth": False,
        "frames": frame_count,
        "instances": instance_count,
        "groups": group_count,
        "candidates": candidate_count,
        "candidates_per_instance_requested": args.candidates_per_instance,
        "crop_size": args.crop_size,
        "crop_scale": args.crop_scale,
        "shards": writer.shard_counts,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
