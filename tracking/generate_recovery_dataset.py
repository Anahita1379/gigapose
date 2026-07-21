"""Generate hard pose failures and CAD evidence for recovery-head training."""

from __future__ import annotations

import argparse
import io
import json
import math
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

from tracking.config import TrackerConfig
from tracking.geometry import (
    flipped_pose,
    pose_from_rt,
    rotate_pose,
    rotation_error_deg,
    shift_projected_center,
    so3_log,
    translation_error_m,
)
from tracking.io import decode_uncompressed_rle
from tracking.recovery import FEATURE_NAMES, breakdown_to_features
from tracking.rendering import CADRenderer
from tracking.scoring import CandidateScorer, bbox_from_mask
from tracking.types import Detection, FrameData, PoseHypothesis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", default="train_pbr_web_gsam_clean")
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--center-mesh", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--perturbations-per-instance", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--no-depth",
        action="store_true",
        help=(
            "Ignore depth.png even when the prepared split contains it. "
            "Use this when the recovery head will be deployed with "
            "tracking.run_tracking --no-depth."
        ),
    )
    parser.add_argument("--occlusion-probability", type=float, default=0.30)
    parser.add_argument("--positive-translation-m", type=float, default=0.15)
    parser.add_argument("--positive-rotation-deg", type=float, default=10.0)
    return parser.parse_args()


def _resolve_mesh(args: argparse.Namespace) -> Path:
    if args.mesh is not None:
        return args.mesh
    candidate = args.dataset_dir / "models" / "obj_000001.ply"
    if candidate.is_file():
        return candidate
    fallback = Path("gigaPose_datasets/datasets/racecar/models/obj_000001.ply")
    if fallback.is_file():
        return fallback
    raise FileNotFoundError("Pass --mesh.")


def _shard_path(split_dir: Path, value: Any) -> Path:
    if isinstance(value, str) and value.endswith(".tar"):
        return split_dir / value
    return split_dir / f"shard-{int(value):06d}.tar"


def _extract(tar: tarfile.TarFile, key: str, suffix: str) -> bytes | None:
    try:
        handle = tar.extractfile(f"{key}.{suffix}")
    except KeyError:
        return None
    return handle.read() if handle is not None else None


def iter_gt_frames(
    dataset_dir: Path,
    split: str,
    *,
    load_depth: bool = True,
) -> Iterator[tuple[str, np.ndarray, np.ndarray, np.ndarray | None, list[dict], dict]]:
    split_dir = dataset_dir / split
    mapping = json.loads((split_dir / "key_to_shard.json").read_text())
    shard_to_keys: dict[str, list[str]] = defaultdict(list)
    for key, value in mapping.items():
        shard_to_keys[str(value)].append(key)
    for value, keys in sorted(shard_to_keys.items()):
        shard = _shard_path(split_dir, value)
        with tarfile.open(shard) as tar:
            for key in sorted(keys):
                rgb_bytes = _extract(tar, key, "rgb.jpg") or _extract(tar, key, "rgb.png")
                camera_bytes = _extract(tar, key, "camera.json")
                gt_bytes = _extract(tar, key, "gt.json")
                info_bytes = _extract(tar, key, "gt_info.json")
                masks_bytes = _extract(tar, key, "mask_visib.json")
                if None in (rgb_bytes, camera_bytes, gt_bytes, info_bytes, masks_bytes):
                    continue
                image = np.asarray(Image.open(io.BytesIO(rgb_bytes)).convert("RGB"))
                camera = json.loads(camera_bytes)
                K = np.asarray(camera["cam_K"], dtype=float).reshape(3, 3)
                depth_m = None
                depth_bytes = _extract(tar, key, "depth.png") if load_depth else None
                if load_depth and depth_bytes is not None:
                    raw = np.asarray(Image.open(io.BytesIO(depth_bytes)), dtype=float)
                    depth_m = raw * float(camera.get("depth_scale", 1.0)) * 0.001
                yield (
                    key,
                    image,
                    K,
                    depth_m,
                    json.loads(gt_bytes),
                    {
                        "infos": json.loads(info_bytes),
                        "masks": json.loads(masks_bytes),
                    },
                )


def _occlude(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    output = mask.copy()
    ys, xs = np.where(output)
    if xs.size < 4:
        return output
    width = int(rng.uniform(0.15, 0.45) * (xs.max() - xs.min() + 1))
    height = int(rng.uniform(0.15, 0.45) * (ys.max() - ys.min() + 1))
    x0 = int(rng.integers(xs.min(), max(xs.max() - width + 2, xs.min() + 1)))
    y0 = int(rng.integers(ys.min(), max(ys.max() - height + 2, ys.min() + 1)))
    output[y0 : y0 + height, x0 : x0 + width] = False
    return output


def perturb_pose(
    pose: np.ndarray,
    K: np.ndarray,
    index: int,
    rng: np.random.Generator,
) -> np.ndarray:
    # Ensure every object receives clean, moderate, large, and flipped examples.
    if index == 0:
        return pose.copy()
    if index == 1:
        return flipped_pose(pose, "z")
    severity = "moderate" if index % 3 else "large"
    center_sigma = 12.0 if severity == "moderate" else 55.0
    log_depth_sigma = 0.08 if severity == "moderate" else 0.30
    rotation_sigma = 6.0 if severity == "moderate" else 28.0
    candidate = shift_projected_center(
        pose,
        K,
        delta_uv_px=rng.normal(0.0, center_sigma, size=2),
        delta_log_depth=float(rng.normal(0.0, log_depth_sigma)),
    )
    axis = rng.normal(size=3)
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    candidate = rotate_pose(
        candidate,
        axis * math.radians(float(rng.normal(0.0, rotation_sigma))),
        side="left",
    )
    return candidate


def main() -> None:
    args = parse_args()
    if args.perturbations_per_instance < 2:
        raise ValueError("Use at least two perturbations so clean and flipped cases exist.")
    rng = np.random.default_rng(args.seed)
    config = TrackerConfig.load(args.config)
    renderer = CADRenderer(
        _resolve_mesh(args),
        mesh_scale=args.mesh_scale,
        center_mesh=args.center_mesh,
    )
    features, rotation_targets, translation_targets = [], [], []
    confidence_targets, quality_targets, group_ids = [], [], []
    frame_count = instance_count = depth_frame_count = 0
    try:
        scorer = CandidateScorer(renderer, config)
        for frame_index, (key, image, K, depth_m, gt, extra) in enumerate(
            iter_gt_frames(
                args.dataset_dir,
                args.split,
                load_depth=not args.no_depth,
            )
        ):
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
            if depth_m is not None:
                depth_frame_count += 1
            scene_id, im_id = (int(value) for value in key.split("_"))
            for instance_index, pose_data in enumerate(gt):
                mask_rle = extra["masks"].get(str(instance_index))
                if mask_rle is None:
                    continue
                mask = decode_uncompressed_rle(mask_rle)
                if not mask.any():
                    continue
                gt_pose = pose_from_rt(
                    np.asarray(pose_data["cam_R_m2c"], dtype=float).reshape(3, 3),
                    np.asarray(pose_data["cam_t_m2c"], dtype=float) * 0.001,
                )
                group_id = instance_count
                for perturbation_index in range(args.perturbations_per_instance):
                    observed_mask = (
                        _occlude(mask, rng)
                        if rng.random() < args.occlusion_probability
                        else mask
                    )
                    detection = Detection(
                        detection_id=instance_index,
                        bbox_xywh=bbox_from_mask(observed_mask),
                        mask=observed_mask,
                        obj_id=int(pose_data.get("obj_id", 1)),
                    )
                    frame = FrameData(
                        scene_id,
                        im_id,
                        image,
                        K,
                        [detection],
                        depth_m=depth_m,
                    )
                    candidate_pose = perturb_pose(
                        gt_pose, K, perturbation_index, rng
                    )
                    hypothesis = PoseHypothesis(
                        candidate_pose,
                        source="synthetic_failure",
                        measurement_score=float(rng.uniform(0.15, 0.85)),
                        obj_id=detection.obj_id,
                    )
                    evaluated = scorer.evaluate(hypothesis, frame, detection)
                    rotation_delta = so3_log(
                        gt_pose[:3, :3] @ candidate_pose[:3, :3].T
                    )
                    translation_delta = gt_pose[:3, 3] - candidate_pose[:3, 3]
                    translation_error = translation_error_m(gt_pose, candidate_pose)
                    rotation_error = rotation_error_deg(gt_pose, candidate_pose)
                    quality = (
                        translation_error / max(args.positive_translation_m, 1e-6)
                        + rotation_error / max(args.positive_rotation_deg, 1e-6)
                    )
                    confidence = float(
                        translation_error <= args.positive_translation_m
                        and rotation_error <= args.positive_rotation_deg
                    )
                    features.append(
                        breakdown_to_features(
                            evaluated.score, hypothesis.measurement_score
                        )
                    )
                    rotation_targets.append(rotation_delta.astype(np.float32))
                    translation_targets.append(translation_delta.astype(np.float32))
                    confidence_targets.append(confidence)
                    quality_targets.append(quality)
                    group_ids.append(group_id)
                instance_count += 1
            frame_count += 1
            if frame_count % 25 == 0:
                print(
                    f"generated {len(features)} candidates from "
                    f"{frame_count} frames/{instance_count} instances"
                )
    finally:
        renderer.close()
    if not features:
        raise RuntimeError("No recovery samples were generated.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=np.asarray(features, dtype=np.float32),
        rotation_targets=np.asarray(rotation_targets, dtype=np.float32),
        translation_targets=np.asarray(translation_targets, dtype=np.float32),
        confidence_targets=np.asarray(confidence_targets, dtype=np.float32),
        quality_targets=np.asarray(quality_targets, dtype=np.float32),
        group_ids=np.asarray(group_ids, dtype=np.int64),
        feature_names=np.asarray(FEATURE_NAMES),
        depth_enabled=np.asarray(not args.no_depth, dtype=np.bool_),
    )
    report = {
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "frames": frame_count,
        "instances": instance_count,
        "candidates": len(features),
        "features": list(FEATURE_NAMES),
        "depth_enabled": not args.no_depth,
        "frames_with_depth": depth_frame_count,
        "seed": args.seed,
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
