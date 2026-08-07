"""Isolated recovery-data generator with direct-mask and GT-render fallback."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from tracking.generate_recovery_dataset import iter_gt_frames
from tracking.geometry import pose_from_rt, rotation_error_deg, translation_error_m
from tracking.io import decode_uncompressed_rle
from tracking.rendering import CADRenderer
from tracking.rgb_self_recovery import generate_dataset as base
from . import assetto_export
from tracking.rgb_self_recovery.render_inputs import (
    crop_spec_from_bbox,
    hide_rendered_occluders,
    recovery_targets,
    render_candidate_channels,
    warp_mask,
    warp_rgb,
)
from tracking.scoring import bbox_from_mask


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path)
    p.add_argument("--split", default="train_pbr_web_gsam_clean")
    p.add_argument("--mesh", type=Path)
    p.add_argument("--mesh-scale", type=float, default=1)
    p.add_argument("--center-mesh", action="store_true")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--candidates-per-instance", type=int, default=12)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--crop-scale", type=float, default=2.5)
    p.add_argument("--minimum-crop-side-px", type=float, default=48)
    p.add_argument("--groups-per-shard", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260720)
    p.add_argument("--occlusion-probability", type=float, default=.3)
    p.add_argument("--moderate-center-sigma-px", type=float, default=18)
    p.add_argument("--large-center-sigma-px", type=float, default=70)
    p.add_argument("--moderate-log-depth-sigma", type=float, default=.1)
    p.add_argument("--large-log-depth-sigma", type=float, default=.4)
    p.add_argument("--moderate-rotation-sigma-deg", type=float, default=12)
    p.add_argument("--large-rotation-sigma-deg", type=float, default=65)
    p.add_argument("--correctable-center-crop-px", type=float, default=56)
    p.add_argument("--correctable-log-depth", type=float, default=.6)
    p.add_argument("--correctable-rotation-deg", type=float, default=70)
    p.add_argument("--positive-translation-m", type=float, default=.15)
    p.add_argument("--positive-rotation-deg", type=float, default=10)
    p.add_argument("--mask-dir", type=Path)
    p.add_argument(
        "--mask-pattern",
        default="{key}_i{instance_index:02d}.png",
        help="Binary per-instance path, or a frame-level integer mask path.",
    )
    p.add_argument("--mask-fallback", choices=("dataset", "render"), default="dataset")
    p.add_argument("--assetto-export-root", type=Path)
    p.add_argument(
        "--assetto-split", choices=("train", "validation", "all"), default="train"
    )
    p.add_argument("--assetto-validation-run", action="append", default=[])
    p.add_argument("--assetto-validation-run-count", type=int, default=3)
    p.add_argument("--assetto-cameras", default="front,rear")
    p.add_argument("--assetto-max-depth-m", type=float, default=150.0)
    p.add_argument("--assetto-min-mask-pixels", type=int, default=64)
    p.add_argument("--assetto-exclude-truncated", action="store_true")
    p.add_argument(
        "--assetto-frames-per-bin", type=int,
        help="Balanced unique-image target for every distance bin.",
    )
    p.add_argument(
        "--assetto-distance-bins",
        default="all",
        help="Comma-separated distance-bin directory names, or 'all'.",
    )
    p.add_argument(
        "--assetto-short-bin-policy", choices=("error", "all"), default="error",
        help="Fail on a short bin, or keep every available image without duplication.",
    )
    p.add_argument(
        "--assetto-crop-shaded-sides",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optionally remove equal invalid strips from front/rear image sides.",
    )
    p.add_argument("--assetto-shaded-side-pixels", type=int, default=258)
    p.add_argument(
        "--cad-axis-convention", choices=("x-forward-z-up", "csp"),
        default="x-forward-z-up",
    )
    p.add_argument(
        "--fit-aabb", choices=("none", "uniform", "nonuniform"),
        default="nonuniform",
    )
    p.add_argument("--overwrite", action="store_true")
    return p


def _external_mask(args, key, instance_index, shape):
    if args.mask_dir is None:
        return None
    path = args.mask_dir / args.mask_pattern.format(
        key=key, instance_index=instance_index
    )
    if not path.is_file():
        return None
    value = np.asarray(Image.open(path))
    if value.ndim == 3:
        if value.shape[2] >= 3:
            value = (
                value[..., 0].astype(np.int64)
                + (value[..., 1].astype(np.int64) << 8)
                + (value[..., 2].astype(np.int64) << 16)
            )
        else:
            value = value[..., 0]
    if value.shape != shape:
        import cv2
        value = cv2.resize(value, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    unique = np.unique(value)
    if len(unique) <= 2:
        return value > 0
    return value == instance_index + 1


def _resolve_mask(args, key, instance_index, extra, gt_pose, K, renderer, shape):
    value = _external_mask(args, key, instance_index, shape)
    if value is not None and value.any():
        return value, "mask_directory"
    value = extra.get("mask_arrays", {}).get(str(instance_index))
    if value is not None and np.asarray(value, dtype=bool).any():
        return np.asarray(value, dtype=bool), "assetto_packed_instance_mask"
    rle = extra.get("masks", {}).get(str(instance_index))
    if rle is not None:
        value = decode_uncompressed_rle(rle)
        if value.any():
            return value, "dataset_rle"
    if args.mask_fallback == "render":
        value, _ = renderer.render(gt_pose, K, shape)
        if value.any():
            return np.asarray(value, dtype=bool), "gt_cad_render"
    return None, "missing"


def main():
    args = parser().parse_args()
    if (args.dataset_dir is None) == (args.assetto_export_root is None):
        raise ValueError("Pass exactly one of --dataset-dir or --assetto-export-root")
    if args.assetto_export_root is not None and not args.assetto_export_root.is_dir():
        raise FileNotFoundError(args.assetto_export_root)
    if args.mask_dir is not None and not args.mask_dir.is_dir():
        raise FileNotFoundError(args.mask_dir)
    if args.candidates_per_instance < 3:
        raise ValueError("Use at least three candidates per instance")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    assetto_index = alignment = None
    assetto_selection_report = None
    selected_assetto_bins = None
    mesh_path = None
    if args.assetto_export_root is not None:
        assetto_index = assetto_export.load_index(args.assetto_export_root)
        assetto_index, selected_assetto_bins = assetto_export.select_bins(
            assetto_index, args.assetto_distance_bins
        )
        assetto_index, assetto_selection_report = (
            assetto_export.balanced_consecutive_subset(
                assetto_index,
                args.assetto_frames_per_bin,
                args.assetto_short_bin_policy,
            )
        )
        alignment = assetto_export.make_alignment(args, assetto_index)
        mesh_path = args.output_dir / "assetto_aligned_model.ply"
        alignment.mesh.export(mesh_path)
        frames_iter = assetto_export.iter_frames(args, assetto_index, alignment)
    else:
        mesh_path = base.resolve_mesh(args)
        frames_iter = iter_gt_frames(args.dataset_dir, args.split, load_depth=False)
    renderer = CADRenderer(
        mesh_path,
        mesh_scale=1.0 if alignment is not None else args.mesh_scale,
        center_mesh=False if alignment is not None else args.center_mesh,
    )
    writer = base.RecoveryShardWriter(args.output_dir, args.groups_per_shard)
    frames = groups = candidates = 0
    mask_counts = {}
    started = time.perf_counter()
    try:
        for frame_index, (key, image, K, _depth, gt, extra) in enumerate(frames_iter):
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
            frames += 1
            for instance_index, pose_data in enumerate(gt):
                gt_pose = pose_from_rt(
                    np.asarray(pose_data["cam_R_m2c"], dtype=float).reshape(3, 3),
                    np.asarray(pose_data["cam_t_m2c"], dtype=float) * .001,
                )
                full_mask, mask_source = _resolve_mask(
                    args, key, instance_index, extra, gt_pose, K, renderer, image.shape[:2]
                )
                mask_counts[mask_source] = mask_counts.get(mask_source, 0) + 1
                if full_mask is None or int(full_mask.sum()) < 16:
                    continue
                observed = (
                    base.occlude_mask(full_mask, rng)
                    if rng.random() < args.occlusion_probability else full_mask
                )
                if int(observed.sum()) < 8:
                    continue
                crop = crop_spec_from_bbox(
                    bbox_from_mask(observed), output_size=args.crop_size,
                    crop_scale=args.crop_scale,
                    minimum_side_px=args.minimum_crop_side_px,
                )
                K_crop = crop.transform_intrinsics(K)
                rgb_crop = warp_rgb(image, crop)
                observed_crop = warp_mask(observed, crop)
                occluder_crop = warp_mask(
                    np.asarray(full_mask, bool) & ~np.asarray(observed, bool), crop
                )
                rgb_crop = base.synthesize_rgb_occluder(rgb_crop, occluder_crop, rng)
                values = {
                    "rendered": [], "center_targets": [], "log_depth_targets": [],
                    "rotation_targets": [], "correctable_targets": [],
                    "confidence_targets": [], "quality_targets": [],
                    "candidate_poses": [],
                }
                sources = []
                for candidate_index in range(args.candidates_per_instance):
                    candidate, source = base.random_candidate(
                        gt_pose, K, candidate_index, rng, args
                    )
                    if candidate[2, 3] <= 1e-5:
                        continue
                    encoded, _, _ = render_candidate_channels(
                        renderer, candidate, K_crop, args.crop_size
                    )
                    encoded = hide_rendered_occluders(encoded, occluder_crop)
                    duv, dlogz, drot = recovery_targets(candidate, gt_pose, K_crop)
                    te = translation_error_m(candidate, gt_pose)
                    re = rotation_error_deg(candidate, gt_pose)
                    quality = math.log1p(
                        te / max(args.positive_translation_m, 1e-6)
                        + re / max(args.positive_rotation_deg, 1e-6)
                    )
                    values["rendered"].append(encoded)
                    values["center_targets"].append(duv)
                    values["log_depth_targets"].append(dlogz)
                    values["rotation_targets"].append(drot)
                    values["correctable_targets"].append(
                        np.linalg.norm(duv) <= args.correctable_center_crop_px
                        and abs(dlogz) <= args.correctable_log_depth
                        and math.degrees(np.linalg.norm(drot)) <= args.correctable_rotation_deg
                    )
                    values["confidence_targets"].append(
                        te <= args.positive_translation_m and re <= args.positive_rotation_deg
                    )
                    values["quality_targets"].append(quality)
                    values["candidate_poses"].append(candidate.astype(np.float32))
                    sources.append(source)
                if len(values["rendered"]) < 3:
                    continue
                group_key = f"{key}_i{instance_index:02d}_g{groups:08d}"
                data = base.encode_npz(
                    observed_mask=observed_crop.astype(np.uint8),
                    rendered=np.stack(values["rendered"]).astype(np.uint8),
                    center_targets=np.asarray(values["center_targets"], np.float32),
                    log_depth_targets=np.asarray(values["log_depth_targets"], np.float32),
                    rotation_targets=np.asarray(values["rotation_targets"], np.float32),
                    correctable_targets=np.asarray(values["correctable_targets"], np.bool_),
                    confidence_targets=np.asarray(values["confidence_targets"], np.float32),
                    quality_targets=np.asarray(values["quality_targets"], np.float32),
                    candidate_poses=np.stack(values["candidate_poses"]).astype(np.float32),
                    gt_pose=gt_pose.astype(np.float32), K=np.asarray(K, np.float32),
                    K_crop=K_crop.astype(np.float32),
                    crop=np.asarray([crop.x0, crop.y0, crop.side, crop.output_size], np.float32),
                )
                writer.write(group_key, base.encode_jpeg(rgb_crop), data, {
                    "source_key": key, "instance_index": instance_index,
                    "obj_id": int(pose_data.get("obj_id", 1)),
                    "mask_source": mask_source, "sources": sources,
                })
                groups += 1
                candidates += len(values["rendered"])
            if frames % 25 == 0:
                print(f"generated {groups} groups from {frames} frames", flush=True)
    finally:
        writer.close()
        renderer.close()
    if groups == 0:
        raise RuntimeError("No RGB recovery groups were generated")
    manifest = {
        "format": "rgb_render_self_recovery_v1",
        "generator_extension": "sliding_window_mask_sources_v1",
        "dataset_dir": None if args.dataset_dir is None else str(args.dataset_dir),
        "assetto_export_root": None if args.assetto_export_root is None else str(args.assetto_export_root),
        "split": args.assetto_split if alignment is not None else args.split,
        "mesh": str(mesh_path), "uses_observed_depth": False,
        "frames": frames, "groups": groups, "candidates": candidates,
        "mask_source_counts": mask_counts, "crop_size": args.crop_size,
        "crop_scale": args.crop_scale, "shards": writer.shard_counts,
        "elapsed_s": time.perf_counter() - started,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    if alignment is not None:
        _, held_out = assetto_export.selected_runs(
            args.assetto_export_root, args.assetto_split,
            args.assetto_validation_run, args.assetto_validation_run_count,
        )
        manifest["assetto_held_out_runs"] = held_out
        manifest["assetto_camera_conversion"] = "X-right/Y-up/Z-forward to OpenCV"
        manifest["assetto_mask_identity"] = "bboxes_3d.instance_id"
        manifest["assetto_balanced_selection"] = assetto_selection_report
        manifest["assetto_distance_bins"] = selected_assetto_bins
        manifest["assetto_shaded_side_crop"] = {
            "enabled": bool(args.assetto_crop_shaded_sides),
            "pixels_per_side": (
                int(args.assetto_shaded_side_pixels)
                if args.assetto_crop_shaded_sides else 0
            ),
        }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
