"""Debug prepared fine-tuning samples and GigaPose keypoint correspondences.

This script is intentionally lightweight and visual: it writes RGB/mask/depth
panels plus a JSON report. If GigaPose training reports zero valid patch pairs,
run this on the same dataset/split to see whether masks/depth/poses/templates
are producing usable correspondences.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from fine_tuning.dataloader import SplitWebSceneDataset
from fine_tuning.train import REPO_ROOT, make_dataset_config
from src.custom_megapose.web_scene_dataset import IterableWebSceneDataset
from src.dataloader.keypoints import Keypoint, KeypointInput
from src.megapose.datasets.scene_dataset import SceneObservation
from src.lib3d.torch import inverse_affine
from src.utils.inout import MAX_VALUES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="assettocorsa_front_only")
    parser.add_argument("--root-dir", type=Path, default=Path("gigaPose_datasets/datasets"))
    parser.add_argument("--split", default="train_pbr_web")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--max-visuals", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fine_tuning/debug_training_geometry"),
    )
    parser.add_argument("--seed", type=int, default=2023)
    return parser.parse_args()


def depth_to_image(depth: np.ndarray) -> Image.Image:
    valid = depth > 0
    if not valid.any():
        return Image.fromarray(np.zeros((*depth.shape, 3), dtype=np.uint8))
    clipped = np.zeros_like(depth, dtype=np.float32)
    lo, hi = np.percentile(depth[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    clipped[valid] = np.clip((depth[valid] - lo) / (hi - lo), 0, 1)
    gray = np.uint8(clipped * 255)
    return Image.fromarray(np.stack([gray, gray, gray], axis=-1))


def union_masks(observation) -> np.ndarray:
    mask = np.zeros(observation.rgb.shape[:2], dtype=bool)
    for object_data in observation.object_datas:
        mask |= observation.binary_masks[object_data.unique_id].astype(bool)
    return mask


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(0, 255, 0)) -> Image.Image:
    out = rgb.copy()
    color_arr = np.array(color, dtype=np.uint8)
    out[mask] = np.uint8(0.45 * out[mask] + 0.55 * color_arr)
    return Image.fromarray(out)


def draw_objects(image: Image.Image, observation) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for object_data in observation.object_datas:
        x, y, w, h = [int(v) for v in object_data.bbox_modal]
        draw.rectangle((x, y, x + w - 1, y + h - 1), outline=(255, 255, 0), width=3)
        pose = object_data.TWO.matrix
        draw.text((x, max(0, y - 14)), f"id={object_data.unique_id} z={pose[2,3]:.3f}", fill=(255, 255, 0))
    return out


def save_panel(observation, output_path: Path) -> dict[str, object]:
    rgb = observation.rgb
    depth = observation.depth
    mask = union_masks(observation)
    depth_valid = depth > 0
    mask_no_depth = mask & ~depth_valid

    rgb_img = draw_objects(Image.fromarray(rgb), observation)
    mask_img = overlay_mask(rgb, mask, color=(0, 255, 0))
    depth_img = depth_to_image(depth)
    bad_img = overlay_mask(rgb, mask_no_depth, color=(255, 0, 0))

    width, height = rgb_img.size
    panel = Image.new("RGB", (width * 4, height))
    panel.paste(rgb_img, (0, 0))
    panel.paste(mask_img, (width, 0))
    panel.paste(depth_img, (width * 2, 0))
    panel.paste(bad_img, (width * 3, 0))
    panel.save(output_path)

    object_stats = []
    for object_data in observation.object_datas:
        obj_mask = observation.binary_masks[object_data.unique_id].astype(bool)
        object_stats.append(
            {
                "unique_id": object_data.unique_id,
                "label": object_data.label,
                "mask_pixels": int(obj_mask.sum()),
                "mask_depth_pixels": int((obj_mask & depth_valid).sum()),
                "pose_z": float(object_data.TWO.matrix[2, 3]),
                "bbox_modal": [int(v) for v in object_data.bbox_modal],
            }
        )

    return {
        "scene_id": observation.infos.scene_id,
        "view_id": observation.infos.view_id,
        "rgb_shape": list(rgb.shape),
        "depth_positive_pixels": int(depth_valid.sum()),
        "mask_pixels": int(mask.sum()),
        "mask_without_depth_pixels": int(mask_no_depth.sum()),
        "objects": object_stats,
        "panel": str(output_path),
    }


def keypoint_stage_stats(dataset, real_data) -> dict[str, object]:
    """Mirror KeyPointSampler.sample_pts with counters at each filter stage."""
    template_data, T_real2temp, T_temp2real = dataset.process_template(real_data)
    all_data = {}
    for name, data in zip(["real", "template"], [real_data, template_data]):
        all_data[name] = KeypointInput(
            full_rgb=data.full_rgb,
            full_depth=data.full_depth,
            K=data.K,
            M=data.M,
            mask=data.mask,
            rgb=data.rgb,
        )

    sampler = dataset.keypoint_sampler
    batch_size = T_temp2real.shape[0]
    init_points = sampler.grid_points.clone().unsqueeze(0).repeat(batch_size, 1, 1)

    key_pts2d = Keypoint(src=init_points.clone(), tar=init_points.clone())
    key_pts2d.mask("src", all_data["template"].mask)
    key_pts2d.mask("tar", all_data["real"].mask)
    key_pts2d_cropped = key_pts2d.clone()
    initial_src = key_pts2d_cropped.src[:, :, 0] != -1
    initial_tar = key_pts2d_cropped.tar[:, :, 0] != -1

    key_pts2d.apply_affine("src", inverse_affine(all_data["template"].M))
    key_pts2d.apply_affine("tar", inverse_affine(all_data["real"].M))
    key_pts3d_src = key_pts2d.unproject(
        "src", K=all_data["template"].K, depth=all_data["template"].full_depth
    )
    key_pts3d_tar = key_pts2d.unproject(
        "tar", K=all_data["real"].K, depth=all_data["real"].full_depth
    )
    src_depth_valid = key_pts3d_src[:, :, 2] > 0
    tar_depth_valid = key_pts3d_tar[:, :, 2] > 0

    key_pts3d = Keypoint(src=key_pts3d_src, tar=key_pts3d_tar)
    key_pts3d.apply_3D_transform("src", T_temp2real)
    key_pts3d.apply_3D_transform("tar", T_real2temp)

    reproj_src = key_pts3d.project("src", K=all_data["real"].K)
    reproj_tar = key_pts3d.project("tar", K=all_data["template"].K)
    reproj_key_pts2d = Keypoint(src=reproj_src, tar=reproj_tar)
    reproj_key_pts2d.apply_affine("src", all_data["real"].M)
    reproj_key_pts2d.apply_affine("tar", all_data["template"].M)
    reproj_key_pts2d.mask("src", all_data["real"].mask)
    reproj_key_pts2d.mask("tar", all_data["template"].mask)
    reproj_src_valid = reproj_key_pts2d.src[:, :, 0] != -1
    reproj_tar_valid = reproj_key_pts2d.tar[:, :, 0] != -1
    usable_src_columns = torch.logical_and(initial_src, reproj_src_valid)
    usable_tar_rows = torch.logical_and(initial_tar, reproj_tar_valid)

    final_counts = []
    min_distances = []
    for idx in range(batch_size):
        mask_tar_all = torch.logical_or(
            key_pts2d_cropped.tar[idx, :, 0] == -1,
            reproj_key_pts2d.tar[idx, :, 0] == -1,
        )
        mask_src_all = torch.logical_or(
            key_pts2d_cropped.src[idx, :, 0] == -1,
            reproj_key_pts2d.src[idx, :, 0] == -1,
        )
        distance = torch.cdist(
            reproj_key_pts2d.tar[idx].float(), key_pts2d.src[idx].float()
        )
        distance[mask_tar_all] = MAX_VALUES
        distance[:, mask_src_all] = MAX_VALUES
        per_tar_distance, _ = torch.min(distance, dim=1)
        finite = per_tar_distance < MAX_VALUES
        final_counts.append(int((per_tar_distance < 1000.0).sum().item()))
        min_distances.append(float(per_tar_distance[finite].min().item()) if finite.any() else None)

    return {
        "template_mask_pixels_after_crop": [int(v.item()) for v in template_data.mask.flatten(1).sum(dim=1)],
        "real_mask_pixels_after_crop": [int(v.item()) for v in real_data.mask.flatten(1).sum(dim=1)],
        "initial_template_grid_points_in_mask": [int(v.item()) for v in initial_src.sum(dim=1)],
        "initial_real_grid_points_in_mask": [int(v.item()) for v in initial_tar.sum(dim=1)],
        "template_depth_valid_grid_points": [int(v.item()) for v in src_depth_valid.sum(dim=1)],
        "real_depth_valid_grid_points": [int(v.item()) for v in tar_depth_valid.sum(dim=1)],
        "template_to_real_reprojected_points_in_real_mask": [int(v.item()) for v in reproj_src_valid.sum(dim=1)],
        "real_to_template_reprojected_points_in_template_mask": [int(v.item()) for v in reproj_tar_valid.sum(dim=1)],
        "usable_source_grid_points_initial_and_reprojected": [int(v.item()) for v in usable_src_columns.sum(dim=1)],
        "usable_target_grid_points_initial_and_reprojected": [int(v.item()) for v in usable_tar_rows.sum(dim=1)],
        "final_valid_patch_pairs_per_instance": final_counts,
        "min_final_patch_distance_per_instance": min_distances,
    }


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = args.root_dir / args.dataset_name
    split_dir = dataset_dir / args.split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    report: dict[str, object] = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "sample_visuals": [],
        "batches": [],
    }

    raw_dataset = SplitWebSceneDataset(split_dir, depth_scale=1.0)
    raw_iter = iter(IterableWebSceneDataset(raw_dataset).datapipeline)
    for idx in range(args.max_visuals):
        try:
            observation = next(raw_iter)
        except StopIteration:
            break
        panel_path = args.output_dir / f"{idx:06d}_{observation.infos.scene_id}_{observation.infos.view_id}.jpg"
        report["sample_visuals"].append(save_panel(observation, panel_path))

    with initialize_config_dir(
        version_base=None, config_dir=str((REPO_ROOT / "configs").resolve())
    ):
        cfg = compose(config_name="train")
    OmegaConf.set_struct(cfg, False)
    dataset_cfg = make_dataset_config(cfg, args, args.split, augment=False)
    dataset = instantiate(dataset_cfg)
    data_iter = iter(dataset.web_dataloader.datapipeline)
    for batch_idx in range(args.max_batches):
        observations = []
        for _ in range(args.batch_size):
            try:
                observations.append(next(data_iter))
            except StopIteration:
                break
        if not observations:
            break
        batch = dataset.collate_fn(observations)
        if batch is None:
            report["batches"].append({"batch_index": batch_idx, "collate": "none"})
            continue
        scene_batch = SceneObservation.collate_fn(observations)
        real_data = dataset.process_real(scene_batch)
        stage_stats = keypoint_stage_stats(dataset, real_data)
        src_valid = torch.logical_and(batch.src_pts[:, :, 0] != -1, batch.src_pts[:, :, 1] != -1)
        tar_valid = torch.logical_and(batch.tar_pts[:, :, 0] != -1, batch.tar_pts[:, :, 1] != -1)
        pair_valid = torch.logical_and(src_valid, tar_valid)
        report["batches"].append(
            {
                "batch_index": batch_idx,
                "batch_size_after_collate": int(batch.src_pts.shape[0]),
                "src_valid_patches": int(src_valid.sum().item()),
                "tar_valid_patches": int(tar_valid.sum().item()),
                "valid_patch_pairs": int(pair_valid.sum().item()),
                "keypoint_stage_stats": stage_stats,
                "instances": [
                    {
                        "scene_id": str(row.scene_id),
                        "view_id": str(row.view_id),
                        "label": str(row.label),
                    }
                    for row in batch.infos.itertuples()
                ],
            }
        )

    report_path = args.output_dir / "debug_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["batches"], indent=2))
    print(f"Wrote {report_path}")
    print(f"Wrote visual panels to {args.output_dir}")


if __name__ == "__main__":
    main()
