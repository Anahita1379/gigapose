"""Validate rendered training shards before starting a long fine-tuning run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fine_tuning.dataloader import SplitWebSceneDataset
from src.custom_megapose.web_scene_dataset import IterableWebSceneDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("gigaPose_datasets/datasets/assettocorsa"),
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--max-missing-depth-pixels",
        type=int,
        default=16,
        help=(
            "Allow this many mask pixels per object to have zero rendered depth. "
            "This tolerates tiny renderer edge artifacts while still catching bad shards."
        ),
    )
    parser.add_argument(
        "--max-missing-depth-fraction",
        type=float,
        default=1e-3,
        help="Allow this fraction of each object mask to have zero rendered depth.",
    )
    return parser.parse_args()


def validate_split(
    split_dir: Path,
    max_samples: int | None,
    max_missing_depth_pixels: int,
    max_missing_depth_fraction: float,
) -> dict[str, object]:
    dataset = SplitWebSceneDataset(split_dir, depth_scale=1.0)
    observations = IterableWebSceneDataset(dataset).datapipeline
    sample_count = instance_count = foreground_depth_pixels = 0
    missing_depth_instances = 0
    missing_depth_pixels = 0
    worst_missing_depth = {
        "sample_index": None,
        "scene_id": None,
        "view_id": None,
        "object_unique_id": None,
        "missing_pixels": 0,
        "mask_pixels": 0,
        "missing_fraction": 0.0,
    }
    determinants = []
    for observation in observations:
        if max_samples is not None and sample_count >= max_samples:
            break
        if observation.depth is None or not np.any(observation.depth > 0):
            raise ValueError(f"{split_dir}: sample {sample_count} has no foreground depth")
        if not observation.object_datas:
            raise ValueError(f"{split_dir}: sample {sample_count} has no GT objects")
        for object_data in observation.object_datas:
            mask = observation.binary_masks[object_data.unique_id]
            if not mask.any():
                raise ValueError(f"{split_dir}: object {object_data.unique_id} has empty mask")
            missing_depth_mask = np.logical_and(mask, observation.depth <= 0)
            num_missing_depth = int(missing_depth_mask.sum())
            num_mask_pixels = int(mask.sum())
            missing_depth_fraction = (
                float(num_missing_depth / num_mask_pixels) if num_mask_pixels else 0.0
            )
            if num_missing_depth:
                missing_depth_instances += 1
                missing_depth_pixels += num_missing_depth
                if num_missing_depth > worst_missing_depth["missing_pixels"]:
                    worst_missing_depth = {
                        "sample_index": sample_count,
                        "scene_id": observation.infos.scene_id,
                        "view_id": observation.infos.view_id,
                        "object_unique_id": object_data.unique_id,
                        "missing_pixels": num_missing_depth,
                        "mask_pixels": num_mask_pixels,
                        "missing_fraction": missing_depth_fraction,
                    }
            if (
                num_missing_depth > max_missing_depth_pixels
                and missing_depth_fraction > max_missing_depth_fraction
            ):
                raise ValueError(
                    f"{split_dir}: sample {sample_count} "
                    f"({observation.infos.scene_id}_{observation.infos.view_id}), "
                    f"object {object_data.unique_id} has {num_missing_depth}/"
                    f"{num_mask_pixels} mask pixels without rendered depth "
                    f"({missing_depth_fraction:.6f}). Re-run "
                    "fine_tuning.prepare_ac_training_data with the updated mask-depth "
                    "intersection, or raise the tolerance only if this is a tiny edge artifact."
                )
            determinants.append(float(np.linalg.det(object_data.TWO.matrix[:3, :3])))
        sample_count += 1
        instance_count += len(observation.object_datas)
        foreground_depth_pixels += int((observation.depth > 0).sum())
    if not sample_count:
        raise RuntimeError(f"No samples found in {split_dir}")
    if not np.allclose(determinants, 1.0, atol=1e-4):
        raise ValueError(f"Improper rotations in {split_dir}: {determinants}")
    return {
        "samples_checked": sample_count,
        "instances_checked": instance_count,
        "foreground_depth_pixels": foreground_depth_pixels,
        "missing_depth_instances": missing_depth_instances,
        "missing_depth_pixels": missing_depth_pixels,
        "worst_missing_depth": worst_missing_depth,
        "rotation_determinant_min": min(determinants),
        "rotation_determinant_max": max(determinants),
    }


def main() -> None:
    args = parse_args()
    report = {
        split: validate_split(
            args.dataset_dir / split,
            args.max_samples,
            args.max_missing_depth_pixels,
            args.max_missing_depth_fraction,
        )
        for split in ("train_pbr_web", "val_pbr_web")
    }
    output_path = args.dataset_dir / "validation_report.json"
    output_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
