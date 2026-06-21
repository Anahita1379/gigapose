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
    return parser.parse_args()


def validate_split(split_dir: Path, max_samples: int | None) -> dict[str, object]:
    dataset = SplitWebSceneDataset(split_dir, depth_scale=1.0)
    observations = IterableWebSceneDataset(dataset).datapipeline
    sample_count = instance_count = foreground_depth_pixels = 0
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
            if not np.all(observation.depth[mask] > 0):
                raise ValueError(
                    f"{split_dir}: object {object_data.unique_id} has mask pixels "
                    "without rendered depth"
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
        "rotation_determinant_min": min(determinants),
        "rotation_determinant_max": max(determinants),
    }


def main() -> None:
    args = parse_args()
    report = {
        split: validate_split(args.dataset_dir / split, args.max_samples)
        for split in ("train_pbr_web", "val_pbr_web")
    }
    output_path = args.dataset_dir / "validation_report.json"
    output_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
