"""Merge selector, GigaPose, and regenerated-GT bundles by physical frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from tracking.geometry import rotation_error_deg, translation_error_m

from .dataset import MeasurementBundle
from .export_measurements import _ground_truth_by_source


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--measurements", type=Path, required=True)
    value.add_argument("--baseline-measurements", type=Path, required=True)
    value.add_argument("--ground-truth-dataset-dir", type=Path, required=True)
    value.add_argument("--split", default="test")
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--overwrite", action="store_true")
    return value


def _keys(arrays) -> list[tuple[str, str, int]]:
    return [
        (str(run), str(camera), int(frame))
        for run, camera, frame in zip(
            arrays["source_run"], arrays["camera_id"], arrays["source_frame"]
        )
    ]


def main() -> None:
    args = parser().parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    measurement = MeasurementBundle(args.measurements)
    baseline = MeasurementBundle(args.baseline_measurements)
    baseline_lookup = {
        key: index for index, key in enumerate(_keys(baseline.arrays))
    }
    ground_truth = _ground_truth_by_source(args.ground_truth_dataset_dir, args.split)
    measurement_keys = _keys(measurement.arrays)
    missing_baseline = [key for key in measurement_keys if key not in baseline_lookup]
    missing_gt = [key for key in measurement_keys if key not in ground_truth]
    if missing_baseline or missing_gt:
        raise ValueError(
            f"Missing baseline={len(missing_baseline)}, regenerated_gt={len(missing_gt)}"
        )

    arrays = {name: value.copy() for name, value in measurement.arrays.items()}
    baseline_indices = np.asarray(
        [baseline_lookup[key] for key in measurement_keys], dtype=np.int64
    )
    arrays["baseline_pose"] = baseline.arrays["measurement_pose"][baseline_indices]
    arrays["baseline_score"] = baseline.arrays["measurement_score"][baseline_indices]
    arrays["ground_truth_pose"] = np.asarray(
        [ground_truth[key] for key in measurement_keys], dtype=np.float32
    )
    arrays["raw_translation_error_m"] = np.asarray([
        translation_error_m(pose, gt)
        for pose, gt in zip(arrays["measurement_pose"], arrays["ground_truth_pose"])
    ])
    arrays["raw_rotation_error_deg"] = np.asarray([
        rotation_error_deg(pose, gt)
        for pose, gt in zip(arrays["measurement_pose"], arrays["ground_truth_pose"])
    ])
    np.savez_compressed(args.output_dir / "measurements.npz", **arrays)

    manifest = dict(measurement.manifest)
    manifest.update({
        "format": "rgb_self_recovery_gru_kalman_measurements_v2",
        "fallback_baseline_measurements": str(args.baseline_measurements),
        "ground_truth_dataset_dir": str(args.ground_truth_dataset_dir),
        "frame_count": len(measurement_keys),
        "merged_by": ["source_run", "camera_id", "source_frame"],
    })
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({
        "format": manifest["format"],
        "frame_count": len(measurement_keys),
        "missing_baseline": len(missing_baseline),
        "missing_regenerated_ground_truth": len(missing_gt),
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
