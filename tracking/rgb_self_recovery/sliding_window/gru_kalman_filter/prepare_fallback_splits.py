"""Create disjoint fallback-head fit/validation/test measurement bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from tracking.io import WebDatasetSequence

from .dataset import MeasurementBundle


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--data", type=Path, required=True)
    value.add_argument("--fit-dataset-dir", type=Path, required=True)
    value.add_argument("--held-out-dataset-dir", type=Path, required=True)
    value.add_argument("--split", default="test")
    value.add_argument("--validation-fraction", type=float, default=0.2)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--overwrite", action="store_true")
    return value


def _dataset_keys(path: Path, split: str) -> set[tuple[str, str, int]]:
    sequence = WebDatasetSequence(path, split, load_depth=False)
    return {
        (str(row["source_run"]), str(row["camera_id"]), int(row["source_frame"]))
        for row in sequence.rows
    }


def _write_subset(
    root: Path,
    bundle: MeasurementBundle,
    indices: np.ndarray,
    *,
    role: str,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    arrays = {name: values[indices] for name, values in bundle.arrays.items()}
    np.savez_compressed(root / "measurements.npz", **arrays)
    manifest = dict(bundle.manifest)
    manifest.update({
        "format": "rgb_self_recovery_gru_kalman_measurements_v2",
        "fallback_split_role": role,
        "frame_count": int(len(indices)),
        "run_count": int(len(set(arrays["source_run"].tolist()))),
        "sequence_count": int(len(set(zip(
            arrays["source_run"].tolist(), arrays["camera_id"].tolist()
        )))),
        "segment_count": int(len(set(arrays["segment_id"].tolist()))),
    })
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    args = parser().parse_args()
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be between 0 and 0.5")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite")
        shutil.rmtree(args.output_dir)

    bundle = MeasurementBundle(args.data)
    arrays = bundle.arrays
    row_keys = [
        (str(run), str(camera), int(frame))
        for run, camera, frame in zip(
            arrays["source_run"], arrays["camera_id"], arrays["source_frame"]
        )
    ]
    fit_keys = _dataset_keys(args.fit_dataset_dir, args.split)
    held_out_keys = _dataset_keys(args.held_out_dataset_dir, args.split)
    if fit_keys & held_out_keys:
        raise ValueError("Fit and held-out datasets contain overlapping physical frames")

    fit_indices = np.asarray(
        [index for index, key in enumerate(row_keys) if key in fit_keys], dtype=np.int64
    )
    test_indices = np.asarray(
        [index for index, key in enumerate(row_keys) if key in held_out_keys], dtype=np.int64
    )
    if len(fit_indices) != len(fit_keys):
        raise ValueError(
            f"Only matched {len(fit_indices)}/{len(fit_keys)} requested fit frames"
        )
    if len(test_indices) != len(held_out_keys):
        raise ValueError(
            f"Only matched {len(test_indices)}/{len(held_out_keys)} held-out frames"
        )

    # Keep consecutive temporal blocks. Each original sequence segment gives
    # its final fraction to validation; individual frames are never shuffled.
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    for segment_id in np.unique(arrays["segment_id"][fit_indices]):
        segment = fit_indices[arrays["segment_id"][fit_indices] == segment_id]
        if len(segment) < 4:
            train_parts.append(segment)
            continue
        validation_count = max(2, int(round(len(segment) * args.validation_fraction)))
        validation_count = min(validation_count, len(segment) - 2)
        train_parts.append(segment[:-validation_count])
        validation_parts.append(segment[-validation_count:])
    train_indices = np.concatenate(train_parts)
    validation_indices = np.concatenate(validation_parts)
    if set(train_indices.tolist()) & set(validation_indices.tolist()):
        raise AssertionError("Fallback fit and validation indices overlap")

    _write_subset(args.output_dir / "train", bundle, train_indices, role="fit")
    _write_subset(
        args.output_dir / "validation", bundle, validation_indices,
        role="internal_validation",
    )
    _write_subset(args.output_dir / "test", bundle, test_indices, role="held_out_test")
    _write_subset(
        args.output_dir / "fit_full", bundle, fit_indices,
        role="full_designated_validation",
    )
    report = {
        "format": "rgb_self_recovery_fallback_split_v1",
        "fit_dataset_dir": str(args.fit_dataset_dir),
        "held_out_dataset_dir": str(args.held_out_dataset_dir),
        "fit_requested_frames": len(fit_keys),
        "train_frames": len(train_indices),
        "validation_frames": len(validation_indices),
        "held_out_test_frames": len(test_indices),
        "train_runs": sorted(set(arrays["source_run"][train_indices].tolist())),
        "validation_runs": sorted(set(arrays["source_run"][validation_indices].tolist())),
        "held_out_test_runs": sorted(set(arrays["source_run"][test_indices].tolist())),
        "validation_fraction": args.validation_fraction,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "split_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
