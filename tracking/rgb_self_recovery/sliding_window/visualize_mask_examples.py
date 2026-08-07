"""Export a small, human-readable sample of packed instance masks.

The Assetto Corsa masks encode an integer instance ID in RGB byte order.  This
tool leaves those training masks untouched and writes vivid color masks and
RGB overlays to a separate output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


DEFAULT_BINS = (
    "distance0_20",
    "distance20_40",
    "distance40_60",
    "distance60_80",
    "distance80_100",
    "distance100_120",
)

# Distinct RGB colors. Index zero is deliberately unused (background).
PALETTE = np.asarray(
    [
        [255, 64, 64],
        [64, 220, 255],
        [255, 210, 64],
        [128, 255, 96],
        [220, 96, 255],
        [255, 128, 32],
        [64, 128, 255],
        [255, 96, 180],
    ],
    dtype=np.uint8,
)


def decode_instance_ids(path: Path) -> np.ndarray:
    """Decode an integer ID image; alpha, when present, is not part of the ID."""
    value = np.asarray(Image.open(path))
    if value.ndim == 2:
        return value.astype(np.int64)
    if value.shape[2] < 3:
        return value[..., 0].astype(np.int64)
    return (
        value[..., 0].astype(np.int64)
        + (value[..., 1].astype(np.int64) << 8)
        + (value[..., 2].astype(np.int64) << 16)
    )


def colorize(instance_ids: np.ndarray) -> np.ndarray:
    colored = np.zeros((*instance_ids.shape, 3), dtype=np.uint8)
    for instance_id in np.unique(instance_ids):
        if instance_id == 0:
            continue
        colored[instance_ids == instance_id] = PALETTE[(int(instance_id) - 1) % len(PALETTE)]
    return colored


def overlay(image: np.ndarray, colored: np.ndarray, alpha: float) -> np.ndarray:
    if colored.shape[:2] != image.shape[:2]:
        colored = cv2.resize(
            colored, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    foreground = np.any(colored != 0, axis=2)
    result = image.copy()
    result[foreground] = np.clip(
        (1.0 - alpha) * image[foreground] + alpha * colored[foreground], 0, 255
    ).astype(np.uint8)
    contours, _ = cv2.findContours(
        foreground.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(result, contours, -1, (255, 255, 255), 1)
    return result


def evenly_spaced(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if len(rows) <= count:
        return rows
    indices = np.linspace(0, len(rows) - 1, count, dtype=int)
    return [rows[index] for index in indices]


def resolve(root: Path, bin_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    local = bin_dir / path
    return local if local.exists() else root / path


def image_and_mask_values(row: dict[str, str]) -> tuple[str, str]:
    """Support both training and benchmark distance-bin manifests."""
    if row.get("image") and row.get("masks"):
        return row["image"], row["masks"]
    output_filename = row.get("output_filename", "")
    camera = row.get("camera_id", "")
    if output_filename and camera:
        image = Path("images") / camera / output_filename
        mask = Path("masks") / camera / Path(output_filename).with_suffix(".png")
        return str(image), str(mask)
    raise ValueError(f"Manifest row has no resolvable image/mask pair: {row}")


def export_examples(
    root: Path,
    output: Path,
    bins: tuple[str, ...],
    examples_per_folder: int,
    alpha: float,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for bin_name in bins:
        bin_dir = root / bin_name
        manifest_path = bin_dir / "source_manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        with manifest_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))

        cameras = sorted({row["camera_id"] for row in rows})
        for camera in cameras:
            camera_rows = [row for row in rows if row["camera_id"] == camera]
            for row in evenly_spaced(camera_rows, examples_per_folder):
                image_value, mask_value = image_and_mask_values(row)
                mask_path = resolve(root, bin_dir, mask_value)
                image_path = resolve(root, bin_dir, image_value)
                if not mask_path.is_file() or not image_path.is_file():
                    raise FileNotFoundError(
                        f"Missing image/mask pair: {image_path}, {mask_path}"
                    )

                instance_ids = decode_instance_ids(mask_path)
                colored = colorize(instance_ids)
                rgb = np.asarray(Image.open(image_path).convert("RGB"))
                blended = overlay(rgb, colored, alpha)

                relative_dir = Path(bin_name) / camera
                color_output = output / "colored_masks" / relative_dir / mask_path.name
                overlay_output = output / "overlays" / relative_dir / mask_path.name
                color_output.parent.mkdir(parents=True, exist_ok=True)
                overlay_output.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(colored).save(color_output)
                Image.fromarray(blended).save(overlay_output)

                ids, counts = np.unique(instance_ids, return_counts=True)
                records.append(
                    {
                        "bin": bin_name,
                        "camera": camera,
                        "source_image": str(image_path),
                        "source_mask": str(mask_path),
                        "colored_mask": str(color_output),
                        "overlay": str(overlay_output),
                        "instance_pixel_counts": {
                            str(int(key)): int(value)
                            for key, value in zip(ids, counts)
                            if key != 0
                        },
                    }
                )

    report: dict[str, object] = {
        "format": "rgb_self_recovery_colored_mask_examples_v1",
        "source_root": str(root),
        "bins": list(bins),
        "examples_per_camera_folder": examples_per_folder,
        "example_count": len(records),
        "examples": records,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--distance-bins",
        nargs="+",
        default=list(DEFAULT_BINS),
        help="Bin directory names (default: bins from 0 through 120 m).",
    )
    parser.add_argument("--examples-per-folder", type=int, default=5)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    args = parser.parse_args()
    if args.examples_per_folder < 1:
        parser.error("--examples-per-folder must be at least 1")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        parser.error("--overlay-alpha must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    report = export_examples(
        args.dataset_root,
        args.output_dir,
        tuple(args.distance_bins),
        args.examples_per_folder,
        args.overlay_alpha,
    )
    print(
        f"Wrote {report['example_count']} colored masks and overlays to "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
