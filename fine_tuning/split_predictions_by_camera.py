"""Split GigaPose prediction files into per-camera folders.

``Assetto_data_prep.prepare_inference --cameras all`` writes one BOP-style test
dataset because GigaPose expects a single ``test`` split and saves one combined
prediction CSV.  The prepared Assetto dataset also writes ``frame_map.json``,
which records the original ``camera_id`` for each ``scene_id``/``im_id``.

This helper uses that map to copy rows from prediction CSVs into:

    output_dir/front/<prediction-name>.csv
    output_dir/rear/<prediction-name>.csv
    ...

It also splits the intermediate per-batch ``.npz`` files written by GigaPose:

    output_dir/front/0.npz
    output_dir/rear/0.npz
    ...

For ``.npz`` files, arrays whose first dimension matches ``scene_id``/``im_id``
are filtered along that first dimension. Other arrays are copied unchanged.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split GigaPose prediction CSV/NPZ files by Assetto camera_id."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="A prediction .csv/.npz file, or a directory containing them.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Prepared Assetto dataset directory containing frame_map.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for camera subfolders. Defaults to "
            "<prediction parent>/by_camera."
        ),
    )
    parser.add_argument(
        "--unknown-camera-name",
        default="unknown_camera",
        help="Folder name for prediction rows missing from frame_map.json.",
    )
    return parser.parse_args()


def load_camera_map(dataset_dir: Path) -> dict[tuple[int, int], str]:
    frame_map_path = dataset_dir / "frame_map.json"
    if not frame_map_path.is_file():
        raise FileNotFoundError(f"Missing {frame_map_path}")

    import json

    camera_map: dict[tuple[int, int], str] = {}
    rows = json.loads(frame_map_path.read_text())
    for row in rows:
        key = (int(row["scene_id"]), int(row["im_id"]))
        camera_map[key] = str(row["camera_id"])
    return camera_map


def safe_folder_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)


def camera_for_key(
    camera_map: dict[tuple[int, int], str],
    scene_id: int,
    im_id: int,
    unknown_camera_name: str,
) -> tuple[str, bool]:
    camera_id = camera_map.get((scene_id, im_id))
    if camera_id is None:
        return unknown_camera_name, True
    return camera_id, False


def split_csv(
    path: Path,
    output_dir: Path,
    camera_map: dict[tuple[int, int], str],
    unknown_camera_name: str,
) -> tuple[int, int]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        rows_by_camera: dict[str, list[dict[str, str]]] = defaultdict(list)
        missing = 0
        for row in reader:
            camera_id, is_missing = camera_for_key(
                camera_map,
                int(row["scene_id"]),
                int(row["im_id"]),
                unknown_camera_name,
            )
            missing += int(is_missing)
            rows_by_camera[camera_id].append(row)

    for camera_id, rows in sorted(rows_by_camera.items()):
        camera_dir = output_dir / safe_folder_name(camera_id)
        camera_dir.mkdir(parents=True, exist_ok=True)
        output_path = camera_dir / path.name
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"{path.name}: {camera_id}: {len(rows)} rows -> {output_path}")
    return sum(len(rows) for rows in rows_by_camera.values()), missing


def split_npz(
    path: Path,
    output_dir: Path,
    camera_map: dict[tuple[int, int], str],
    unknown_camera_name: str,
) -> tuple[int, int]:
    data = np.load(path)
    if "scene_id" not in data.files or "im_id" not in data.files:
        raise ValueError(f"{path} does not contain scene_id/im_id arrays")

    scene_ids = np.asarray(data["scene_id"]).reshape(-1)
    im_ids = np.asarray(data["im_id"]).reshape(-1)
    if len(scene_ids) != len(im_ids):
        raise ValueError(f"{path} has mismatched scene_id/im_id lengths")

    indices_by_camera: dict[str, list[int]] = defaultdict(list)
    missing = 0
    for idx, (scene_id, im_id) in enumerate(zip(scene_ids, im_ids)):
        camera_id, is_missing = camera_for_key(
            camera_map,
            int(scene_id),
            int(im_id),
            unknown_camera_name,
        )
        missing += int(is_missing)
        indices_by_camera[camera_id].append(idx)

    n_samples = len(scene_ids)
    for camera_id, indices in sorted(indices_by_camera.items()):
        camera_dir = output_dir / safe_folder_name(camera_id)
        camera_dir.mkdir(parents=True, exist_ok=True)
        output_path = camera_dir / path.name
        idx = np.asarray(indices, dtype=int)
        out = {}
        for name in data.files:
            value = data[name]
            if value.shape[:1] == (n_samples,):
                out[name] = value[idx]
            else:
                out[name] = value
        np.savez(output_path, **out)
        print(f"{path.name}: {camera_id}: {len(indices)} samples -> {output_path}")
    return n_samples, missing


def prediction_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(
        [
            item
            for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in {".csv", ".npz"}
        ],
        key=lambda item: (item.suffix.lower(), item.name),
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (
        args.predictions / "by_camera"
        if args.predictions.is_dir()
        else args.predictions.parent / "by_camera"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_map = load_camera_map(args.dataset_dir)

    total_rows = 0
    total_missing = 0
    files = prediction_files(args.predictions)
    if not files:
        raise FileNotFoundError(f"No .csv or .npz prediction files in {args.predictions}")

    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            rows, missing = split_csv(
                path, output_dir, camera_map, args.unknown_camera_name
            )
        elif suffix == ".npz":
            rows, missing = split_npz(
                path, output_dir, camera_map, args.unknown_camera_name
            )
        else:
            continue
        total_rows += rows
        total_missing += missing

    print(
        f"Split {len(files)} prediction file(s), {total_rows} row/sample entries "
        f"under {output_dir}"
    )
    if total_missing:
        print(
            f"Warning: {total_missing} entries were not found in frame_map.json and were "
            f"written to {args.unknown_camera_name}/"
        )


if __name__ == "__main__":
    main()
