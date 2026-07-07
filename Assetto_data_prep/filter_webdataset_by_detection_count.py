#!/usr/bin/env python3
"""Filter a BOP/WebDataset split by independent detection counts.

This is meant for the "sanity filter" workflow:

1. Export/run an independent detector/segmenter such as Grounded-SAM on the RGB
   images from a prepared split.
2. Convert those detections to the usual CNOS/FastSAM/Grounded-SAM JSON format
   with scene_id + image_id/im_id entries.
3. Keep only WebDataset samples where the detector saw at least as many car
   masks as the GT Assetto/GigaPose training mask says are visible.

The script does not run Grounded-SAM itself. It rewrites the tar shards after
Grounded-SAM detections already exist.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-split",
        type=Path,
        required=True,
        help="Existing split dir, e.g. gigaPose_datasets/datasets/assettocorsa_all/val_pbr_web.",
    )
    parser.add_argument(
        "--output-split",
        type=Path,
        required=True,
        help="New cleaned split dir to write, e.g. .../val_pbr_web_gsam_clean.",
    )
    parser.add_argument(
        "--detections",
        type=Path,
        required=True,
        help=(
            "Grounded-SAM/CNOS-style detection JSON. Each row should have scene_id, "
            "image_id or im_id, and optionally score/bbox/segmentation."
        ),
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Optional detector score threshold before counting GSAM detections.",
    )
    parser.add_argument(
        "--min-gsam-detections",
        type=int,
        default=0,
        help="Optional absolute minimum number of GSAM detections required.",
    )
    parser.add_argument(
        "--reject-if-gt-more-than-gsam",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Default: reject samples where GT visible car count > GSAM detection count.",
    )
    parser.add_argument(
        "--max-samples-per-shard",
        type=int,
        default=1000,
        help="How many samples to put into each output shard.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sample_key_from_member(name: str) -> str:
    return name.split(".", 1)[0]


def grouped_members(tar: tarfile.TarFile) -> dict[str, list[tarfile.TarInfo]]:
    groups: dict[str, list[tarfile.TarInfo]] = defaultdict(list)
    for member in tar.getmembers():
        if not member.isfile():
            continue
        groups[sample_key_from_member(member.name)].append(member)
    return dict(groups)


def read_member_bytes(tar: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    handle = tar.extractfile(member)
    if handle is None:
        raise FileNotFoundError(member.name)
    return handle.read()


def load_sample_json(
    tar: tarfile.TarFile, members: list[tarfile.TarInfo], suffix: str
) -> Any:
    for member in members:
        if member.name.endswith(suffix):
            return json.loads(read_member_bytes(tar, member).decode("utf-8"))
    raise KeyError(f"No {suffix} in sample {[m.name for m in members]}")


def gt_count_from_sample(tar: tarfile.TarFile, members: list[tarfile.TarInfo]) -> int:
    # mask_visib.json is the most direct "visible mask" count. Fall back to
    # gt.json if an older shard does not have mask_visib.json.
    try:
        mask_visib = load_sample_json(tar, members, ".mask_visib.json")
        if isinstance(mask_visib, dict):
            return len(mask_visib)
        if isinstance(mask_visib, list):
            return len(mask_visib)
    except Exception:
        pass

    gt = load_sample_json(tar, members, ".gt.json")
    return len(gt) if isinstance(gt, list) else 0


def load_detection_counts(path: Path, min_score: float | None) -> dict[tuple[int, int], int]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        rows = list(data.values())
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(f"Unsupported detection JSON format: {path}")

    counts: dict[tuple[int, int], int] = defaultdict(int)
    for row in rows:
        if not isinstance(row, dict):
            continue
        if "objects" in row and isinstance(row["objects"], list):
            if "scene_id" not in row:
                key = row.get("key")
                if key and isinstance(key, str) and "_" in key:
                    scene_text, im_text = key.split("_", 1)
                    scene_id = int(scene_text)
                    image_id = int(im_text)
                else:
                    continue
            else:
                scene_id = int(row["scene_id"])
                image_id = int(row.get("image_id", row.get("im_id", row.get("frame_index", 0))))
            for obj in row["objects"]:
                if not isinstance(obj, dict):
                    continue
                if min_score is not None and float(obj.get("score", 1.0)) < min_score:
                    continue
                counts[(scene_id, image_id)] += 1
            continue

        if min_score is not None and float(row.get("score", 1.0)) < min_score:
            continue
        if "scene_id" not in row:
            continue
        image_id = row.get("image_id", row.get("im_id"))
        if image_id is None:
            continue
        counts[(int(row["scene_id"]), int(image_id))] += 1
    return dict(counts)


def parse_key(key: str) -> tuple[int, int]:
    scene_id, im_id = key.split("_", 1)
    return int(scene_id), int(im_id)


def add_bytes_to_tar(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def output_shard_path(output_split: Path, shard_index: int) -> Path:
    return output_split / f"shard-{shard_index:06d}.tar"


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["key"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.max_samples_per_shard <= 0:
        raise ValueError("--max-samples-per-shard must be positive")
    if args.output_split.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_split} exists; pass --overwrite")
        shutil.rmtree(args.output_split)
    args.output_split.mkdir(parents=True, exist_ok=True)

    detection_counts = load_detection_counts(args.detections, args.min_score)

    kept_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    key_to_shard: dict[str, int] = {}

    output_tar: tarfile.TarFile | None = None
    output_shard_index = -1
    samples_in_current_shard = 0

    def open_next_output_shard() -> tarfile.TarFile:
        nonlocal output_tar, output_shard_index, samples_in_current_shard
        if output_tar is not None:
            output_tar.close()
        output_shard_index += 1
        samples_in_current_shard = 0
        return tarfile.open(output_shard_path(args.output_split, output_shard_index), "w")

    output_tar = open_next_output_shard()
    try:
        for shard_path in sorted(args.input_split.glob("shard-*.tar")):
            with tarfile.open(shard_path) as in_tar:
                groups = grouped_members(in_tar)
                for key in sorted(groups):
                    scene_id, im_id = parse_key(key)
                    gt_count = gt_count_from_sample(in_tar, groups[key])
                    gsam_count = detection_counts.get((scene_id, im_id), 0)
                    keep = gsam_count >= args.min_gsam_detections
                    if args.reject_if_gt_more_than_gsam and gt_count > gsam_count:
                        keep = False
                    row = {
                        "key": key,
                        "scene_id": scene_id,
                        "im_id": im_id,
                        "gt_visible_count": gt_count,
                        "gsam_detection_count": gsam_count,
                        "input_shard": str(shard_path),
                        "kept": keep,
                    }
                    if keep:
                        if samples_in_current_shard >= args.max_samples_per_shard:
                            output_tar = open_next_output_shard()
                        assert output_tar is not None
                        for member in sorted(groups[key], key=lambda item: item.name):
                            add_bytes_to_tar(output_tar, member.name, read_member_bytes(in_tar, member))
                        key_to_shard[key] = output_shard_index
                        samples_in_current_shard += 1
                        kept_rows.append(row)
                    else:
                        rejected_rows.append(row)
    finally:
        if output_tar is not None:
            output_tar.close()

    # If nothing was kept, remove the empty first shard to avoid confusing
    # downstream readers.
    if not kept_rows:
        for shard_path in args.output_split.glob("shard-*.tar"):
            shard_path.unlink()

    (args.output_split / "key_to_shard.json").write_text(json.dumps(key_to_shard, indent=2))
    write_rows(args.output_split / "kept_samples.csv", kept_rows)
    write_rows(args.output_split / "rejected_samples.csv", rejected_rows)
    report = {
        "input_split": str(args.input_split),
        "output_split": str(args.output_split),
        "detections": str(args.detections),
        "min_score": args.min_score,
        "min_gsam_detections": args.min_gsam_detections,
        "reject_if_gt_more_than_gsam": args.reject_if_gt_more_than_gsam,
        "kept_samples": len(kept_rows),
        "rejected_samples": len(rejected_rows),
        "input_samples": len(kept_rows) + len(rejected_rows),
    }
    (args.output_split / "filter_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
