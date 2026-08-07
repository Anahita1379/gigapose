"""Add aligned, training-only depth crops to existing recovery shards."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import shutil
import tarfile

import cv2
import numpy as np

from tracking.rgb_self_recovery.generate_dataset import RecoveryShardWriter, encode_npz
from tracking.rgb_self_recovery.render_inputs import CropSpec, warp_mask


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--input-data", type=Path, required=True)
    value.add_argument("--depth-root", type=Path)
    value.add_argument(
        "--depth-manifest-root",
        type=Path,
        help="Root containing distance*/depth_info_manifest.csv files.",
    )
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument(
        "--depth-pattern",
        default="{source_run}/{camera_id}/depth/{frame_id:06d}.npy",
        help="Relative path template; fields: source_key/source_run/camera_id/frame_id.",
    )
    value.add_argument("--depth-unit-scale", type=float, default=1.0)
    value.add_argument("--minimum-depth-m", type=float, default=0.05)
    value.add_argument("--maximum-depth-m", type=float, default=200.0)
    value.add_argument("--groups-per-shard", type=int, default=500)
    value.add_argument("--skip-missing", action="store_true")
    value.add_argument("--overwrite", action="store_true")
    return value


def source_fields(metadata):
    source_key = metadata["source_key"]
    source_run, camera_id, frame = source_key.rsplit("__", 2)
    return {
        "source_key": source_key,
        "source_run": source_run,
        "camera_id": camera_id,
        "frame_id": int(frame),
    }


def load_depth_index(root: Path) -> dict[tuple[str, int, str], Path]:
    manifests = sorted(root.glob("distance*/depth_info_manifest.csv"))
    if not manifests:
        raise FileNotFoundError(f"No distance*/depth_info_manifest.csv under {root}")
    output = {}
    for manifest in manifests:
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") != "written" or not row.get("depth_info"):
                    continue
                key = (row["source_run"], int(row["frame"]), row["camera_id"])
                path = manifest.parent / row["depth_info"]
                if key in output and output[key] != path:
                    raise ValueError(f"Duplicate pseudo-depth index key {key}")
                output[key] = path
    if not output:
        raise ValueError(f"Depth manifests under {root} contain no written rows")
    return output


def read_depth(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        value = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as payload:
            if {"depth_m", "valid", "bbox_xyxy", "image_shape"} <= set(payload.files):
                shape = tuple(np.asarray(payload["image_shape"], dtype=int).tolist())
                x0, y0, x1, y1 = np.asarray(payload["bbox_xyxy"], dtype=int)
                crop = np.asarray(payload["depth_m"], dtype=np.float32)
                valid = np.asarray(payload["valid"], dtype=bool)
                if crop.shape != (y1 - y0, x1 - x0) or valid.shape != crop.shape:
                    raise ValueError(f"Invalid sparse depth geometry in {path}")
                value = np.zeros(shape, dtype=np.float32)
                value[y0:y1, x0:x1] = np.where(valid, crop, 0.0)
            else:
                key = "depth" if "depth" in payload.files else payload.files[0]
                value = np.asarray(payload[key]).copy()
    else:
        value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if value is None:
            raise ValueError(f"Could not decode depth image {path}")
    value = np.asarray(value)
    if value.ndim == 3:
        value = value[..., 0]
    if value.ndim != 2:
        raise ValueError(f"Depth must be HxW, got {value.shape} from {path}")
    return value.astype(np.float32)


def crop_depth(depth: np.ndarray, crop_values: np.ndarray, valid: np.ndarray):
    x0, y0, side, output_size = np.asarray(crop_values, dtype=float).reshape(4)
    crop = CropSpec(x0, y0, side, int(round(output_size)))
    value = cv2.warpAffine(
        depth, crop.affine, (crop.output_size, crop.output_size),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return value, warp_mask(valid, crop)


def records(path: Path):
    pending = {}
    with tarfile.open(path) as archive:
        for member in archive:
            if not member.isfile():
                continue
            key = None
            for extension, name in ((".rgb.jpg", "rgb"), (".data.npz", "data"), (".json", "meta")):
                if member.name.endswith(extension):
                    key = member.name.removesuffix(extension)
                    handle = archive.extractfile(member)
                    if handle is not None:
                        pending.setdefault(key, {})[name] = handle.read()
                    break
            if key is not None and {"rgb", "data", "meta"} <= pending[key].keys():
                yield key, pending.pop(key)


def main():
    args = parser().parse_args()
    if (args.depth_root is None) == (args.depth_manifest_root is None):
        raise ValueError("Pass exactly one of --depth-root or --depth-manifest-root")
    if args.depth_unit_scale <= 0 or args.minimum_depth_m < 0:
        raise ValueError("Depth scale must be positive and minimum non-negative")
    if args.maximum_depth_m <= args.minimum_depth_m:
        raise ValueError("Maximum depth must exceed minimum depth")
    manifest = json.loads((args.input_data / "manifest.json").read_text())
    if manifest.get("format") != "rgb_render_self_recovery_v1":
        raise ValueError("Input must be existing RGB recovery data")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = RecoveryShardWriter(args.output_dir, args.groups_per_shard)
    depth_index = (
        load_depth_index(args.depth_manifest_root)
        if args.depth_manifest_root is not None
        else None
    )
    written = missing = invalid = 0
    try:
        for shard in sorted(args.input_data.glob("shard-*.tar")):
            for key, record in records(shard):
                metadata = json.loads(record["meta"])
                fields = source_fields(metadata)
                if depth_index is None:
                    path = args.depth_root / args.depth_pattern.format(**fields)
                else:
                    path = depth_index.get(
                        (fields["source_run"], fields["frame_id"], fields["camera_id"])
                    )
                    if path is None:
                        path = args.depth_manifest_root / "__missing_depth__"
                if not path.is_file():
                    missing += 1
                    if args.skip_missing:
                        continue
                    raise FileNotFoundError(path)
                depth = read_depth(path) * args.depth_unit_scale
                shaded = manifest.get("assetto_shaded_side_crop", {})
                if shaded.get("enabled"):
                    side = int(shaded["pixels_per_side"])
                    if side <= 0 or 2 * side >= depth.shape[1]:
                        raise ValueError(f"Invalid shaded-side crop for depth {path}")
                    depth = depth[:, side:-side]
                valid = np.isfinite(depth) & (depth >= args.minimum_depth_m) & (depth <= args.maximum_depth_m)
                with np.load(io.BytesIO(record["data"])) as payload:
                    values = {name: np.asarray(payload[name]).copy() for name in payload.files}
                depth_crop, valid_crop = crop_depth(depth, values["crop"], valid)
                object_valid = valid_crop & values["observed_mask"].astype(bool)
                if not object_valid.any():
                    invalid += 1
                    if args.skip_missing:
                        continue
                    raise ValueError(f"No valid object depth for {key} from {path}")
                values["observed_depth_m"] = np.where(object_valid, depth_crop, 0).astype(np.float32)
                values["observed_depth_valid"] = object_valid.astype(np.bool_)
                metadata["privileged_depth_path"] = str(path)
                writer.write(key, record["rgb"], encode_npz(**values), metadata)
                written += 1
    finally:
        writer.close()
    output = {
        **manifest,
        "format": "rgb_render_self_recovery_privileged_depth_v1",
        "uses_observed_depth": "training_only",
        "source_recovery_data": str(args.input_data),
        "depth_root": None if args.depth_root is None else str(args.depth_root),
        "depth_manifest_root": (
            None if args.depth_manifest_root is None else str(args.depth_manifest_root)
        ),
        "depth_pattern": args.depth_pattern,
        "depth_unit_scale": args.depth_unit_scale,
        "groups": written,
        "missing_depth_groups": missing,
        "invalid_depth_groups": invalid,
        "shards": writer.shard_counts,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
