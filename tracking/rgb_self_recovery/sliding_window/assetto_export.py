"""Read the distance-binned Assetto export without copying its RGB images."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from Assetto_data_prep.common import (
    build_aligned_mesh,
    cad_to_camera_pose,
    validate_model_row,
)


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def discover_bins(root: Path) -> list[Path]:
    bins = sorted(
        path for path in root.iterdir()
        if path.is_dir() and (path / "source_manifest.csv").is_file()
    )
    if not bins:
        raise RuntimeError(f"No distance-bin exports found under {root}")
    return bins


def discover_runs(root: Path) -> list[str]:
    path = root / "source_runs.txt"
    if path.is_file():
        values = [line.strip() for line in path.read_text().splitlines()]
        values = [value for value in values if value and value != "source_run"]
        if values:
            return values
    runs: set[str] = set()
    for bin_dir in discover_bins(root):
        runs.update(row["source_run"] for row in _rows(bin_dir / "source_manifest.csv"))
    return sorted(runs)


def selected_runs(
    root: Path,
    split: str,
    validation_runs: list[str] | None,
    validation_run_count: int,
) -> tuple[set[str], list[str]]:
    runs = discover_runs(root)
    unknown = sorted(set(validation_runs or ()) - set(runs))
    if unknown:
        raise ValueError(f"Unknown --assetto-validation-run values: {unknown}")
    held_out = list(validation_runs or runs[-validation_run_count:])
    if not held_out or len(held_out) >= len(runs):
        raise ValueError("The held-out run list must leave at least one training run")
    if split == "train":
        chosen = set(runs) - set(held_out)
    elif split == "validation":
        chosen = set(held_out)
    elif split == "all":
        chosen = set(runs)
    else:
        raise ValueError("Assetto split must be train, validation, or all")
    return chosen, held_out


def _key(row: dict[str, str]) -> tuple[str, int, str]:
    return row["source_run"], int(row["frame"]), row["camera_id"]


def normalize_manifest(
    manifest: dict[str, str], camera: dict[str, str]
) -> dict[str, str]:
    """Normalize both cleaned-export manifest schemas used by Assetto data."""
    normalized = dict(manifest)
    source_frame = normalized.get("source_frame") or normalized.get("frame")
    if source_frame is None:
        raise ValueError("Assetto source manifest has neither source_frame nor frame")
    normalized["source_frame"] = source_frame
    normalized.setdefault("frame", source_frame)
    normalized.setdefault("sim_time_ms", camera["sim_time_ms"])

    output_filename = normalized.get("output_filename")
    camera_id = normalized.get("camera_id")
    if not normalized.get("image"):
        if not output_filename or not camera_id:
            raise ValueError(
                "Assetto source manifest needs image or output_filename/camera_id"
            )
        normalized["image"] = str(Path("images") / camera_id / output_filename)
    if not normalized.get("masks") and output_filename and camera_id:
        mask_filename = str(Path(output_filename).with_suffix(".png"))
        normalized["masks"] = str(Path("masks") / camera_id / mask_filename)
    if not normalized.get("masks_visualized") and output_filename and camera_id:
        mask_filename = str(Path(output_filename).with_suffix(".png"))
        normalized["masks_visualized"] = str(
            Path("masks_visualized") / camera_id / mask_filename
        )
    return normalized


def load_index(
    root: Path,
    *,
    skip_incomplete: bool = False,
    skipped: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Join manifest, camera, transform, and packed-mask-ID metadata."""
    output: list[dict[str, object]] = []
    seen: set[tuple[str, int, str]] = set()
    for bin_dir in discover_bins(root):
        manifest_rows = _rows(bin_dir / "source_manifest.csv")
        # Cleaners may retain an empty far-distance bin without creating CSV tables.
        if not manifest_rows:
            continue
        cameras = {_key(row): row for row in _rows(bin_dir / "csv/camera_frames.csv")}
        transforms = {_key(row): row for row in _rows(bin_dir / "csv/transforms.csv")}
        boxes = {_key(row): row for row in _rows(bin_dir / "csv/bboxes_3d.csv")}
        for raw_manifest in manifest_rows:
            source_frame = raw_manifest.get("source_frame") or raw_manifest.get("frame")
            if source_frame is None:
                raise ValueError(
                    f"Manifest under {bin_dir} has neither source_frame nor frame"
                )
            key = (
                raw_manifest["source_run"],
                int(source_frame),
                raw_manifest["camera_id"],
            )
            if key in seen:
                raise ValueError(f"Duplicate Assetto frame across distance bins: {key}")
            seen.add(key)
            missing = [
                name for name, values in (
                    ("camera_frames", cameras),
                    ("transforms", transforms),
                    ("bboxes_3d", boxes),
                )
                if key not in values
            ]
            if missing:
                if not skip_incomplete:
                    raise ValueError(
                        f"Incomplete CSV join for Assetto frame {key}: {missing}"
                    )
                if skipped is not None:
                    skipped.append({
                        "source_run": key[0],
                        "source_frame": key[1],
                        "camera_id": key[2],
                        "distance_bin": bin_dir.name,
                        "reason": "incomplete_csv_join",
                        "missing": missing,
                    })
                continue
            camera, transform, box = cameras[key], transforms[key], boxes[key]
            manifest = normalize_manifest(raw_manifest, camera)
            output.append({
                "key": key,
                "manifest": manifest,
                "camera": camera,
                "transform": transform,
                "box": box,
                "bin_dir": bin_dir,
                "bin_name": bin_dir.name,
            })
    output.sort(key=lambda item: item["key"])
    return output


def _even_allocations(capacities: dict[str, int], target: int) -> dict[str, int]:
    """Round-robin quotas, redistributing capacity from short runs."""
    allocation = {key: 0 for key in sorted(capacities)}
    remaining = int(target)
    while remaining:
        eligible = [
            key for key in sorted(capacities)
            if allocation[key] < capacities[key]
        ]
        if not eligible:
            break
        for key in eligible:
            if not remaining:
                break
            allocation[key] += 1
            remaining -= 1
    if remaining:
        raise ValueError(f"Could not allocate {target} samples from {capacities}")
    return allocation


def balanced_consecutive_subset(
    index: list[dict[str, object]],
    frames_per_bin: int | None,
    short_bin_policy: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Select balanced consecutive blocks without duplicating images."""
    if frames_per_bin is None:
        counts: dict[str, int] = defaultdict(int)
        for item in index:
            counts[str(item["bin_name"])] += 1
        return index, {
            "frames_per_bin_requested": None,
            "short_bin_policy": short_bin_policy,
            "bins": {
                key: {"available": value, "selected": value}
                for key, value in sorted(counts.items())
            },
        }
    if frames_per_bin < 1:
        raise ValueError("--assetto-frames-per-bin must be positive")
    if short_bin_policy not in {"error", "all"}:
        raise ValueError("short-bin policy must be error or all")

    by_bin: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in index:
        by_bin[str(item["bin_name"])].append(item)
    selected: list[dict[str, object]] = []
    report: dict[str, object] = {
        "frames_per_bin_requested": int(frames_per_bin),
        "short_bin_policy": short_bin_policy,
        "selection": "even run quotas; consecutive block per run and camera",
        "bins": {},
    }
    for bin_name, items in sorted(by_bin.items()):
        available = len(items)
        if available < frames_per_bin and short_bin_policy == "error":
            raise ValueError(
                f"{bin_name} has only {available} unique images; cannot select "
                f"{frames_per_bin} without duplication. Use "
                "--assetto-short-bin-policy all to keep every available image."
            )
        target = min(frames_per_bin, available)
        by_run: dict[str, list[dict[str, object]]] = defaultdict(list)
        for item in items:
            by_run[str(item["key"][0])].append(item)
        run_allocations = _even_allocations(
            {key: len(value) for key, value in by_run.items()}, target
        )
        per_run_report = {}
        for run, run_items in sorted(by_run.items()):
            run_target = run_allocations[run]
            by_camera: dict[str, list[dict[str, object]]] = defaultdict(list)
            for item in run_items:
                by_camera[str(item["key"][2])].append(item)
            camera_allocations = _even_allocations(
                {key: len(value) for key, value in by_camera.items()}, run_target
            )
            camera_report = {}
            for camera, camera_items in sorted(by_camera.items()):
                camera_items.sort(key=lambda item: int(item["key"][1]))
                count = camera_allocations[camera]
                start = max(0, (len(camera_items) - count) // 2)
                chosen = camera_items[start:start + count]
                selected.extend(chosen)
                camera_report[camera] = {
                    "available": len(camera_items),
                    "selected": count,
                    "first_frame": None if not chosen else int(chosen[0]["key"][1]),
                    "last_frame": None if not chosen else int(chosen[-1]["key"][1]),
                }
            per_run_report[run] = {
                "available": len(run_items),
                "selected": run_target,
                "cameras": camera_report,
            }
        report["bins"][bin_name] = {
            "available": available,
            "selected": target,
            "short": available < frames_per_bin,
            "runs": per_run_report,
        }
    selected.sort(key=lambda item: item["key"])
    return selected, report


def select_bins(
    index: list[dict[str, object]], requested: str,
) -> tuple[list[dict[str, object]], list[str]]:
    available = sorted({str(item["bin_name"]) for item in index})
    if requested.strip().lower() == "all":
        return index, available
    selected_names = [value.strip() for value in requested.split(",") if value.strip()]
    unknown = sorted(set(selected_names) - set(available))
    if unknown:
        raise ValueError(
            f"Unknown Assetto distance bins {unknown}; available bins: {available}"
        )
    if not selected_names:
        raise ValueError("--assetto-distance-bins must select at least one bin")
    selected_set = set(selected_names)
    return [item for item in index if item["bin_name"] in selected_set], selected_names


def make_alignment(args, index: list[dict[str, object]]):
    if args.mesh is None:
        raise ValueError("--mesh is required for a raw Assetto export")
    reference = index[0]["transform"]
    return build_aligned_mesh(
        args.mesh,
        reference,
        args.mesh_scale,
        args.cad_axis_convention,
        args.fit_aabb,
    )


def _decode_mask(path: Path, instance_id: int, shape: tuple[int, int]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint32)
    if rgb.shape[:2] != shape:
        raise ValueError(f"Mask/image shape mismatch for {path}: {rgb.shape[:2]} != {shape}")
    packed = rgb[..., 0] + (rgb[..., 1] << 8) + (rgb[..., 2] << 16)
    return packed == int(instance_id)


def iter_frames(args, index, alignment) -> Iterator[tuple]:
    chosen, _ = selected_runs(
        args.assetto_export_root,
        args.assetto_split,
        args.assetto_validation_run,
        args.assetto_validation_run_count,
    )
    cameras = {value.strip() for value in args.assetto_cameras.split(",") if value.strip()}
    if not cameras:
        raise ValueError("--assetto-cameras must select at least one camera")
    for item in index:
        source_run, frame_id, camera_id = item["key"]
        if source_run not in chosen or camera_id not in cameras:
            continue
        row, camera, box = item["transform"], item["camera"], item["box"]
        if args.assetto_exclude_truncated and int(row["truncated"]):
            continue
        validate_model_row(alignment, row)
        pose = cad_to_camera_pose(row, alignment.local_center)
        if pose[2, 3] <= 1e-5 or pose[2, 3] > args.assetto_max_depth_m:
            continue
        image_path = item["bin_dir"] / item["manifest"]["image"]
        mask_name = item["manifest"].get("masks", "")
        mask_path = item["bin_dir"] / mask_name if mask_name else None
        image = np.asarray(Image.open(image_path).convert("RGB"))
        mask = None
        if mask_path is not None and mask_path.is_file():
            mask = _decode_mask(mask_path, int(box["instance_id"]), image.shape[:2])
        if mask is not None and int(mask.sum()) < args.assetto_min_mask_pixels:
            mask = None
        if mask is None and args.mask_fallback != "render":
            continue
        K = np.asarray([
            [float(camera["fx"]), 0.0, float(camera["cx"])],
            [0.0, float(camera["fy"]), float(camera["cy"])],
            [0.0, 0.0, 1.0],
        ])
        if args.assetto_crop_shaded_sides:
            side = int(args.assetto_shaded_side_pixels)
            if side <= 0 or 2 * side >= image.shape[1]:
                raise ValueError(
                    "--assetto-shaded-side-pixels must leave a positive image width"
                )
            image = image[:, side:-side]
            if mask is not None:
                mask = mask[:, side:-side]
            K[0, 2] -= side
        key = f"{source_run}__{camera_id}__{frame_id:06d}"
        gt = [{
            "cam_R_m2c": pose[:3, :3].reshape(-1).tolist(),
            "cam_t_m2c": (pose[:3, 3] * 1000.0).tolist(),
            "obj_id": 1,
        }]
        extra = {
            "masks": {},
            "mask_arrays": {} if mask is None else {"0": mask},
            "source_run": source_run,
            "camera_id": camera_id,
            "frame_id": frame_id,
            "sim_time_ms": int(item["manifest"]["sim_time_ms"]),
        }
        yield key, image, K, None, gt, extra
