"""Prepare full cleaned Assetto sequences as one GigaPose/WebDataset dataset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import io
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import webdataset as wds
from bop_toolkit_lib import pycoco_utils

from Assetto_data_prep.common import (
    bbox_from_mask,
    build_aligned_mesh,
    cad_to_camera_pose,
    key_to_shard_map,
    validate_model_row,
    write_models,
)
from tracking.rgb_self_recovery.sliding_window.assetto_export import load_index


DEFAULT_SOURCE = Path(
    "/mnt/ssd2tb/.local_share_backup/Steam/steamapps/common/assettocorsa/"
    "apps/lua/multi_cam_obs/frames/Assettocorsa_new_dataset_distance_bin_cleaned"
)
DEFAULT_CAD = Path("gigaPose_datasets/datasets/racecar/models/obj_000001.ply")
DEFAULT_VALIDATION_RUN = "20260730_laguna2026_fog_1opp_2laps_mostlyFront"
DEFAULT_EVALUATION_RUN = "20260730_putnam_rain_1opp_2lap_farRear"
DEFAULT_BINS = (
    "distance0_20,distance20_40,distance40_60,distance60_80,"
    "distance80_100,distance100_120"
)
ROLE_DATASET_NAMES = {
    "train": "rgb_recovery_gru_train_assetto",
    "validation": "rgb_recovery_gru_validation_laguna_putnam",
    "evaluation": "rgb_recovery_gru_evaluation_putnam_rear",
    "benchmark": "rgb_recovery_benchmark_assetto_20260803",
}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--assetto-export-root", type=Path, default=DEFAULT_SOURCE)
    value.add_argument("--role", choices=tuple(ROLE_DATASET_NAMES), required=True)
    value.add_argument("--dataset-name")
    value.add_argument(
        "--output-root", type=Path, default=Path("gigaPose_datasets/datasets")
    )
    value.add_argument("--validation-run", default=DEFAULT_VALIDATION_RUN)
    value.add_argument("--evaluation-run", default=DEFAULT_EVALUATION_RUN)
    value.add_argument(
        "--include-runs",
        help=(
            "Explicit comma-separated physical runs, or 'all'. Overrides the "
            "role's default run selection; useful for leakage-safe custom splits."
        ),
    )
    value.add_argument(
        "--exclude-runs",
        help="Optional comma-separated physical runs removed after inclusion.",
    )
    value.add_argument("--distance-bins", default=DEFAULT_BINS)
    value.add_argument("--cameras", default="front,rear")
    value.add_argument("--minimum-mask-pixels", type=int, default=64)
    value.add_argument("--maximum-depth-m", type=float, default=120.0)
    value.add_argument(
        "--validation-frames-per-run",
        type=int,
        default=164,
        help=(
            "Cap each validation run/camera to this many frames. A consecutive "
            "window with the best distance-bin diversity is selected; 0 disables."
        ),
    )
    value.add_argument("--cad-path", type=Path, default=DEFAULT_CAD)
    value.add_argument("--cad-scale", type=float, default=1.0)
    value.add_argument(
        "--cad-axis-convention",
        choices=("x-forward-z-up", "csp"),
        default="x-forward-z-up",
    )
    value.add_argument(
        "--fit-aabb", choices=("none", "uniform", "nonuniform"), default="nonuniform"
    )
    value.add_argument("--object-id", type=int, default=1)
    value.add_argument("--max-shard-size", type=int, default=500)
    value.add_argument(
        "--template-source-dataset", default="assettocorsa_new_dataset"
    )
    value.add_argument(
        "--expected-cleaned-width",
        type=int,
        default=1548,
        help="Reject inputs that do not have the already shade-cleaned width; 0 disables.",
    )
    value.add_argument("--max-frames", type=int)
    value.add_argument("--overwrite", action="store_true")
    return value


def selected_runs(
    available: set[str], role: str, validation_run: str, evaluation_run: str
) -> set[str]:
    if role == "benchmark":
        if not available:
            raise ValueError("Benchmark export contains no runs")
        return set(available)
    held_out = {validation_run, evaluation_run}
    missing = held_out - available
    if missing:
        raise ValueError(f"Held-out runs are absent from the export: {sorted(missing)}")
    if validation_run == evaluation_run:
        raise ValueError("Validation and evaluation runs must differ")
    if role == "train":
        result = available - held_out
    elif role == "validation":
        # Validate on both camera regimes: Laguna/front and Putnam/rear.
        result = held_out
    elif role == "evaluation":
        result = {evaluation_run}
    else:
        raise ValueError(role)
    if not result:
        raise ValueError(f"Role {role} selected no runs")
    return result


def resolve_runs(
    available: set[str],
    role: str,
    validation_run: str,
    evaluation_run: str,
    include_runs: str | None = None,
    exclude_runs: str | None = None,
) -> tuple[set[str], str]:
    """Resolve an auditable physical-run split without frame-level leakage."""
    if include_runs is None:
        result = selected_runs(available, role, validation_run, evaluation_run)
        source = "role_default"
    elif include_runs.strip().lower() == "all":
        result = set(available)
        source = "explicit_all"
    else:
        result = parse_csv_set(include_runs)
        source = "explicit_include"
    unknown = result - available
    if unknown:
        raise ValueError(f"Requested runs are absent: {sorted(unknown)}")
    excluded = set() if exclude_runs is None else parse_csv_set(exclude_runs)
    unknown_excluded = excluded - available
    if unknown_excluded:
        raise ValueError(f"Excluded runs are absent: {sorted(unknown_excluded)}")
    result -= excluded
    if not result:
        raise ValueError("Run selection is empty after exclusions")
    return result, source


def assign_scene_ids(rows: list[dict[str, object]]) -> dict[tuple[str, str], int]:
    sequences = sorted({(str(row["key"][0]), str(row["key"][2])) for row in rows})
    return {key: index + 1 for index, key in enumerate(sequences)}


def _window_diversity(rows: list[dict[str, object]]) -> tuple[int, float]:
    counts = Counter(str(row["bin_name"]) for row in rows)
    total = max(len(rows), 1)
    entropy = -sum(
        (count / total) * math.log(count / total) for count in counts.values()
    )
    return len(counts), entropy


def balanced_validation_subset(
    rows: list[dict[str, object]], frames_per_run: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Cap each run/camera using a diverse, genuinely consecutive window."""
    if frames_per_run < 1:
        return rows, []
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["key"][0]), str(row["key"][2]))].append(row)
    selected = []
    report = []
    for (run, camera), values in sorted(grouped.items()):
        values.sort(key=lambda row: int(row["key"][1]))
        if len(values) <= frames_per_run:
            chosen = values
            selection_strategy = "all_available"
        else:
            candidates = []
            for start in range(len(values) - frames_per_run + 1):
                window = values[start : start + frames_per_run]
                first = int(window[0]["key"][1])
                last = int(window[-1]["key"][1])
                if last - first != frames_per_run - 1:
                    continue
                diversity = _window_diversity(window)
                center_distance = abs(
                    (start + (frames_per_run - 1) / 2) - (len(values) - 1) / 2
                )
                candidates.append((diversity[0], diversity[1], -center_distance, window))
            if candidates:
                chosen = max(candidates, key=lambda item: item[:3])[3]
                selection_strategy = "single_consecutive_window"
            else:
                # Some cameras have several long valid stretches separated by
                # filtered frames. Fill the requested quota from the largest
                # complete stretches and, if necessary, one consecutive slice
                # of the next stretch. Dataset segment IDs still prevent any
                # temporal model window from crossing those gaps.
                segments = []
                start = 0
                for index in range(1, len(values) + 1):
                    boundary = (
                        index == len(values)
                        or int(values[index]["key"][1])
                        != int(values[index - 1]["key"][1]) + 1
                    )
                    if boundary:
                        segments.append(values[start:index])
                        start = index
                segments.sort(
                    key=lambda segment: (
                        -len(segment), int(segment[0]["key"][1])
                    )
                )
                chosen = []
                remaining = frames_per_run
                for segment in segments:
                    if remaining <= 0:
                        break
                    if len(segment) <= remaining:
                        chosen.extend(segment)
                        remaining -= len(segment)
                        continue
                    slices = []
                    for start in range(len(segment) - remaining + 1):
                        window = segment[start : start + remaining]
                        diversity = _window_diversity(window)
                        center_distance = abs(
                            (start + (remaining - 1) / 2)
                            - (len(segment) - 1) / 2
                        )
                        slices.append((
                            diversity[0], diversity[1], -center_distance, window
                        ))
                    chosen.extend(max(slices, key=lambda item: item[:3])[3])
                    remaining = 0
                if remaining or len(chosen) != frames_per_run:
                    raise RuntimeError(
                        f"Could not select {frames_per_run} frames from {run}/{camera}"
                    )
                chosen.sort(key=lambda row: int(row["key"][1]))
                selection_strategy = "largest_consecutive_segments"
        selected.extend(chosen)
        report.append({
            "source_run": run,
            "camera_id": camera,
            "available": len(values),
            "selected": len(chosen),
            "selection_strategy": selection_strategy,
            "first_source_frame": int(chosen[0]["key"][1]),
            "last_source_frame": int(chosen[-1]["key"][1]),
            "distance_bin_counts": dict(
                sorted(Counter(str(row["bin_name"]) for row in chosen).items())
            ),
        })
    selected.sort(key=lambda row: (str(row["key"][0]), str(row["key"][2]), int(row["key"][1])))
    return selected, report


def decode_instance_mask(path: Path, instance_id: int, shape: tuple[int, int]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    packed_rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint32)
    if packed_rgb.shape[:2] != shape:
        raise ValueError(f"Mask/image shape mismatch for {path}")
    packed = packed_rgb[..., 0] + (packed_rgb[..., 1] << 8) + (packed_rgb[..., 2] << 16)
    return packed == int(instance_id)


def png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def csv_bbox(row: dict[str, str]) -> list[int]:
    x0, y0 = int(float(row["xmin"])), int(float(row["ymin"]))
    x1, y1 = int(float(row["xmax"])), int(float(row["ymax"]))
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def parse_csv_set(value: str) -> set[str]:
    output = {part.strip() for part in value.split(",") if part.strip()}
    if not output:
        raise ValueError("A comma-separated selection cannot be empty")
    return output


def _remove_outputs(
    dataset_dir: Path, detection_path: Path, template_path: Path, overwrite: bool
) -> None:
    existing = [
        path for path in (dataset_dir, detection_path, template_path)
        if path.exists() or path.is_symlink()
    ]
    if existing and not overwrite:
        raise FileExistsError(f"Outputs exist: {existing}; pass --overwrite")
    if not overwrite:
        return
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    detection_path.unlink(missing_ok=True)
    if template_path.is_symlink() or template_path.is_file():
        template_path.unlink()
    elif template_path.is_dir():
        shutil.rmtree(template_path)


def _link_templates(output_root: Path, dataset_name: str, source_name: str) -> Path:
    source = output_root / "templates" / source_name
    destination = output_root / "templates" / dataset_name
    if not source.is_dir():
        raise FileNotFoundError(f"Template source is absent: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(
        os.path.relpath(source, destination.parent), target_is_directory=True
    )
    return destination


def _sequence_report(frame_map: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in frame_map:
        grouped[(str(row["source_run"]), str(row["camera_id"]))].append(
            int(row["source_frame"])
        )
    report = []
    for (run, camera), values in sorted(grouped.items()):
        frames = sorted(values)
        segments = 1 + sum(
            current - previous != 1
            for previous, current in zip(frames, frames[1:])
        )
        report.append({
            "source_run": run,
            "camera_id": camera,
            "frame_count": len(frames),
            "segment_count": segments,
            "first_source_frame": frames[0],
            "last_source_frame": frames[-1],
        })
    return report


def main() -> None:
    args = parser().parse_args()
    if args.minimum_mask_pixels < 1 or args.maximum_depth_m <= 0:
        raise ValueError("Mask pixel and depth limits must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be positive")
    dataset_name = args.dataset_name or ROLE_DATASET_NAMES[args.role]
    if dataset_name.startswith("assettocorsa"):
        raise ValueError(
            "Dataset name must not start with assettocorsa because GigaPose would "
            "select the shared legacy detection file"
        )
    if not args.assetto_export_root.is_dir():
        raise FileNotFoundError(args.assetto_export_root)
    index_skipped: list[dict[str, object]] = []
    index = load_index(
        args.assetto_export_root,
        skip_incomplete=True,
        skipped=index_skipped,
    )
    bins = parse_csv_set(args.distance_bins)
    cameras = parse_csv_set(args.cameras)
    available_bins = {str(item["bin_name"]) for item in index}
    unknown_bins = bins - available_bins
    if unknown_bins:
        raise ValueError(f"Unknown distance bins: {sorted(unknown_bins)}")
    available_runs = {str(item["key"][0]) for item in index}
    runs, run_selection_source = resolve_runs(
        available_runs,
        args.role,
        args.validation_run,
        args.evaluation_run,
        args.include_runs,
        args.exclude_runs,
    )
    selected = [
        item for item in index
        if str(item["bin_name"]) in bins
        and str(item["key"][0]) in runs
        and str(item["key"][2]) in cameras
    ]
    selected.sort(key=lambda item: (str(item["key"][0]), str(item["key"][2]), int(item["key"][1])))
    selected_before_role_balance = len(selected)
    validation_balance_report = []
    if args.role == "validation":
        selected, validation_balance_report = balanced_validation_subset(
            selected, args.validation_frames_per_run
        )
    if args.max_frames is not None:
        selected = selected[: args.max_frames]
    if not selected:
        raise RuntimeError("The requested role/bins/cameras selected no frames")
    scene_ids = assign_scene_ids(selected)

    dataset_dir = args.output_root / dataset_name
    split_dir = dataset_dir / "test"
    detection_path = args.output_root / "cnos-fastsam" / f"cnos-fastsam_{dataset_name}-test.json"
    template_path = args.output_root / "templates" / dataset_name
    _remove_outputs(dataset_dir, detection_path, template_path, args.overwrite)
    split_dir.mkdir(parents=True, exist_ok=True)
    detection_path.parent.mkdir(parents=True, exist_ok=True)

    alignment = build_aligned_mesh(
        args.cad_path,
        selected[0]["transform"],
        args.cad_scale,
        args.cad_axis_convention,
        args.fit_aabb,
    )
    write_models(alignment, dataset_dir, args.object_id)
    _link_templates(args.output_root, dataset_name, args.template_source_dataset)

    targets: list[dict[str, int]] = []
    detections: list[dict[str, object]] = []
    frame_map: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = list(index_skipped)
    writer = wds.ShardWriter(
        pattern=str(split_dir / "shard-%06d.tar"),
        maxcount=args.max_shard_size,
        encoder=False,
    )
    try:
        for item in selected:
            source_run, source_frame, camera_id = item["key"]
            scene_id = scene_ids[(str(source_run), str(camera_id))]
            manifest = item["manifest"]
            camera = item["camera"]
            transform = item["transform"]
            box = item["box"]
            image_path = item["bin_dir"] / manifest["image"]
            mask_path = item["bin_dir"] / manifest["masks"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            with Image.open(image_path) as image_handle:
                image_size = image_handle.size
            expected_size = (int(camera["image_width"]), int(camera["image_height"]))
            if image_size != expected_size:
                raise ValueError(f"Image/intrinsics size mismatch: {image_path}")
            if args.expected_cleaned_width and image_size[0] != args.expected_cleaned_width:
                raise ValueError(
                    f"Expected already shade-cleaned width {args.expected_cleaned_width}, "
                    f"found {image_size[0]} in {image_path}. Do not silently mix cropped "
                    "and uncropped images."
                )
            mask = decode_instance_mask(
                mask_path, int(box["instance_id"]), (image_size[1], image_size[0])
            )
            mask_pixels = int(mask.sum())
            if mask_pixels < args.minimum_mask_pixels:
                skipped.append({
                    "source_run": source_run,
                    "source_frame": int(source_frame),
                    "camera_id": camera_id,
                    "reason": "insufficient_exact_instance_mask",
                    "mask_pixels": mask_pixels,
                })
                continue
            validate_model_row(alignment, transform)
            pose = cad_to_camera_pose(transform, alignment.local_center)
            depth_m = float(pose[2, 3])
            if depth_m <= 0 or depth_m > args.maximum_depth_m:
                skipped.append({
                    "source_run": source_run,
                    "source_frame": int(source_frame),
                    "camera_id": camera_id,
                    "reason": "depth_out_of_range",
                    "depth_m": depth_m,
                })
                continue

            bbox = bbox_from_mask(mask)
            rle = pycoco_utils.binary_mask_to_rle(mask.astype(np.uint8))
            K = [
                float(camera["fx"]), 0.0, float(camera["cx"]),
                0.0, float(camera["fy"]), float(camera["cy"]),
                0.0, 0.0, 1.0,
            ]
            gt = [{
                "cam_R_m2c": pose[:3, :3].reshape(-1).tolist(),
                "cam_t_m2c": (pose[:3, 3] * 1000.0).tolist(),
                "obj_id": args.object_id,
            }]
            gt_info = [{
                "bbox_obj": csv_bbox(transform),
                "bbox_visib": bbox,
                "px_count_all": mask_pixels,
                "px_count_valid": mask_pixels,
                "px_count_visib": mask_pixels,
                "visib_fract": 1.0,
            }]
            key = f"{scene_id:06d}_{int(source_frame):06d}"
            writer.write({
                "__key__": key,
                "rgb.jpg": image_path.read_bytes(),
                "depth.png": png_bytes(np.zeros((image_size[1], image_size[0]), dtype=np.uint16)),
                "camera.json": json.dumps({"cam_K": K, "depth_scale": 1.0}).encode(),
                "gt.json": json.dumps(gt).encode(),
                "gt_info.json": json.dumps(gt_info).encode(),
                "mask_visib.json": json.dumps({"0": rle}).encode(),
            })
            targets.append({
                "scene_id": scene_id, "im_id": int(source_frame),
                "obj_id": args.object_id, "inst_count": 1,
            })
            detections.append({
                "scene_id": scene_id,
                "image_id": int(source_frame),
                "category_id": args.object_id,
                "score": 1.0,
                "bbox": bbox,
                "segmentation": rle,
                "time": 0.0,
                "opp_id": int(transform["opp_id"]),
                "instance_id": int(box["instance_id"]),
                "mask_provenance": "cleaned_assetto_packed_instance_mask",
            })
            timestamp_ms = int(manifest["sim_time_ms"])
            frame_map.append({
                "scene_id": scene_id,
                "im_id": int(source_frame),
                "source_run": source_run,
                "source_frame": int(source_frame),
                "camera_id": camera_id,
                "timestamp_ms": timestamp_ms,
                "sim_time_ms": timestamp_ms,
                "distance_bin": item["bin_name"],
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "cam_K": K,
                "instances": [{
                    "obj_id": args.object_id,
                    "opp_id": int(transform["opp_id"]),
                    "instance_id": int(box["instance_id"]),
                    "sam_object_id": int(box["instance_id"]),
                    "bbox": bbox,
                    "truncated": bool(int(transform["truncated"])),
                    "visible_pixels": mask_pixels,
                }],
            })
            if len(frame_map) % 250 == 0:
                print(f"prepared {len(frame_map)}/{len(selected)} frames", flush=True)
    finally:
        writer.close()

    if not frame_map:
        raise RuntimeError("No valid frames were written")
    (split_dir / "key_to_shard.json").write_text(
        json.dumps(key_to_shard_map(split_dir), indent=2)
    )
    (dataset_dir / "test_targets_bop19.json").write_text(json.dumps(targets, indent=2))
    (dataset_dir / "frame_map.json").write_text(json.dumps(frame_map, indent=2))
    detection_path.write_text(json.dumps(detections, indent=2))
    sequence_report = _sequence_report(frame_map)
    report = {
        "format": "rgb_self_recovery_gru_assetto_dataset_v1",
        "role": args.role,
        "dataset_name": dataset_name,
        "dataset_dir": str(dataset_dir),
        "assetto_export_root": str(args.assetto_export_root),
        "runs": sorted(runs),
        "run_selection_source": run_selection_source,
        "run_count": len(runs),
        "sequence_count": len(sequence_report),
        "segment_count": sum(int(row["segment_count"]) for row in sequence_report),
        "selected_before_validation": len(selected),
        "selected_before_role_balance": selected_before_role_balance,
        "frames_written": len(frame_map),
        "skipped_count": len(skipped),
        "skipped_reason_counts": dict(Counter(row["reason"] for row in skipped)),
        "skipped": skipped,
        "sequences": sequence_report,
        "distance_bins": sorted(bins),
        "cameras": sorted(cameras),
        "minimum_mask_pixels": args.minimum_mask_pixels,
        "maximum_depth_m": args.maximum_depth_m,
        "shade_regions_already_removed": True,
        "expected_cleaned_width": args.expected_cleaned_width,
        "validation_runs_excluded_from_training": (
            [] if run_selection_source != "role_default" or args.role == "benchmark"
            else [args.validation_run, args.evaluation_run]
        ),
        "validation_frames_per_run": args.validation_frames_per_run,
        "validation_balance": validation_balance_report,
        "template_path": str(template_path),
        "detection_path": str(detection_path),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (dataset_dir / "evaluation_metadata.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
