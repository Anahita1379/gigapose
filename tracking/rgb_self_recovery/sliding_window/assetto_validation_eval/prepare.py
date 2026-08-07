"""Reassemble distance-binned Assetto validation runs as GigaPose datasets."""

from __future__ import annotations

import argparse
import io
import json
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


DEFAULT_EXPORT_ROOT = Path(
    "/mnt/ssd2tb/.local_share_backup/Steam/steamapps/common/assettocorsa/"
    "apps/lua/multi_cam_obs/frames/Assettocorsa_new_dataset_copy"
)
DEFAULT_CAD = Path("gigaPose_datasets/datasets/racecar/models/obj_000001.ply")
DEFAULT_RUNS = {
    "rgb_recovery_val_putnam_rear": "20260730_putnam_rain_1opp_2lap_farRear",
    "rgb_recovery_val_laguna_front": (
        "20260730_laguna2026_fog_1opp_2laps_mostlyFront"
    ),
}
CAMERA_SCENE_IDS = {"front": 1, "rear": 2, "stereo_left": 3, "stereo_right": 4}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--assetto-export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    value.add_argument(
        "--run",
        action="append",
        help=(
            "Repeat DATASET_NAME=SOURCE_RUN. If omitted, prepare the held-out "
            "Putnam-rear and Laguna-front runs."
        ),
    )
    value.add_argument(
        "--output-root", type=Path, default=Path("gigaPose_datasets/datasets")
    )
    value.add_argument("--cad-path", type=Path, default=DEFAULT_CAD)
    value.add_argument("--object-id", type=int, default=1)
    value.add_argument("--cad-scale", type=float, default=1.0)
    value.add_argument(
        "--cad-axis-convention",
        choices=("x-forward-z-up", "csp"),
        default="x-forward-z-up",
    )
    value.add_argument(
        "--fit-aabb", choices=("none", "uniform", "nonuniform"), default="nonuniform"
    )
    value.add_argument("--max-shard-size", type=int, default=500)
    value.add_argument(
        "--template-source-dataset",
        default="assettocorsa_new_dataset",
        help="Existing aligned template folder to link for each new dataset.",
    )
    value.add_argument("--overwrite", action="store_true")
    return value


def parse_run_specs(values: list[str] | None) -> dict[str, str]:
    if not values:
        return dict(DEFAULT_RUNS)
    output: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--run must use DATASET_NAME=SOURCE_RUN: {value}")
        dataset_name, source_run = (part.strip() for part in value.split("=", 1))
        if not dataset_name or not source_run or dataset_name in output:
            raise ValueError(f"Invalid or duplicate --run specification: {value}")
        if dataset_name.startswith("assettocorsa"):
            raise ValueError(
                "Dataset names must not start with 'assettocorsa': GigaPose would "
                "select its shared legacy detection file instead of this dataset's file."
            )
        output[dataset_name] = source_run
    return output


def decode_packed_mask(path: Path, instance_id: int, shape: tuple[int, int]) -> np.ndarray:
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


def _remove_existing(
    dataset_dir: Path, detection_path: Path, template_path: Path, overwrite: bool
) -> None:
    existing = [path for path in (dataset_dir, detection_path, template_path) if path.exists() or path.is_symlink()]
    if existing and not overwrite:
        raise FileExistsError(f"Evaluation output exists: {existing}; pass --overwrite")
    if not overwrite:
        return
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    detection_path.unlink(missing_ok=True)
    if template_path.is_symlink() or template_path.is_file():
        template_path.unlink()
    elif template_path.is_dir():
        shutil.rmtree(template_path)


def _template_link(output_root: Path, dataset_name: str, source_name: str) -> Path:
    template_root = output_root / "templates"
    source = template_root / source_name
    destination = template_root / dataset_name
    if not source.is_dir():
        raise FileNotFoundError(
            f"Template source is absent: {source}. Render or choose aligned templates."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source, destination.parent), target_is_directory=True)
    return destination


def prepare_run(
    args: argparse.Namespace,
    dataset_name: str,
    source_run: str,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    selected = [item for item in rows if item["key"][0] == source_run]
    if not selected:
        raise ValueError(f"No exported frames found for source run {source_run}")
    selected.sort(key=lambda item: (int(item["manifest"]["sim_time_ms"]), item["key"][1]))
    cameras = sorted({item["key"][2] for item in selected})
    if len(cameras) != 1:
        raise ValueError(
            f"Development dataset expects one camera per run; {source_run} has {cameras}"
        )
    camera_id = cameras[0]
    scene_id = CAMERA_SCENE_IDS[camera_id]
    frame_ids = [int(item["key"][1]) for item in selected]
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError(f"Duplicate frame IDs in {source_run}")

    dataset_dir = args.output_root / dataset_name
    split_dir = dataset_dir / "test"
    detection_path = (
        args.output_root / "cnos-fastsam" / f"cnos-fastsam_{dataset_name}-test.json"
    )
    template_path = args.output_root / "templates" / dataset_name
    _remove_existing(dataset_dir, detection_path, template_path, args.overwrite)
    split_dir.mkdir(parents=True, exist_ok=True)
    detection_path.parent.mkdir(parents=True, exist_ok=True)

    reference = selected[0]["transform"]
    alignment = build_aligned_mesh(
        args.cad_path,
        reference,
        args.cad_scale,
        args.cad_axis_convention,
        args.fit_aabb,
    )
    write_models(alignment, dataset_dir, args.object_id)
    _template_link(args.output_root, dataset_name, args.template_source_dataset)

    targets: list[dict[str, int]] = []
    detections: list[dict[str, object]] = []
    frame_map: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    writer = wds.ShardWriter(
        pattern=str(split_dir / "shard-%06d.tar"),
        maxcount=args.max_shard_size,
        encoder=False,
    )
    try:
        for item in selected:
            frame_id = int(item["key"][1])
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
                raise ValueError(
                    f"Image size mismatch for {image_path}: {image_size} != {expected_size}"
                )
            mask = decode_packed_mask(
                mask_path, int(box["instance_id"]), (image_size[1], image_size[0])
            )
            if not mask.any():
                skipped.append({"frame": frame_id, "reason": "empty packed instance mask"})
                continue
            validate_model_row(alignment, transform)
            pose = cad_to_camera_pose(transform, alignment.local_center)
            if pose[2, 3] <= 0:
                skipped.append({"frame": frame_id, "reason": "non-positive camera depth"})
                continue

            visible_bbox = bbox_from_mask(mask)
            rle = pycoco_utils.binary_mask_to_rle(mask.astype(np.uint8))
            gt = [{
                "cam_R_m2c": pose[:3, :3].reshape(-1).tolist(),
                "cam_t_m2c": (pose[:3, 3] * 1000.0).tolist(),
                "obj_id": args.object_id,
            }]
            gt_info = [{
                "bbox_obj": csv_bbox(transform),
                "bbox_visib": visible_bbox,
                "px_count_all": int(mask.sum()),
                "px_count_valid": int(mask.sum()),
                "px_count_visib": int(mask.sum()),
                "visib_fract": 1.0,
            }]
            K = [
                float(camera["fx"]), 0.0, float(camera["cx"]),
                0.0, float(camera["fy"]), float(camera["cy"]),
                0.0, 0.0, 1.0,
            ]
            key = f"{scene_id:06d}_{frame_id:06d}"
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
                "scene_id": scene_id,
                "im_id": frame_id,
                "obj_id": args.object_id,
                "inst_count": 1,
            })
            detections.append({
                "scene_id": scene_id,
                "image_id": frame_id,
                "category_id": args.object_id,
                "score": 1.0,
                "bbox": visible_bbox,
                "segmentation": rle,
                "time": 0.0,
                "opp_id": int(transform["opp_id"]),
                "instance_id": int(box["instance_id"]),
                "mask_provenance": "held_out_simulator_instance_mask",
            })
            timestamp_ms = int(manifest["sim_time_ms"])
            frame_map.append({
                "scene_id": scene_id,
                "im_id": frame_id,
                "source_run": source_run,
                "source_frame": frame_id,
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
                    "bbox": visible_bbox,
                    "truncated": bool(int(transform["truncated"])),
                    "visible_pixels": int(mask.sum()),
                }],
            })
    finally:
        writer.close()

    if not targets:
        raise RuntimeError(f"No valid frames were written for {source_run}")
    (split_dir / "key_to_shard.json").write_text(
        json.dumps(key_to_shard_map(split_dir), indent=2)
    )
    (dataset_dir / "test_targets_bop19.json").write_text(json.dumps(targets, indent=2))
    (dataset_dir / "frame_map.json").write_text(json.dumps(frame_map, indent=2))
    detection_path.write_text(json.dumps(detections, indent=2))
    report = {
        "format": "rgb_self_recovery_assetto_validation_eval_v1",
        "development_evaluation_only": True,
        "source_run": source_run,
        "dataset_name": dataset_name,
        "dataset_dir": str(dataset_dir),
        "detection_path": str(detection_path),
        "camera_id": camera_id,
        "scene_id": scene_id,
        "source_rows": len(selected),
        "frames_written": len(targets),
        "skipped": skipped,
        "first_source_frame": frame_map[0]["source_frame"],
        "last_source_frame": frame_map[-1]["source_frame"],
        "template_source_dataset": args.template_source_dataset,
        "template_link": str(template_path),
        "mask_provenance": "held_out_simulator_instance_mask",
        "evaluation_warning": (
            "These two runs were used for checkpoint selection. Results are development-set "
            "diagnostics, not an unbiased benchmark."
        ),
        "cad_axis_convention": args.cad_axis_convention,
        "fit_aabb": args.fit_aabb,
    }
    (dataset_dir / "evaluation_metadata.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    args = parser().parse_args()
    specs = parse_run_specs(args.run)
    rows = load_index(args.assetto_export_root)
    reports = [
        prepare_run(args, dataset_name, source_run, rows)
        for dataset_name, source_run in specs.items()
    ]
    print(json.dumps({"prepared": reports}, indent=2))


if __name__ == "__main__":
    main()
