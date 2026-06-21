"""Convert an instance-mask multi_cam_obs recording for GigaPose inference.

The corrected recorder stores one JPEG and one lossless instance-ID PNG per
camera/frame, plus one row per opponent in ``csv/bboxes_3d.csv``. The exact
``instance_id`` and RGB encoding are read from the CSV. Fully occluded instances
with no mask pixels are skipped.

Example:

    python -m src.scripts.prepare_AC_metadata_dataset \
      --source-root /path/to/20260620_haze_3opp_withInstanceMask \
      --cad-path /path/to/opponent_car.ply \
      --cameras front --max-frames 100
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
import webdataset as wds
from bop_toolkit_lib import pycoco_utils
from PIL import Image

from fine_tuning.ac_geometry import (
    instance_mask_for_row,
    instance_mask_path,
    load_instance_ids,
)


DEFAULT_SOURCE_ROOT = Path(
    "/mnt/ssd2tb/.local_share_backup/Steam/steamapps/common/assettocorsa/"
    "apps/lua/multi_cam_obs/frames/20260620_haze_3opp_withInstanceMask"
)
CAMERA_SCENE_IDS = {
    "front": 1,
    "rear": 2,
    "stereo_left": 3,
    "stereo_right": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare an Assetto Corsa multi_cam_obs recording for GigaPose."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--cad-path",
        type=Path,
        required=True,
        help=(
            "CAD mesh for the opponent car. All selected opponents are treated as "
            "instances of this one object model."
        ),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("gigaPose_datasets/datasets")
    )
    parser.add_argument("--dataset-name", default="assettocorsa")
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument(
        "--cameras",
        default="front",
        help="Comma-separated camera IDs, or 'all'. Default: front.",
    )
    parser.add_argument(
        "--opponent-ids",
        default=None,
        help="Optional comma-separated opponent IDs to include, e.g. 1,2.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of synchronized source frame IDs (before camera expansion).",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--exclude-truncated",
        action="store_true",
        help="Drop boxes clipped by the image boundary or near plane.",
    )
    parser.add_argument(
        "--cad-scale",
        type=float,
        default=1.0,
        help="Scale applied to CAD vertices before writing the GigaPose model.",
    )
    parser.add_argument(
        "--center-cad",
        dest="center_cad",
        action="store_true",
        help="Translate the CAD bounding-box center to the origin (the default).",
    )
    parser.add_argument(
        "--no-center-cad",
        dest="center_cad",
        action="store_false",
        help="Keep the source CAD origin. Use only with templates rendered that way.",
    )
    parser.add_argument(
        "--max-shard-size", type=int, default=1000, help="Images per WebDataset shard."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the test split/detection JSON while preserving training splits.",
    )
    parser.add_argument(
        "--detection-file-name",
        default="cnos-fastsam_assettocorsa-test.json",
    )
    parser.set_defaults(center_cad=True)
    return parser.parse_args()


def parse_csv_ints(value: str | None) -> set[int] | None:
    if value is None:
        return None
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def parse_cameras(value: str) -> list[str]:
    cameras = list(CAMERA_SCENE_IDS) if value.strip().lower() == "all" else [
        part.strip() for part in value.split(",") if part.strip()
    ]
    unknown = sorted(set(cameras) - set(CAMERA_SCENE_IDS))
    if unknown:
        raise ValueError(
            f"Unknown camera(s): {', '.join(unknown)}. "
            f"Available: {', '.join(CAMERA_SCENE_IDS)}"
        )
    if not cameras:
        raise ValueError("At least one camera must be selected.")
    return cameras


def read_annotations(
    csv_path: Path,
    cameras: list[str],
    opponent_ids: set[int] | None,
    exclude_truncated: bool,
) -> dict[tuple[str, int], list[dict[str, str]]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {
            "frame", "sim_time_ms", "camera_id", "opp_id", "instance_id",
            "mask_r", "mask_g", "mask_b", "image_width", "image_height",
            "fx", "fy", "cx", "cy", "xmin", "ymin", "xmax", "ymax",
            "truncated",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        for row in reader:
            if row["camera_id"] not in cameras:
                continue
            if opponent_ids is not None and int(row["opp_id"]) not in opponent_ids:
                continue
            if exclude_truncated and int(row["truncated"]):
                continue
            grouped[(row["camera_id"], int(row["frame"]))].append(row)
    return grouped


def select_frames(
    grouped: dict[tuple[str, int], list[dict[str, str]]],
    frame_stride: int,
    max_frames: int | None,
) -> dict[tuple[str, int], list[dict[str, str]]]:
    if frame_stride < 1:
        raise ValueError("--frame-stride must be at least 1.")
    frame_ids = sorted({frame for _, frame in grouped})[::frame_stride]
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError("--max-frames must be positive.")
        frame_ids = frame_ids[:max_frames]
    selected = set(frame_ids)
    return {key: rows for key, rows in grouped.items() if key[1] in selected}


def row_image_stem(row: dict[str, str]) -> str:
    return f"{int(row['sim_time_ms']):010d}_{int(row['frame']):06d}"


def bbox_xywh(row: dict[str, str]) -> list[int]:
    x0, y0 = int(row["xmin"]), int(row["ymin"])
    x1, y1 = int(row["xmax"]), int(row["ymax"])
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def bbox_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def mesh_info(mesh: trimesh.Trimesh) -> dict[str, float]:
    bounds = np.asarray(mesh.bounds)
    size = bounds[1] - bounds[0]
    return {
        "diameter": float(np.linalg.norm(size)),
        "min_x": float(bounds[0, 0]),
        "min_y": float(bounds[0, 1]),
        "min_z": float(bounds[0, 2]),
        "size_x": float(size[0]),
        "size_y": float(size[1]),
        "size_z": float(size[2]),
    }


def write_model(
    cad_path: Path, model_dir: Path, object_id: int, scale: float, center: bool
) -> None:
    if not cad_path.is_file():
        raise FileNotFoundError(f"CAD mesh not found: {cad_path}")
    if scale <= 0:
        raise ValueError("--cad-scale must be positive.")
    mesh = trimesh.load(cad_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError(f"Could not load a non-empty mesh from {cad_path}")
    mesh = mesh.copy()
    mesh.apply_scale(scale)
    if center:
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
    model_dir.mkdir(parents=True, exist_ok=True)
    mesh.export(model_dir / f"obj_{object_id:06d}.ply")
    mesh.export(model_dir / f"obj_{object_id:06d}.obj")
    (model_dir / "models_info.json").write_text(
        json.dumps({str(object_id): mesh_info(mesh)}, indent=2)
    )


def key_to_shard_map(test_dir: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for shard_path in sorted(test_dir.glob("shard-*.tar")):
        shard_id = int(re.findall(r"\d+", shard_path.name)[0])
        with tarfile.open(shard_path) as tar:
            for key in {name.split(".")[0] for name in tar.getnames()}:
                mapping[key] = shard_id
    return mapping


def validate_group(rows: list[dict[str, str]], image: Image.Image, image_path: Path) -> None:
    width, height = image.size
    for row in rows:
        annotated_size = (int(row["image_width"]), int(row["image_height"]))
        if annotated_size != image.size:
            raise ValueError(
                f"Image size mismatch for {image_path}: {image.size} vs {annotated_size}"
            )
        x, y, box_width, box_height = bbox_xywh(row)
        if x < 0 or y < 0 or x + box_width > width or y + box_height > height:
            raise ValueError(f"Out-of-range bbox in {image_path}: {bbox_xywh(row)}")
    calibration_fields = ("fx", "fy", "cx", "cy")
    first = tuple(rows[0][name] for name in calibration_fields)
    if any(tuple(row[name] for name in calibration_fields) != first for row in rows[1:]):
        raise ValueError(f"Inconsistent intrinsics for {image_path}")


def prepare_output(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    dataset_dir = args.output_root / args.dataset_name
    test_dir = dataset_dir / "test"
    detections_dir = args.output_root / "cnos-fastsam"
    detection_path = detections_dir / args.detection_file_name
    if test_dir.exists() or detection_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output exists ({test_dir} or {detection_path}); pass --overwrite."
            )
        # Preserve train_pbr_web and val_pbr_web when adding/replacing inference.
        if test_dir.exists():
            shutil.rmtree(test_dir)
        detection_path.unlink(missing_ok=True)
        for metadata_name in ("test_targets_bop19.json", "frame_map.json"):
            (dataset_dir / metadata_name).unlink(missing_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    detections_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir, test_dir, detection_path


def main() -> None:
    args = parse_args()
    cameras = parse_cameras(args.cameras)
    csv_path = args.source_root / "csv" / "bboxes_3d.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Annotation CSV not found: {csv_path}")

    grouped = read_annotations(
        csv_path,
        cameras,
        parse_csv_ints(args.opponent_ids),
        args.exclude_truncated,
    )
    grouped = select_frames(grouped, args.frame_stride, args.max_frames)
    if not grouped:
        raise RuntimeError("No annotated images matched the requested filters.")

    dataset_dir, test_dir, detection_path = prepare_output(args)
    write_model(
        args.cad_path,
        dataset_dir / "models",
        args.object_id,
        args.cad_scale,
        args.center_cad,
    )

    targets: list[dict[str, int]] = []
    detections: list[dict[str, object]] = []
    frame_map: list[dict[str, object]] = []
    writer = wds.ShardWriter(
        pattern=str(test_dir / "shard-%06d.tar"),
        maxcount=args.max_shard_size,
        encoder=False,
    )
    try:
        for (camera_id, frame), rows in sorted(
            grouped.items(), key=lambda item: (CAMERA_SCENE_IDS[item[0][0]], item[0][1])
        ):
            scene_id = CAMERA_SCENE_IDS[camera_id]
            stem = row_image_stem(rows[0])
            image_path = args.source_root / "images" / camera_id / f"{stem}.jpg"
            if not image_path.is_file():
                raise FileNotFoundError(f"Image referenced by CSV is missing: {image_path}")
            image = Image.open(image_path).convert("RGB")
            validate_group(rows, image, image_path)
            width, height = image.size
            mask_path = instance_mask_path(args.source_root, camera_id, rows[0])
            instance_ids = load_instance_ids(mask_path, image.size)
            instance_data = []
            for row in sorted(rows, key=lambda item: int(item["opp_id"])):
                mask = instance_mask_for_row(instance_ids, row)
                if not mask.any():
                    continue
                instance_data.append((row, mask.astype(np.uint8), bbox_from_mask(mask)))
            if not instance_data:
                continue

            first = rows[0]
            camera = {
                "cam_K": [
                    float(first["fx"]), 0.0, float(first["cx"]),
                    0.0, float(first["fy"]), float(first["cy"]),
                    0.0, 0.0, 1.0,
                ],
                "depth_scale": 1.0,
            }
            key = f"{scene_id:06d}_{frame:06d}"
            zero_depth = Image.fromarray(np.zeros((height, width), dtype=np.uint16))
            writer.write(
                {
                    "__key__": key,
                    "rgb.jpg": image_path.read_bytes(),
                    "depth.png": png_bytes(zero_depth),
                    "camera.json": json.dumps(camera).encode(),
                }
            )

            targets.append(
                {
                    "scene_id": scene_id,
                    "im_id": frame,
                    "obj_id": args.object_id,
                    "inst_count": len(instance_data),
                }
            )
            instance_records = []
            for row, mask, bbox in instance_data:
                detections.append(
                    {
                        "scene_id": scene_id,
                        "image_id": frame,
                        "category_id": args.object_id,
                        "score": 1.0,
                        "bbox": bbox,
                        "segmentation": pycoco_utils.binary_mask_to_rle(mask),
                        "time": 0.0,
                        "opp_id": int(row["opp_id"]),
                        "instance_id": int(row["instance_id"]),
                        "truncated": int(row["truncated"]),
                    }
                )
                instance_records.append(
                    {
                        "opp_id": int(row["opp_id"]),
                        "instance_id": int(row["instance_id"]),
                        "bbox": bbox,
                        "bbox_3d_projection": bbox_xywh(row),
                        "visible_pixels": int(mask.sum()),
                        "truncated": bool(int(row["truncated"])),
                    }
                )
            frame_map.append(
                {
                    "scene_id": scene_id,
                    "im_id": frame,
                    "camera_id": camera_id,
                    "source_frame": frame,
                    "sim_time_ms": int(first["sim_time_ms"]),
                    "image_path": str(image_path),
                    "instance_mask_path": str(mask_path),
                    "instances": instance_records,
                }
            )
    finally:
        writer.close()

    (test_dir / "key_to_shard.json").write_text(
        json.dumps(key_to_shard_map(test_dir), indent=2)
    )
    (dataset_dir / "test_targets_bop19.json").write_text(
        json.dumps(targets, indent=2)
    )
    (dataset_dir / "frame_map.json").write_text(json.dumps(frame_map, indent=2))
    detection_path.write_text(json.dumps(detections, indent=2))

    print(f"Prepared {len(targets)} annotated camera frames ({len(detections)} boxes).")
    print(f"Dataset: {dataset_dir}")
    print(f"Detections: {detection_path}")
    print("Detection RLEs and boxes came from the visible per-instance PNG masks.")


if __name__ == "__main__":
    main()
