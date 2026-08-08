#!/usr/bin/env python3
"""Prepare a real Grounded-SAM folder for GigaPose inference.

This handles folders like:

    .../frames/2026-05-26-12-19-49/front/
      images/
      images_jpg_new/
      metadata/
      Grounded_Sam_v4/
        metadata.json
        masks_png/
        masks_npy/

It writes a GigaPose/BOP-style test WebDataset plus a CNOS/FastSAM-style
detection JSON using the Grounded-SAM masks.

Example:

    python -m Assetto_data_prep.prepare_grounded_sam_inference \
      --source-root /path/to/frames/2026-05-26-12-19-49/front \
      --cad-path gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
      --dataset-name real_20260526_front_gsam_v4 \
      --overwrite
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import webdataset as wds
from bop_toolkit_lib import pycoco_utils
from PIL import Image

from Assetto_data_prep.common import bbox_from_mask, key_to_shard_map, png_bytes


DEFAULT_CAD = Path("gigaPose_datasets/datasets/racecar/models/obj_000001.ply")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare real Grounded-SAM masks for GigaPose inference."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        action="append",
        required=True,
        help="Camera folder, e.g. .../frames/<session>/front. Repeatable.",
    )
    parser.add_argument("--cad-path", type=Path, default=DEFAULT_CAD)
    parser.add_argument("--output-root", type=Path, default=Path("gigaPose_datasets/datasets"))
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--grounded-sam-dir", default="Grounded_Sam_v4")
    parser.add_argument("--grounded-sam-metadata", default="metadata.json")
    parser.add_argument("--sample-metadata-dir", default="metadata")
    parser.add_argument(
        "--image-dirs",
        default="images_jpg_new,images,images_jpg",
        help="Comma-separated image directory search order under each source root.",
    )
    parser.add_argument(
        "--camera-k",
        default=None,
        help=(
            "Fallback camera intrinsics as 'fx,fy,cx,cy' if per-sample YAML "
            "metadata is missing."
        ),
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-session", type=int, default=None)
    parser.add_argument("--min-mask-pixels", type=int, default=25)
    parser.add_argument("--score", type=float, default=1.0)
    parser.add_argument("--max-shard-size", type=int, default=1000)
    parser.add_argument(
        "--single-instance-policy",
        choices=("all", "epnp_bbox", "closest_principal_point"),
        default="all",
        help=(
            "How to handle frames containing several Grounded-SAM objects. "
            "The default preserves all detections. epnp_bbox keeps the mask "
            "best matching detector_bbox_xywh in a reference label and falls "
            "back to closest_principal_point when no label exists."
        ),
    )
    parser.add_argument(
        "--reference-label-dir",
        default="EPnPv2_gt_mesh_z_hybrid_labels",
        help=(
            "Directory below each --source-root containing reference JSONs "
            "used by --single-instance-policy epnp_bbox."
        ),
    )
    parser.add_argument(
        "--reference-label-policy",
        choices=("all", "quality_approved", "quality_approved_unique"),
        default="all",
        help=(
            "Which EPnP detector boxes may identify the target. Quality "
            "approval requires label_weight>=0.5, center_error_px<=25, and "
            "projected_bbox_iou>=0.25. The unique mode skips ambiguous frames."
        ),
    )
    parser.add_argument(
        "--missing-reference-policy",
        choices=("closest_principal_point", "skip", "error"),
        default="closest_principal_point",
        help=(
            "Behavior for an epnp_bbox frame with no usable reference. "
            "Use skip for calibration-target datasets."
        ),
    )
    parser.add_argument(
        "--detection-file-name",
        default=None,
        help="Defaults to cnos-fastsam_<dataset-name>-test.json.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_camera_k(value: str | None) -> np.ndarray | None:
    if value is None:
        return None
    vals = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(vals) != 4:
        raise ValueError("--camera-k must be formatted as fx,fy,cx,cy")
    fx, fy, cx, cy = vals
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "This script needs PyYAML to read per-frame metadata. Install with "
            "`pip install pyyaml`, or pass --camera-k and remove/use no metadata."
        ) from exc
    with path.open() as f:
        data = yaml.safe_load(f)
    return data or {}


def camera_k_from_yaml(path: Path) -> np.ndarray:
    data = load_yaml(path)
    intr = data.get("camera_intrinsics", {})
    if "k" not in intr:
        raise ValueError(f"{path} does not contain camera_intrinsics.k")
    return np.asarray(intr["k"], dtype=float).reshape(3, 3)


def timestamp_from_frame_file(frame_file: str) -> str | None:
    match = re.search(r"image_(\d+)", frame_file)
    return match.group(1) if match else None


def find_image(source_root: Path, frame_file: str, image_dirs: list[str]) -> Path | None:
    frame_path = Path(frame_file)
    names = [frame_path.name]
    if frame_path.suffix.lower() == ".jpg":
        names.append(frame_path.with_suffix(".png").name)
    elif frame_path.suffix.lower() == ".png":
        names.append(frame_path.with_suffix(".jpg").name)
    for image_dir in image_dirs:
        for name in names:
            path = source_root / image_dir / name
            if path.is_file():
                return path
    return None


def find_sample_yaml(source_root: Path, metadata_dir: str, frame_file: str) -> Path | None:
    timestamp = timestamp_from_frame_file(frame_file)
    if timestamp is None:
        return None
    candidates = sorted((source_root / metadata_dir).glob(f"sample_{timestamp}_*.yaml"))
    return candidates[0] if candidates else None


def xyxy_to_xywh(bbox_xyxy: list[Any] | None) -> list[int] | None:
    if bbox_xyxy is None:
        return None
    x0, y0, x1, y1 = [int(round(float(v))) for v in bbox_xyxy]
    return [x0, y0, max(0, x1 - x0 + 1), max(0, y1 - y0 + 1)]


def bbox_iou_xywh(a: list[Any], b: list[Any]) -> float:
    ax, ay, aw, ah = (float(value) for value in a)
    bx, by, bw, bh = (float(value) for value in b)
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = max(0.0, aw * ah) + max(0.0, bw * bh) - intersection
    return intersection / union if union > 0.0 else 0.0


def bbox_center_distance_sq(a: list[Any], b: list[Any]) -> float:
    ax, ay, aw, ah = (float(value) for value in a)
    bx, by, bw, bh = (float(value) for value in b)
    return ((ax + 0.5 * aw) - (bx + 0.5 * bw)) ** 2 + (
        (ay + 0.5 * ah) - (by + 0.5 * bh)
    ) ** 2


def load_reference_bboxes(
    root: Path, policy: str = "all"
) -> dict[str, list[list[float]]]:
    """Index detector boxes by source-image timestamp without using pose values."""

    indexed: dict[str, list[list[float]]] = {}
    if not root.is_dir():
        return indexed
    for path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        records = data if isinstance(data, list) else [data]
        for record in records:
            if not isinstance(record, dict) or record.get("detector_bbox_xywh") is None:
                continue
            if policy != "all" and not (
                float(record.get("label_weight", 1.0)) >= 0.5
                and float(record.get("center_error_px", float("inf"))) <= 25.0
                and float(record.get("projected_bbox_iou", 0.0)) >= 0.25
            ):
                continue
            source_name = Path(str(record.get("source_image_path", ""))).name
            timestamp = timestamp_from_frame_file(source_name)
            if timestamp is None:
                match = re.match(r"(\d+)(?:_\d+)?$", path.stem)
                timestamp = match.group(1) if match else None
            if timestamp is None:
                continue
            bbox = [float(value) for value in record["detector_bbox_xywh"]]
            if len(bbox) == 4 and bbox[2] > 0.0 and bbox[3] > 0.0:
                indexed.setdefault(timestamp, []).append(bbox)
    return indexed


def select_single_instance(
    candidates: list[dict[str, Any]],
    policy: str,
    K: np.ndarray,
    reference_bboxes: list[list[float]],
) -> tuple[list[dict[str, Any]], str]:
    if len(candidates) <= 1 or policy == "all":
        return candidates, "all" if policy == "all" else "single_available"

    if policy == "epnp_bbox" and reference_bboxes:
        # Maximize overlap first. Center distance resolves non-overlap and ties.
        chosen = min(
            candidates,
            key=lambda candidate: min(
                (
                    -bbox_iou_xywh(candidate["bbox"], reference),
                    bbox_center_distance_sq(candidate["bbox"], reference),
                )
                for reference in reference_bboxes
            ),
        )
        return [chosen], "epnp_bbox"

    cx, cy = float(K[0, 2]), float(K[1, 2])
    chosen = min(
        candidates,
        key=lambda candidate: (
            (float(candidate["bbox"][0]) + 0.5 * float(candidate["bbox"][2]) - cx) ** 2
            + (float(candidate["bbox"][1]) + 0.5 * float(candidate["bbox"][3]) - cy) ** 2
        ),
    )
    reason = "closest_principal_point"
    if policy == "epnp_bbox":
        reason = "closest_principal_point_no_reference"
    return [chosen], reason


def image_bytes(path: Path) -> bytes:
    # Normalize to RGB JPEG bytes regardless of source extension/mode.
    with Image.open(path) as image:
        image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()


def load_mask(path: Path, expected_size: tuple[int, int]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    if (mask.shape[1], mask.shape[0]) != expected_size:
        raise ValueError(f"Mask size mismatch: {path}")
    return mask


def write_models_from_cad(cad_path: Path, dataset_dir: Path, object_id: int) -> None:
    import trimesh

    mesh = trimesh.load(cad_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError(f"Could not load CAD mesh: {cad_path}")
    models_dir = dataset_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    mesh.export(models_dir / f"obj_{object_id:06d}.ply")
    mesh.export(models_dir / f"obj_{object_id:06d}.obj")

    bounds = np.asarray(mesh.bounds, dtype=float)
    size = bounds[1] - bounds[0]
    info = {
        "diameter": float(np.linalg.norm(size)),
        "min_x": float(bounds[0, 0]),
        "min_y": float(bounds[0, 1]),
        "min_z": float(bounds[0, 2]),
        "size_x": float(size[0]),
        "size_y": float(size[1]),
        "size_z": float(size[2]),
    }
    (models_dir / "models_info.json").write_text(json.dumps({str(object_id): info}, indent=2))
    (dataset_dir / "models_info.json").write_text(json.dumps([{"obj_id": object_id}], indent=2))


def load_grounded_sam_metadata(source_root: Path, grounded_sam_dir: str, metadata_name: str) -> list[dict[str, Any]]:
    path = source_root / grounded_sam_dir / metadata_name
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        rows = list(data.values())
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(f"Unsupported Grounded-SAM metadata format in {path}")
    return sorted(rows, key=lambda row: int(row.get("frame_index", 0)))


def main() -> None:
    args = parse_args()
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if args.min_mask_pixels < 0:
        raise ValueError("--min-mask-pixels must be non-negative")

    image_dirs = [item.strip() for item in args.image_dirs.split(",") if item.strip()]
    fallback_K = parse_camera_k(args.camera_k)
    detection_file_name = args.detection_file_name or f"cnos-fastsam_{args.dataset_name}-test.json"

    dataset_dir = args.output_root / args.dataset_name
    test_dir = dataset_dir / "test"
    detection_dir = args.output_root / "cnos-fastsam"
    detection_path = detection_dir / detection_file_name

    existing = [path for path in (test_dir, detection_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output exists: {existing}; pass --overwrite")
    if args.overwrite:
        if test_dir.exists():
            shutil.rmtree(test_dir)
        detection_path.unlink(missing_ok=True)
        for name in ("test_targets_bop19.json", "frame_map.json", "inference_metadata.json"):
            (dataset_dir / name).unlink(missing_ok=True)

    test_dir.mkdir(parents=True, exist_ok=True)
    detection_dir.mkdir(parents=True, exist_ok=True)
    write_models_from_cad(args.cad_path, dataset_dir, args.object_id)

    detections: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    frame_map: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    instance_selection_counts: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    def record_selection(reason: str) -> None:
        instance_selection_counts[reason] = instance_selection_counts.get(reason, 0) + 1

    writer = wds.ShardWriter(
        pattern=str(test_dir / "shard-%06d.tar"),
        maxcount=args.max_shard_size,
        encoder=False,
    )
    try:
        for session_index, source_root in enumerate(args.source_root):
            sam_rows = load_grounded_sam_metadata(
                source_root, args.grounded_sam_dir, args.grounded_sam_metadata
            )
            reference_bboxes = (
                load_reference_bboxes(
                    source_root / args.reference_label_dir,
                    args.reference_label_policy,
                )
                if args.single_instance_policy == "epnp_bbox"
                else {}
            )
            if args.single_instance_policy == "epnp_bbox" and not reference_bboxes:
                raise FileNotFoundError(
                    "No detector_bbox_xywh references were found under "
                    f"{source_root / args.reference_label_dir}"
                )
            sam_rows = sam_rows[:: args.frame_stride]
            if args.max_frames_per_session is not None:
                sam_rows = sam_rows[: args.max_frames_per_session]

            scene_id = session_index + 1
            for row in sam_rows:
                frame_file = str(row.get("frame_file", ""))
                image_path = find_image(source_root, frame_file, image_dirs)
                if image_path is None:
                    skip("missing_image")
                    continue
                mask_path = source_root / args.grounded_sam_dir / str(row["mask_png"])
                with Image.open(image_path) as image:
                    width, height = image.size
                mask_values = load_mask(mask_path, (width, height))

                yaml_path = find_sample_yaml(source_root, args.sample_metadata_dir, frame_file)
                if yaml_path is not None:
                    K = camera_k_from_yaml(yaml_path)
                elif fallback_K is not None:
                    K = fallback_K
                else:
                    skip("missing_camera_intrinsics")
                    continue

                instance_candidates: list[dict[str, Any]] = []
                for obj in row.get("objects", []):
                    if obj.get("bbox_xyxy") is None:
                        continue
                    sam_object_id = int(obj.get("object_id", len(instance_candidates) + 1))
                    object_mask = mask_values == sam_object_id
                    if not object_mask.any() and len(row.get("objects", [])) == 1:
                        object_mask = mask_values > 0
                    if int(object_mask.sum()) < args.min_mask_pixels:
                        continue
                    bbox = bbox_from_mask(object_mask)
                    if bbox == [0, 0, 0, 0]:
                        bbox = xyxy_to_xywh(obj.get("bbox_xyxy")) or bbox
                    rle = pycoco_utils.binary_mask_to_rle(object_mask.astype(np.uint8))
                    instance_candidates.append(
                        {
                            "sam_object_id": sam_object_id,
                            "bbox": bbox,
                            "bbox_xyxy": obj.get("bbox_xyxy"),
                            "class_name": obj.get("class_name", ""),
                            "score": float(obj.get("score", args.score)),
                            "segmentation": rle,
                        }
                    )

                timestamp = timestamp_from_frame_file(frame_file)
                frame_references = reference_bboxes.get(timestamp or "", [])
                if args.single_instance_policy == "epnp_bbox":
                    if not frame_references:
                        if args.missing_reference_policy == "skip":
                            skip("missing_epnp_reference")
                            continue
                        if args.missing_reference_policy == "error":
                            raise ValueError(
                                f"No usable EPnP detector reference for {frame_file}"
                            )
                    if (
                        args.reference_label_policy == "quality_approved_unique"
                        and len(frame_references) > 1
                    ):
                        skip("ambiguous_epnp_reference")
                        continue
                selected_candidates, selection_reason = select_single_instance(
                    instance_candidates,
                    args.single_instance_policy,
                    K,
                    frame_references,
                )
                record_selection(selection_reason)
                instance_count = len(selected_candidates)
                instances = []
                for candidate in selected_candidates:
                    detections.append(
                        {
                            "scene_id": scene_id,
                            "image_id": int(row.get("frame_index", len(frame_map))),
                            "category_id": args.object_id,
                            "score": candidate["score"],
                            "bbox": candidate["bbox"],
                            "segmentation": candidate["segmentation"],
                            "time": 0.0,
                            "sam_object_id": candidate["sam_object_id"],
                            "class_name": candidate["class_name"],
                            "mask_provenance": args.grounded_sam_dir,
                            "instance_selection": selection_reason,
                        }
                    )
                    instances.append(
                        {
                            "sam_object_id": candidate["sam_object_id"],
                            "bbox": candidate["bbox"],
                            "bbox_xyxy": candidate["bbox_xyxy"],
                            "class_name": candidate["class_name"],
                        }
                    )

                if instance_count == 0:
                    skip("no_valid_masks")
                    continue

                im_id = int(row.get("frame_index", len(frame_map)))
                key = f"{scene_id:06d}_{im_id:06d}"
                camera = {"cam_K": K.reshape(-1).tolist(), "depth_scale": 1.0}
                writer.write(
                    {
                        "__key__": key,
                        "rgb.jpg": image_bytes(image_path),
                        "depth.png": png_bytes(np.zeros((height, width), dtype=np.uint16)),
                        "camera.json": json.dumps(camera).encode(),
                    }
                )
                targets.append(
                    {
                        "scene_id": scene_id,
                        "im_id": im_id,
                        "obj_id": args.object_id,
                        "inst_count": instance_count,
                    }
                )
                frame_map.append(
                    {
                        "scene_id": scene_id,
                        "im_id": im_id,
                        "session_index": session_index,
                        "source_root": str(source_root),
                        "camera_id": source_root.name,
                        "frame_index": int(row.get("frame_index", im_id)),
                        "frame_file": frame_file,
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "sample_metadata_path": str(yaml_path) if yaml_path else None,
                        "instances": instances,
                        "available_grounded_sam_instances": len(instance_candidates),
                        "instance_selection": selection_reason,
                    }
                )
    finally:
        writer.close()

    if not targets:
        raise RuntimeError(f"No frames were prepared. Skipped counts: {skipped}")

    (test_dir / "key_to_shard.json").write_text(json.dumps(key_to_shard_map(test_dir), indent=2))
    (dataset_dir / "test_targets_bop19.json").write_text(json.dumps(targets, indent=2))
    (dataset_dir / "frame_map.json").write_text(json.dumps(frame_map, indent=2))
    detection_path.write_text(json.dumps(detections, indent=2))
    (dataset_dir / "inference_metadata.json").write_text(
        json.dumps(
            {
                "dataset_name": args.dataset_name,
                "source_roots": [str(path) for path in args.source_root],
                "grounded_sam_dir": args.grounded_sam_dir,
                "grounded_sam_metadata": args.grounded_sam_metadata,
                "image_dirs": image_dirs,
                "cad_path": str(args.cad_path),
                "object_id": args.object_id,
                "detection_file": str(detection_path),
                "prepared_frames": len(targets),
                "detections": len(detections),
                "single_instance_policy": args.single_instance_policy,
                "reference_label_dir": args.reference_label_dir,
                "reference_label_policy": args.reference_label_policy,
                "missing_reference_policy": args.missing_reference_policy,
                "instance_selection_counts": instance_selection_counts,
                "skipped": skipped,
            },
            indent=2,
        )
    )
    print(f"Prepared {len(targets)} frames and {len(detections)} detections.")
    print(f"Dataset: {dataset_dir}")
    print(f"Detections: {detection_path}")
    print(f"Instance selection: {instance_selection_counts}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
