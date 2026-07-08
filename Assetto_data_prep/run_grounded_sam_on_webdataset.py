#!/usr/bin/env python3
"""Run GroundingDINO + SAM2 on a GigaPose/WebDataset split.

This is a command-line version of the Grounded-SAM workflow adapted to prepared
GigaPose WebDataset shards. It reads ``*.rgb.jpg`` images directly from
``shard-*.tar`` files and writes:

- per-frame integer PNG masks;
- per-frame JSON metadata;
- optional overlays;
- one detection JSON that can be consumed by
  ``Assetto_data_prep.filter_webdataset_by_detection_count``.

Unlike the original video-tracking script, this runs SAM2 independently per
image. That is slower, but it is deterministic and matches the dataset-cleaning
goal: count how many cars an independent detector/segmenter sees in each stored
sample.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-split",
        type=Path,
        required=True,
        help="Prepared WebDataset split, e.g. .../val_pbr_web.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output folder for masks, metadata, overlays, and detections JSON.",
    )
    parser.add_argument("--text", default="race car.", help="GroundingDINO prompt.")
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--sam2-model-cfg", required=True)
    parser.add_argument("--grounding-model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--min-mask-pixels", type=int, default=25)
    parser.add_argument(
        "--allowed-label-substrings",
        default="car,race car,vehicle",
        help=(
            "Comma-separated label substrings to keep after GroundingDINO. "
            "Set to an empty string to disable label filtering."
        ),
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.45,
        help="Box NMS IoU threshold before SAM2. Lower removes more duplicate boxes.",
    )
    parser.add_argument(
        "--mask-nms-iou",
        type=float,
        default=0.70,
        help="Mask IoU threshold after SAM2. Lower removes more duplicate masks.",
    )
    parser.add_argument(
        "--min-box-area-frac",
        type=float,
        default=1e-5,
        help="Reject boxes smaller than this fraction of image area.",
    )
    parser.add_argument(
        "--max-box-area-frac",
        type=float,
        default=0.40,
        help="Reject boxes larger than this fraction of image area.",
    )
    parser.add_argument("--min-aspect", type=float, default=0.20)
    parser.add_argument("--max-aspect", type=float, default=6.0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--filter-ego-car",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop boxes near the lower-middle of selected camera views.",
    )
    parser.add_argument(
        "--ego-filter-cameras",
        default="front",
        help=(
            "Comma-separated camera_id values where the ego-car bottom-center "
            "filter should be applied. Default: front. Use 'all' for the old "
            "behavior, or an empty string with --no-filter-ego-car to disable."
        ),
    )
    parser.add_argument(
        "--camera-map",
        type=Path,
        default=None,
        help=(
            "Optional frame_map.json/CSV with scene_id, im_id, camera_id. "
            "If omitted, tries <dataset_dir>/frame_map.json then frame_map.csv."
        ),
    )
    parser.add_argument("--ego-x-min-frac", type=float, default=0.20)
    parser.add_argument("--ego-x-max-frac", type=float, default=0.80)
    parser.add_argument("--ego-y-min-frac", type=float, default=0.65)
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--detections-name", default="gsam_detections.json")
    parser.add_argument("--metadata-name", default="metadata.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def configure_device(args: argparse.Namespace) -> str:
    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if device == "cuda":
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    return device


def iter_rgb_images(split_dir: Path) -> tuple[list[tuple[str, bytes]], dict[str, Any]]:
    items: list[tuple[str, bytes]] = []
    shard_counts: dict[str, int] = {}
    skipped_image_like = 0
    shards = sorted(split_dir.glob("shard-*.tar"))
    for shard in shards:
        count = 0
        with tarfile.open(shard) as tar:
            for member in sorted(tar.getmembers(), key=lambda item: item.name):
                if not member.isfile():
                    continue
                if member.name.endswith(".rgb.jpg"):
                    key = member.name.removesuffix(".rgb.jpg")
                elif member.name.endswith(".rgb.png"):
                    key = member.name.removesuffix(".rgb.png")
                else:
                    if ".rgb." in member.name:
                        skipped_image_like += 1
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                items.append((key, handle.read()))
                count += 1
        shard_counts[shard.name] = count
    scan_report = {
        "input_split": str(split_dir),
        "shards_found": len(shards),
        "rgb_images_found": len(items),
        "rgb_images_per_shard": shard_counts,
        "skipped_unknown_rgb_members": skipped_image_like,
    }
    return items, scan_report


def parse_key(key: str) -> tuple[int, int]:
    scene_id, im_id = key.split("_", 1)
    return int(scene_id), int(im_id)


def parse_camera_filter(value: str) -> set[str]:
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def load_camera_map(path: Path | None, input_split: Path) -> tuple[dict[tuple[int, int], str], str | None]:
    candidates = []
    if path is not None:
        candidates.append(path)
    else:
        dataset_dir = input_split.parent
        candidates.extend([dataset_dir / "frame_map.json", dataset_dir / "frame_map.csv"])

    selected = next((candidate for candidate in candidates if candidate.is_file()), None)
    if selected is None:
        return {}, None

    if selected.suffix.lower() == ".json":
        rows = json.loads(selected.read_text())
    else:
        with selected.open(newline="") as handle:
            rows = list(csv.DictReader(handle))

    camera_map: dict[tuple[int, int], str] = {}
    for row in rows:
        try:
            scene_id = int(row["scene_id"])
            im_id = int(row["im_id"])
            camera_id = str(row["camera_id"])
        except (KeyError, TypeError, ValueError):
            continue
        camera_map[(scene_id, im_id)] = camera_id
    return camera_map, str(selected)


def should_filter_ego_camera(camera_id: str, args: argparse.Namespace) -> bool:
    if not args.filter_ego_car:
        return False
    camera_filter = parse_camera_filter(args.ego_filter_cameras)
    if "all" in camera_filter:
        return True
    return camera_id.lower() in camera_filter


def filter_ego_car(
    boxes: torch.Tensor,
    labels: list[str],
    scores: torch.Tensor,
    image_size: tuple[int, int],
    camera_id: str,
    args: argparse.Namespace,
    device: str,
) -> tuple[torch.Tensor, list[str], torch.Tensor]:
    if not should_filter_ego_camera(camera_id, args) or boxes.numel() == 0:
        return boxes, labels, scores
    width, height = image_size
    keep_boxes = []
    keep_labels = []
    keep_scores = []
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.tolist()
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        is_ego = (
            args.ego_x_min_frac * width < cx < args.ego_x_max_frac * width
            and cy > args.ego_y_min_frac * height
        )
        if not is_ego:
            keep_boxes.append(box)
            keep_labels.append(label)
            keep_scores.append(score)
    if not keep_boxes:
        return torch.empty((0, 4), device=device), [], torch.empty((0,), device=device)
    return torch.stack(keep_boxes).to(device), keep_labels, torch.stack(keep_scores).to(device)


def parse_label_filters(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def box_area(box: np.ndarray) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1 + 1.0) * max(0.0, y2 - y1 + 1.0)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1 + 1.0), max(0.0, iy2 - iy1 + 1.0)
    inter = iw * ih
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def filter_boxes_by_label_and_shape(
    boxes: torch.Tensor,
    labels: list[str],
    scores: torch.Tensor,
    image_size: tuple[int, int],
    args: argparse.Namespace,
    device: str,
) -> tuple[torch.Tensor, list[str], torch.Tensor, list[str]]:
    if boxes.numel() == 0:
        return boxes, labels, scores, []

    width, height = image_size
    image_area = float(width * height)
    allowed = parse_label_filters(args.allowed_label_substrings)
    keep_boxes = []
    keep_labels = []
    keep_scores = []
    rejected_reasons = []

    for box, label, score in zip(boxes, labels, scores):
        label_text = str(label).lower()
        if allowed and not any(token in label_text for token in allowed):
            rejected_reasons.append("label_filter")
            continue
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        bw = max(0.0, x2 - x1 + 1.0)
        bh = max(0.0, y2 - y1 + 1.0)
        area_frac = (bw * bh) / max(image_area, 1.0)
        aspect = bw / max(bh, 1e-9)
        if area_frac < args.min_box_area_frac:
            rejected_reasons.append("box_too_small")
            continue
        if area_frac > args.max_box_area_frac:
            rejected_reasons.append("box_too_large")
            continue
        if aspect < args.min_aspect or aspect > args.max_aspect:
            rejected_reasons.append("bad_aspect")
            continue
        keep_boxes.append(box)
        keep_labels.append(label)
        keep_scores.append(score)

    if not keep_boxes:
        return torch.empty((0, 4), device=device), [], torch.empty((0,), device=device), rejected_reasons
    return (
        torch.stack(keep_boxes).to(device),
        keep_labels,
        torch.stack(keep_scores).to(device),
        rejected_reasons,
    )


def nms_boxes(
    boxes: torch.Tensor,
    labels: list[str],
    scores: torch.Tensor,
    iou_threshold: float,
    device: str,
) -> tuple[torch.Tensor, list[str], torch.Tensor]:
    if boxes.numel() == 0 or iou_threshold <= 0:
        return boxes, labels, scores

    boxes_np = boxes.detach().cpu().numpy()
    scores_np = scores.detach().cpu().numpy()
    order = list(np.argsort(-scores_np))
    keep: list[int] = []
    while order:
        current = order.pop(0)
        keep.append(current)
        order = [
            idx
            for idx in order
            if box_iou(boxes_np[current], boxes_np[idx]) <= iou_threshold
        ]
    keep_tensor = torch.as_tensor(keep, dtype=torch.long, device=boxes.device)
    return boxes[keep_tensor].to(device), [labels[i] for i in keep], scores[keep_tensor].to(device)


def xyxy_to_xywh(box: list[float]) -> list[int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [
        int(round(x1)),
        int(round(y1)),
        max(0, int(round(x2 - x1 + 1))),
        max(0, int(round(y2 - y1 + 1))),
    ]


def bbox_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def draw_overlay(image: Image.Image, objects: list[dict[str, Any]], output_path: Path) -> None:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for obj in objects:
        bbox = obj.get("bbox_xyxy")
        if bbox is None:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        draw.rectangle((x1, y1, x2, y2), outline=(255, 210, 0), width=3)
        draw.text((x1, max(0, y1 - 14)), f"{obj.get('object_id')} {obj.get('score', 0):.2f}", fill=(255, 210, 0))
    overlay.save(output_path, quality=94)


def main() -> None:
    args = parse_args()
    import cv2
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists; pass --overwrite")
        shutil.rmtree(args.output_dir)

    mask_png_dir = args.output_dir / "masks_png"
    json_dir = args.output_dir / "json_data"
    overlay_dir = args.output_dir / "overlay"
    for path in (args.output_dir, mask_png_dir, json_dir):
        path.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    device = configure_device(args)
    print("device", device)

    sam2_model = build_sam2(args.sam2_model_cfg, str(args.sam2_checkpoint), device=device)
    image_predictor = SAM2ImagePredictor(sam2_model)
    processor = AutoProcessor.from_pretrained(args.grounding_model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(args.grounding_model_id).to(device)
    grounding_model.eval()

    rgb_items, scan_report = iter_rgb_images(args.input_split)
    camera_map, camera_map_path = load_camera_map(args.camera_map, args.input_split)
    total_rgb_items_before_sampling = len(rgb_items)
    print(json.dumps(scan_report, indent=2))
    if camera_map_path:
        print(f"Loaded camera map with {len(camera_map)} entries from {camera_map_path}")
    elif args.filter_ego_car and "all" not in parse_camera_filter(args.ego_filter_cameras):
        print(
            "WARNING: no frame_map.json/CSV was found, so camera-specific ego-car "
            "filtering will not run. Pass --camera-map or use --ego-filter-cameras all."
        )
    rgb_items = rgb_items[:: args.frame_stride]
    if args.max_images is not None:
        rgb_items = rgb_items[: args.max_images]
    print(
        f"Processing {len(rgb_items)} RGB images "
        f"(found {total_rgb_items_before_sampling}, frame_stride={args.frame_stride}, "
        f"max_images={args.max_images})"
    )

    metadata: dict[str, Any] = {}
    detections: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for index, (key, image_bytes) in enumerate(rgb_items, start=1):
        scene_id, im_id = parse_key(key)
        camera_id = camera_map.get((scene_id, im_id), "unknown_camera")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image, copy=True)
        width, height = image.size

        inputs = processor(images=image, text=args.text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = grounding_model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=[(height, width)],
        )[0]

        rejected_reasons: list[str] = []
        boxes = results["boxes"]
        labels = list(results["labels"])
        scores = results.get("scores", torch.ones((boxes.shape[0],), device=device))
        boxes, labels, scores, rejected_reasons = filter_boxes_by_label_and_shape(
            boxes, labels, scores, image.size, args, device
        )
        boxes, labels, scores = filter_ego_car(
            boxes, labels, scores, image.size, camera_id, args, device
        )
        boxes, labels, scores = nms_boxes(boxes, labels, scores, args.nms_iou, device)

        objects: list[dict[str, Any]] = []
        kept_masks: list[np.ndarray] = []
        combined_mask = np.zeros((height, width), dtype=np.uint16)
        if boxes.shape[0] > 0:
            image_predictor.set_image(image_np)
            masks, sam_scores, _ = image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes,
                multimask_output=False,
            )
            if masks.ndim == 2:
                masks = masks[None]
            elif masks.ndim == 4:
                masks = masks.squeeze(1)

            for det_idx, (mask, box, label, score) in enumerate(
                zip(masks, boxes.detach().cpu().numpy(), labels, scores.detach().cpu().numpy()),
                start=1,
            ):
                mask_bool = np.asarray(mask, dtype=bool)
                pixels = int(mask_bool.sum())
                if pixels < args.min_mask_pixels:
                    continue
                if any(mask_iou(mask_bool, existing) > args.mask_nms_iou for existing in kept_masks):
                    continue
                object_id = len(objects) + 1
                combined_mask[mask_bool] = object_id
                kept_masks.append(mask_bool)
                bbox_xyxy = [int(round(float(v))) for v in box.tolist()]
                bbox_xywh = bbox_from_mask(mask_bool)
                obj = {
                    "object_id": object_id,
                    "class_name": str(label),
                    "score": float(score),
                    "bbox_xyxy": bbox_xyxy,
                    "bbox": bbox_xywh,
                    "mask_pixels": pixels,
                }
                objects.append(obj)
                detections.append(
                    {
                        "scene_id": scene_id,
                        "image_id": im_id,
                        "im_id": im_id,
                        "camera_id": camera_id,
                        "category_id": 1,
                        "score": float(score),
                        "bbox": bbox_xywh,
                        "bbox_xyxy": bbox_xyxy,
                        "object_id": object_id,
                        "class_name": str(label),
                        "mask_pixels": pixels,
                        "source_key": key,
                    }
                )

        if not objects:
            skip("no_valid_masks")

        mask_name = f"mask_{key}.png"
        json_name = f"mask_{key}.json"
        cv2.imwrite(str(mask_png_dir / mask_name), combined_mask)
        (json_dir / json_name).write_text(json.dumps({"key": key, "objects": objects}, indent=2))
        if args.save_overlays:
            draw_overlay(image, objects, overlay_dir / f"{key}_overlay.jpg")

        metadata[key] = {
            "key": key,
            "scene_id": scene_id,
            "im_id": im_id,
            "camera_id": camera_id,
            "frame_index": im_id,
            "frame_file": f"{key}.rgb.jpg",
            "mask_png": str(Path("masks_png") / mask_name),
            "json": str(Path("json_data") / json_name),
            "rejected_box_reasons": rejected_reasons,
            "objects": objects,
        }

        if index % 50 == 0 or index == len(rgb_items):
            print(f"processed {index}/{len(rgb_items)} images")

    (args.output_dir / args.metadata_name).write_text(json.dumps(metadata, indent=2))
    detections_path = args.output_dir / args.detections_name
    detections_path.write_text(json.dumps(detections, indent=2))
    report = {
        "input_split": str(args.input_split),
        "output_dir": str(args.output_dir),
        "scan_report": scan_report,
        "camera_map": camera_map_path,
        "camera_map_entries": len(camera_map),
        "images_found_before_sampling": total_rgb_items_before_sampling,
        "metadata": str(args.output_dir / args.metadata_name),
        "detections": str(detections_path),
        "images_processed": len(rgb_items),
        "detections_count": len(detections),
        "text": args.text,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "min_mask_pixels": args.min_mask_pixels,
        "allowed_label_substrings": args.allowed_label_substrings,
        "nms_iou": args.nms_iou,
        "mask_nms_iou": args.mask_nms_iou,
        "min_box_area_frac": args.min_box_area_frac,
        "max_box_area_frac": args.max_box_area_frac,
        "min_aspect": args.min_aspect,
        "max_aspect": args.max_aspect,
        "filter_ego_car": args.filter_ego_car,
        "ego_filter_cameras": args.ego_filter_cameras,
        "skipped": skipped,
    }
    (args.output_dir / "run_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
