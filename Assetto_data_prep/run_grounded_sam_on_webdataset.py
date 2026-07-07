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
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--filter-ego-car",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop boxes near the lower-middle of the image, matching your existing script.",
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


def iter_rgb_images(split_dir: Path) -> list[tuple[str, bytes]]:
    items: list[tuple[str, bytes]] = []
    for shard in sorted(split_dir.glob("shard-*.tar")):
        with tarfile.open(shard) as tar:
            for member in sorted(tar.getmembers(), key=lambda item: item.name):
                if not member.isfile() or not member.name.endswith(".rgb.jpg"):
                    continue
                key = member.name.replace(".rgb.jpg", "")
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                items.append((key, handle.read()))
    return items


def parse_key(key: str) -> tuple[int, int]:
    scene_id, im_id = key.split("_", 1)
    return int(scene_id), int(im_id)


def filter_ego_car(
    boxes: torch.Tensor,
    labels: list[str],
    scores: torch.Tensor,
    image_size: tuple[int, int],
    args: argparse.Namespace,
    device: str,
) -> tuple[torch.Tensor, list[str], torch.Tensor]:
    if not args.filter_ego_car or boxes.numel() == 0:
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

    rgb_items = iter_rgb_images(args.input_split)
    rgb_items = rgb_items[:: args.frame_stride]
    if args.max_images is not None:
        rgb_items = rgb_items[: args.max_images]

    metadata: dict[str, Any] = {}
    detections: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for index, (key, image_bytes) in enumerate(rgb_items, start=1):
        scene_id, im_id = parse_key(key)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
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

        boxes = results["boxes"]
        labels = list(results["labels"])
        scores = results.get("scores", torch.ones((boxes.shape[0],), device=device))
        boxes, labels, scores = filter_ego_car(boxes, labels, scores, image.size, args, device)

        objects: list[dict[str, Any]] = []
        combined_mask = np.zeros((height, width), dtype=np.uint16)
        if boxes.shape[0] > 0:
            image_predictor.set_image(np.asarray(image))
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
                object_id = len(objects) + 1
                combined_mask[mask_bool] = object_id
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
            "frame_index": im_id,
            "frame_file": f"{key}.rgb.jpg",
            "mask_png": str(Path("masks_png") / mask_name),
            "json": str(Path("json_data") / json_name),
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
        "metadata": str(args.output_dir / args.metadata_name),
        "detections": str(detections_path),
        "images_processed": len(rgb_items),
        "detections_count": len(detections),
        "text": args.text,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "min_mask_pixels": args.min_mask_pixels,
        "skipped": skipped,
    }
    (args.output_dir / "run_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
