"""I/O adapters for prepared GigaPose datasets and prediction CSV files."""

from __future__ import annotations

import csv
import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Sequence

import cv2
import numpy as np
from PIL import Image

from tracking.geometry import pose_from_rt
from tracking.types import Detection, FrameData, PoseHypothesis


def _parse_numbers(value: str, count: int) -> np.ndarray:
    values = np.fromstring(str(value), sep=" ", dtype=float)
    if values.size != count:
        raise ValueError(f"Expected {count} values, got {values.size}: {value!r}")
    return values


def _bbox_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return np.asarray([0.0, 0.0, 0.0, 0.0])
    return np.asarray(
        [
            float(xs.min()),
            float(ys.min()),
            float(xs.max() - xs.min() + 1),
            float(ys.max() - ys.min() + 1),
        ]
    )


def decode_uncompressed_rle(rle: dict[str, Any]) -> np.ndarray:
    """Decode the uncompressed COCO RLE used by prepared GigaPose shards."""

    height, width = (int(v) for v in rle["size"])
    counts = rle["counts"]
    if isinstance(counts, str):
        raise ValueError(
            "Compressed COCO RLE is unsupported without pycocotools. "
            "Use frame_map masks or install pycocotools."
        )
    flat = np.zeros(height * width, dtype=bool)
    cursor = 0
    value = False
    for count in counts:
        count = int(count)
        if value and count > 0:
            flat[cursor : cursor + count] = True
        cursor += count
        value = not value
    if cursor != flat.size:
        raise ValueError(f"Invalid RLE: decoded {cursor} pixels, expected {flat.size}.")
    return flat.reshape((height, width), order="F")


class PredictionCSVProvider:
    """Index fresh/global GigaPose hypotheses by image and predicted instance."""

    def __init__(
        self,
        path: Path,
        translation_unit: str = "mm",
        source_name: str = "gigapose",
    ):
        self.path = Path(path)
        if translation_unit not in {"m", "mm"}:
            raise ValueError("translation_unit must be 'm' or 'mm'.")
        scale = 1.0 if translation_unit == "m" else 0.001
        grouped: dict[tuple[int, int, int], list[PoseHypothesis]] = defaultdict(list)
        with self.path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"scene_id", "im_id", "obj_id", "score", "R", "t"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{self.path} is missing columns: {sorted(missing)}")
            fallback_instances: dict[tuple[int, int], int] = defaultdict(int)
            previous_key: tuple[int, int] | None = None
            previous_instance: int | None = None
            previous_rank = -1
            for row_index, row in enumerate(reader):
                scene_id, im_id = int(row["scene_id"]), int(row["im_id"])
                rank = int(row.get("rank", 0) or 0)
                instance_text = row.get("instance_id", "")
                if instance_text not in (None, ""):
                    instance_id = int(instance_text)
                else:
                    image_key = (scene_id, im_id)
                    # A conservative fallback for ordinary one-row-per-instance CSVs.
                    if previous_key != image_key or rank <= previous_rank:
                        instance_id = fallback_instances[image_key]
                        fallback_instances[image_key] += 1
                    else:
                        instance_id = int(previous_instance)
                rotation = _parse_numbers(row["R"], 9).reshape(3, 3)
                translation = _parse_numbers(row["t"], 3) * scale
                hypothesis = PoseHypothesis(
                    pose=pose_from_rt(rotation, translation),
                    source=str(source_name),
                    measurement_score=float(row["score"]),
                    obj_id=int(row["obj_id"]),
                    prediction_instance_id=instance_id,
                    rank=rank,
                    metadata={"csv_row": row_index},
                )
                grouped[(scene_id, im_id, instance_id)].append(hypothesis)
                previous_key, previous_instance, previous_rank = (
                    (scene_id, im_id),
                    instance_id,
                    rank,
                )
        self._by_frame: dict[tuple[int, int], list[list[PoseHypothesis]]] = defaultdict(list)
        for (scene_id, im_id, _), values in grouped.items():
            values.sort(key=lambda item: (-item.measurement_score, item.rank))
            for rank, item in enumerate(values):
                item.rank = rank
            self._by_frame[(scene_id, im_id)].append(values)
        for values in self._by_frame.values():
            values.sort(
                key=lambda group: (
                    int(group[0].prediction_instance_id or 0),
                    -group[0].measurement_score,
                )
            )

    def groups_for_frame(self, scene_id: int, im_id: int) -> list[list[PoseHypothesis]]:
        return [
            [hypothesis.copy() for hypothesis in group]
            for group in self._by_frame.get((int(scene_id), int(im_id)), [])
        ]


def merge_prediction_group_sets(
    group_sets: Sequence[Sequence[Sequence[PoseHypothesis]]],
) -> list[list[PoseHypothesis]]:
    """Merge GigaPose/EPnP groups sharing object and external instance ID."""

    merged: dict[tuple[int, int], list[PoseHypothesis]] = defaultdict(list)
    fallback = 0
    for groups in group_sets:
        for group in groups:
            if not group:
                continue
            instance_id = group[0].prediction_instance_id
            if instance_id is None:
                instance_id = -(fallback + 1)
                fallback += 1
            merged[(group[0].obj_id, int(instance_id))].extend(
                item.copy() for item in group
            )
    output = []
    for group in merged.values():
        group.sort(key=lambda item: -item.measurement_score)
        for rank, item in enumerate(group):
            item.rank = rank
        output.append(group)
    output.sort(
        key=lambda group: (
            int(group[0].prediction_instance_id or 0),
            -group[0].measurement_score,
        )
    )
    return output


class WebDatasetSequence:
    """Sequential frame reader for datasets prepared by this repository."""

    def __init__(
        self,
        dataset_dir: Path,
        split: str = "test",
        *,
        load_depth: bool = True,
        scene_id: int | None = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.split_dir = self.dataset_dir / split
        self.load_depth = bool(load_depth)
        frame_map_path = self.dataset_dir / "frame_map.json"
        if not frame_map_path.is_file():
            raise FileNotFoundError(f"Missing sequential frame map: {frame_map_path}")
        rows = json.loads(frame_map_path.read_text())
        self.rows = [
            row
            for row in rows
            if scene_id is None or int(row["scene_id"]) == int(scene_id)
        ]
        self.rows.sort(key=lambda row: (int(row["scene_id"]), int(row["im_id"])))
        mapping_path = self.split_dir / "key_to_shard.json"
        self.key_to_shard = json.loads(mapping_path.read_text()) if mapping_path.is_file() else {}

    def __len__(self) -> int:
        return len(self.rows)

    def _shard_path(self, key: str) -> Path:
        value = self.key_to_shard[key]
        if isinstance(value, str) and value.endswith(".tar"):
            return self.split_dir / value
        return self.split_dir / f"shard-{int(value):06d}.tar"

    def _tar_members(self, key: str, suffixes: Sequence[str]) -> dict[str, bytes]:
        if key not in self.key_to_shard:
            return {}
        result: dict[str, bytes] = {}
        with tarfile.open(self._shard_path(key)) as tar:
            names = {member.name for member in tar.getmembers() if member.isfile()}
            for suffix in suffixes:
                name = f"{key}.{suffix}"
                if name not in names:
                    continue
                handle = tar.extractfile(name)
                if handle is not None:
                    result[suffix] = handle.read()
        return result

    @staticmethod
    def _read_image(path: Path | None, members: dict[str, bytes]) -> np.ndarray:
        if path is not None and path.is_file():
            return np.asarray(Image.open(path).convert("RGB"))
        for suffix in ("rgb.jpg", "rgb.png"):
            if suffix in members:
                return np.asarray(Image.open(io.BytesIO(members[suffix])).convert("RGB"))
        raise FileNotFoundError("RGB image is absent from frame_map and WebDataset shard.")

    @staticmethod
    def _read_integer_mask(path: Path | None, shape: tuple[int, int]) -> np.ndarray | None:
        if path is None or not path.is_file():
            return None
        mask = np.asarray(Image.open(path))
        if mask.ndim == 3:
            if mask.shape[2] >= 3:
                mask = (
                    mask[..., 0].astype(np.int64)
                    + (mask[..., 1].astype(np.int64) << 8)
                    + (mask[..., 2].astype(np.int64) << 16)
                )
            else:
                mask = mask[..., 0]
        if mask.shape != shape:
            mask = cv2.resize(
                mask.astype(np.int32),
                (shape[1], shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return mask

    def _load_frame(self, row: dict[str, Any]) -> FrameData:
        scene_id, im_id = int(row["scene_id"]), int(row["im_id"])
        key = f"{scene_id:06d}_{im_id:06d}"
        suffixes = [
            "rgb.jpg",
            "rgb.png",
            "depth.png",
            "camera.json",
            "camera_data.json",
            "gt_info.json",
            "mask_visib.json",
        ]
        members = self._tar_members(key, suffixes)
        image_path = Path(row["image_path"]) if row.get("image_path") else None
        image = self._read_image(image_path, members)
        height, width = image.shape[:2]

        camera_bytes = members.get("camera.json") or members.get("camera_data.json")
        if camera_bytes is None:
            if "cam_K" not in row:
                raise KeyError(
                    f"No camera intrinsics for {key}; expected shard camera JSON or cam_K."
                )
            camera = row
        else:
            camera = json.loads(camera_bytes)
        K = np.asarray(camera.get("cam_K") or camera.get("K"), dtype=float).reshape(3, 3)

        depth_m = None
        if self.load_depth:
            depth_path = Path(row["depth_path"]) if row.get("depth_path") else None
            if depth_path is not None and depth_path.is_file():
                raw_depth = np.asarray(Image.open(depth_path), dtype=float)
            elif "depth.png" in members:
                raw_depth = np.asarray(Image.open(io.BytesIO(members["depth.png"])), dtype=float)
            else:
                raw_depth = None
            if raw_depth is not None:
                # BOP depth is raw_value * depth_scale millimetres.
                depth_scale = float(camera.get("depth_scale", 1.0))
                depth_m = raw_depth * depth_scale * 0.001

        mask_path_text = row.get("mask_path") or row.get("instance_mask_path")
        mask_path = Path(mask_path_text) if mask_path_text else None
        integer_mask = self._read_integer_mask(mask_path, (height, width))
        instances = list(row.get("instances", []))
        detections: list[Detection] = []
        for index, instance in enumerate(instances):
            mask_id = int(
                instance.get(
                    "sam_object_id",
                    instance.get("object_id", instance.get("opp_id", index + 1)),
                )
            )
            bbox = np.asarray(
                instance.get("bbox", instance.get("bbox_visib", [0, 0, 0, 0])),
                dtype=float,
            )
            if integer_mask is not None:
                mask = integer_mask == mask_id
                if not mask.any() and index + 1 != mask_id:
                    mask = integer_mask == (index + 1)
            else:
                mask = np.zeros((height, width), dtype=bool)
            if not mask.any() and bbox.size == 4:
                x, y, box_width, box_height = np.round(bbox).astype(int)
                x0, y0 = max(x, 0), max(y, 0)
                x1, y1 = min(x + box_width, width), min(y + box_height, height)
                mask[y0:y1, x0:x1] = True
            if bbox.size != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                bbox = _bbox_from_mask(mask)
            detections.append(
                Detection(
                    detection_id=index,
                    bbox_xywh=bbox,
                    mask=mask,
                    obj_id=int(instance.get("obj_id", 1)),
                    external_id=mask_id,
                    score=float(instance.get("score", 1.0)),
                )
            )

        # Generic BOP fallback when frame_map has no explicit instances.
        if not detections and "gt_info.json" in members:
            infos = json.loads(members["gt_info.json"])
            masks = json.loads(members.get("mask_visib.json", b"{}"))
            for index, info in enumerate(infos):
                mask = (
                    decode_uncompressed_rle(masks[str(index)])
                    if str(index) in masks
                    else np.zeros((height, width), dtype=bool)
                )
                bbox = np.asarray(info.get("bbox_visib", _bbox_from_mask(mask)), dtype=float)
                detections.append(
                    Detection(index, bbox, mask, obj_id=1, external_id=index + 1)
                )

        return FrameData(
            scene_id=scene_id,
            im_id=im_id,
            image=image,
            K=K,
            detections=detections,
            depth_m=depth_m,
            image_path=image_path,
            metadata=dict(row),
        )

    def __iter__(self) -> Iterator[FrameData]:
        for row in self.rows:
            yield self._load_frame(row)


def write_tracking_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scene_id",
        "im_id",
        "obj_id",
        "score",
        "R",
        "t",
        "time",
        "instance_id",
        "track_id",
        "tracking_mode",
        "source",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pose_to_csv_row(
    *,
    scene_id: int,
    im_id: int,
    track_id: int,
    obj_id: int,
    confidence: float,
    pose_m: np.ndarray,
    mode: str,
    source: str,
    elapsed_s: float,
) -> dict[str, Any]:
    return {
        "scene_id": scene_id,
        "im_id": im_id,
        "obj_id": obj_id,
        "score": f"{float(confidence):.9g}",
        "R": " ".join(f"{float(value):.9g}" for value in pose_m[:3, :3].reshape(-1)),
        "t": " ".join(f"{float(value * 1000.0):.9g}" for value in pose_m[:3, 3]),
        "time": f"{float(elapsed_s):.9g}",
        "instance_id": track_id,
        "track_id": track_id,
        "tracking_mode": mode,
        "source": source,
    }
