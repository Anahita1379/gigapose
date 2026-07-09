"""Configurable train/validation dataset wrapper for fine-tuning GigaPose."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from bop_toolkit_lib import inout

from src.custom_megapose.template_dataset import TemplateDataset, NearestTemplateFinder
from src.custom_megapose.web_scene_dataset import IterableWebSceneDataset, WebSceneDataset
from src.dataloader.keypoints import KeyPointSampler
from src.dataloader.train import GigaPoseTrainSet
from src.utils.logging import get_logger


logger = get_logger(__name__)


class SplitWebSceneDataset(WebSceneDataset):
    """Read key_to_shard.json from the selected split, including validation."""

    def load_frame_index(self) -> pd.DataFrame:
        mapping_path = self.wds_dir / "key_to_shard.json"
        mapping = json.loads(mapping_path.read_text())
        items = sorted(
            mapping.items(),
            key=lambda item: tuple(int(part) for part in item[0].split("_", 1)),
        )
        keys = [key for key, _ in items]
        shard_filenames = [f"shard-{int(shard_id):06d}.tar" for _, shard_id in items]
        return pd.DataFrame(
            {"key": keys, "shard_fname": shard_filenames}
        )


class AssettoCorsaFineTuneSet(GigaPoseTrainSet):
    PADDED_HEIGHT = 760
    PADDED_WIDTH = 2064
    FRONT_REAR_INVALID_SIDE_PIXELS = 258

    def __init__(
        self,
        batch_size,
        root_dir,
        dataset_name,
        split_name,
        depth_scale,
        template_config,
        transforms,
        retain_heavy_visuals=False,
    ):
        self.batch_size = batch_size
        self.dataset_dir = Path(root_dir) / dataset_name
        self.transforms = transforms
        self.retain_heavy_visuals = bool(retain_heavy_visuals)
        self.deterministic_instance_selection = not bool(
            self.transforms.rgb_augmentation
        )
        if self.transforms.rgb_augmentation:
            self.transforms.rgb_transform.transform = [
                transform for transform in self.transforms.rgb_transform.transform
            ]

        web_dataset = SplitWebSceneDataset(
            self.dataset_dir / split_name, depth_scale=depth_scale
        )
        self.web_dataloader = IterableWebSceneDataset(web_dataset)

        model_infos = inout.load_json(self.dataset_dir / "models_info.json")
        template_config = copy.deepcopy(template_config)
        template_config.dir += f"/{dataset_name}"
        self.template_dataset = TemplateDataset.from_config(model_infos, template_config)
        self.template_finder = NearestTemplateFinder(template_config)
        self.keypoint_sampler = KeyPointSampler()

    def collate_fn(self, batch):
        """Pad every camera to 2064x760 while preserving calibrated geometry.

        Front/rear frames (2064x400) receive 180 rows above and below. Their
        outer 258 columns are also marked invalid, retaining exactly 75% of the
        original width. Stereo frames are already 2064x760 and remain fully
        valid. RGB, depth, segmentation, instance masks, intrinsics, and boxes
        are shifted together.
        """
        if not batch:
            return None

        # Apply photometric training augmentation at native resolution. Padding
        # afterward avoids spending augmentation time on 360 rows of zeros.
        augmentation_applied = bool(self.transforms.rgb_augmentation)
        if augmentation_applied:
            batch = [self.transforms.rgb_transform(sample) for sample in batch]

        valid_masks = []
        original_sizes = []
        padding_offsets = []
        for sample in batch:
            height, width = sample.rgb.shape[:2]
            if height > self.PADDED_HEIGHT or width > self.PADDED_WIDTH:
                raise ValueError(
                    f"Image {width}x{height} exceeds configured padded canvas "
                    f"{self.PADDED_WIDTH}x{self.PADDED_HEIGHT}"
                )

            pad_y = self.PADDED_HEIGHT - height
            pad_x = self.PADDED_WIDTH - width
            top, bottom = pad_y // 2, pad_y - pad_y // 2
            left, right = pad_x // 2, pad_x - pad_x // 2
            padding = ((top, bottom), (left, right))

            valid = np.zeros(
                (self.PADDED_HEIGHT, self.PADDED_WIDTH), dtype=np.float32
            )
            valid[top : top + height, left : left + width] = 1.0
            if height == 400 and width == 2064:
                side = self.FRONT_REAR_INVALID_SIDE_PIXELS
                valid[top : top + height, left : left + side] = 0.0
                valid[
                    top : top + height,
                    left + width - side : left + width,
                ] = 0.0

            sample.rgb = np.pad(
                sample.rgb, padding + ((0, 0),), mode="constant"
            )
            if sample.depth is not None:
                sample.depth = np.pad(sample.depth, padding, mode="constant")
            if sample.segmentation is not None:
                sample.segmentation = np.pad(
                    sample.segmentation, padding, mode="constant"
                )
                sample.segmentation[valid == 0] = 0
            if sample.binary_masks is not None:
                sample.binary_masks = {
                    key: np.logical_and(
                        np.pad(mask, padding, mode="constant"),
                        valid > 0,
                    )
                    for key, mask in sample.binary_masks.items()
                }

            shift = np.array([left, top, 0, 0])
            for object_data in sample.object_datas or []:
                if object_data.bbox_modal is not None:
                    object_data.bbox_modal = object_data.bbox_modal + shift
                if object_data.bbox_amodal is not None:
                    object_data.bbox_amodal = object_data.bbox_amodal + shift

            sample.camera_data.K = np.asarray(
                sample.camera_data.K, dtype=np.float64
            ).copy()
            sample.camera_data.K[0, 2] += left
            sample.camera_data.K[1, 2] += top
            sample.camera_data.resolution = (
                self.PADDED_HEIGHT,
                self.PADDED_WIDTH,
            )
            valid_masks.append(valid)
            original_sizes.append((height, width))
            padding_offsets.append((top, left))

        # process_real is called synchronously inside the parent collator.
        self._collate_valid_masks = np.stack(valid_masks)
        self._collate_original_sizes = np.asarray(original_sizes, dtype=np.int64)
        self._collate_padding_offsets = np.asarray(
            padding_offsets, dtype=np.int64
        )
        self._rgb_augmentation_applied = augmentation_applied
        try:
            return super().collate_fn(batch)
        finally:
            del self._collate_valid_masks
            del self._collate_original_sizes
            del self._collate_padding_offsets
            del self._rgb_augmentation_applied
