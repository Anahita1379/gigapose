"""Configurable train/validation dataset wrapper for fine-tuning GigaPose."""

from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path

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
    def __init__(
        self,
        batch_size,
        root_dir,
        dataset_name,
        split_name,
        depth_scale,
        template_config,
        transforms,
    ):
        self.batch_size = batch_size
        self.dataset_dir = Path(root_dir) / dataset_name
        self.transforms = transforms
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
        """Collate Assetto Corsa samples without mixing full-frame resolutions.

        The AC train split can contain front/rear/stereo cameras with different
        image sizes. MegaPose's base SceneObservation collate stacks full RGB and
        depth before object crops are resized, so mixed resolutions in a single
        DataLoader batch fail with e.g. 760-vs-400 tensor-size errors.
        """
        if len(batch) > 1:
            resolutions = [
                tuple(sample.rgb.shape[:2])
                for sample in batch
                if getattr(sample, "rgb", None) is not None
            ]
            if resolutions and len(set(resolutions)) > 1:
                keep_resolution, _ = Counter(resolutions).most_common(1)[0]
                filtered = [
                    sample
                    for sample in batch
                    if getattr(sample, "rgb", None) is not None
                    and tuple(sample.rgb.shape[:2]) == keep_resolution
                ]
                logger.debug(
                    "Mixed camera resolutions in batch %s; keeping %d/%d samples at %s",
                    sorted(set(resolutions)),
                    len(filtered),
                    len(batch),
                    keep_resolution,
                )
                batch = filtered
        if not batch:
            return None
        return super().collate_fn(batch)
