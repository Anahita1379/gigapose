"""Configurable train/validation dataset wrapper for fine-tuning GigaPose."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
from bop_toolkit_lib import inout

from src.custom_megapose.template_dataset import TemplateDataset, NearestTemplateFinder
from src.custom_megapose.web_scene_dataset import IterableWebSceneDataset, WebSceneDataset
from src.dataloader.keypoints import KeyPointSampler
from src.dataloader.train import GigaPoseTrainSet


class SplitWebSceneDataset(WebSceneDataset):
    """Read key_to_shard.json from the selected split, including validation."""

    def load_frame_index(self) -> pd.DataFrame:
        mapping_path = self.wds_dir / "key_to_shard.json"
        mapping = json.loads(mapping_path.read_text())
        shard_filenames = [
            f"shard-{int(shard_id):06d}.tar" for shard_id in mapping.values()
        ]
        return pd.DataFrame(
            {"key": list(mapping), "shard_fname": shard_filenames}
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
