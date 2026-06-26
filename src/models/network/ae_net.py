import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import logging
from src.utils.batch import BatchedData
from einops import rearrange
from src.utils.logging import get_logger

logger = get_logger(__name__)
descriptor_sizes = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


class AENet(pl.LightningModule):
    def __init__(
        self,
        model_name,
        dinov2_model,
        descriptor_size,
        max_batch_size,
        patch_size=14,
        train_mode="all",
        train_last_n_blocks=1,
        train_block_offsets=None,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.dinov2_model = dinov2_model
        self.descriptor_size = descriptor_size
        self.max_batch_size = max_batch_size
        self.patch_size = patch_size
        self.train_mode = train_mode
        self.train_last_n_blocks = train_last_n_blocks
        self.train_block_offsets = train_block_offsets
        logger.info("Initialize AENet done!")

    def reset_with_pretrained_weights(self):
        device = self.device
        self.dinov2_model = torch.hub.load("facebookresearch/dinov2", self.model_name)
        self.dinov2_model = self.dinov2_model.to(device)

    def get_toUpdate_parameters(self):
        for param in self.dinov2_model.parameters():
            param.requires_grad = False

        if self.train_mode == "all":
            modules = [self.dinov2_model]
        elif self.train_mode in {"last-block", "last-blocks", "block-offsets"}:
            blocks = getattr(self.dinov2_model, "blocks", None)
            if blocks is None:
                raise AttributeError("DINOv2 model has no 'blocks' module list.")
            if self.train_mode == "block-offsets":
                offsets = self.train_block_offsets or [2]
                modules = [blocks[-int(offset)] for offset in offsets]
            else:
                modules = [blocks[-idx] for idx in range(1, self.train_last_n_blocks + 1)]
            if hasattr(self.dinov2_model, "norm"):
                modules.append(self.dinov2_model.norm)
        elif self.train_mode == "norm":
            if not hasattr(self.dinov2_model, "norm"):
                raise AttributeError("DINOv2 model has no 'norm' module.")
            modules = [self.dinov2_model.norm]
        else:
            raise ValueError(
                f"Unknown AE train_mode={self.train_mode!r}; "
                "use all, last-block, last-blocks, block-offsets, or norm."
            )

        params = []
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True
                params.append(param)
        if not params:
            raise RuntimeError(f"AE train_mode={self.train_mode!r} selected no parameters.")
        num_trainable = sum(param.numel() for param in params)
        logger.info(
            "AENet train_mode=%s train_last_n_blocks=%s trainable_params=%d",
            self.train_mode,
            self.train_last_n_blocks,
            num_trainable,
        )
        return params

    def compute_features(self, images):
        # with torch.no_grad():  # no gradients
        features = self.dinov2_model.forward_features(images)
        return features

    def reshape_local_features(self, local_features, num_patches):
        local_features = rearrange(
            local_features, "b (h w) c -> b c h w", h=num_patches[0], w=num_patches[1]
        )
        return local_features

    def forward_by_chunk(self, processed_rgbs, patch_dim=[2, 3]):
        batch_rgbs = BatchedData(batch_size=self.max_batch_size, data=processed_rgbs)
        patch_features = BatchedData(batch_size=self.max_batch_size)

        num_patche_h = processed_rgbs.shape[patch_dim[0]] // self.patch_size
        num_patche_w = processed_rgbs.shape[patch_dim[1]] // self.patch_size

        for idx_sample in range(len(batch_rgbs)):
            feats = self.compute_features(batch_rgbs[idx_sample])
            patch_feats = self.reshape_local_features(
                feats["x_prenorm"][:, 1:, :],
                num_patches=[num_patche_h, num_patche_w],
            )
            patch_features.cat(patch_feats)
        return F.normalize(patch_features.data, dim=1)

    def forward(self, images):
        features = self.forward_by_chunk(images)
        return features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from omegaconf import DictConfig, OmegaConf
    from hydra.utils import instantiate
    from PIL import Image
    from hydra.experimental import compose, initialize

    with initialize(config_path="../../../configs/"):
        cfg = compose(config_name="train.yaml")

    model = instantiate(cfg.model.ae_net)
    model = model.to("cuda")

    images = torch.rand(2, 3, 224, 224).to(device="cuda")
    features = model(images)
    print(features.shape)
