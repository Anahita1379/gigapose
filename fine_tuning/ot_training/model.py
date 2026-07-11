"""GigaPose subclass for isolated OT/soft-correspondence AE fine-tuning."""

from __future__ import annotations

import os
import os.path as osp

import torch

from src.libVis.torch import plot_keypoints_batch, save_tensor_to_image
from src.models.gigaPose import GigaPose
from src.utils.logging import get_logger, log_image

from .losses import ot_correspondence_losses


logger = get_logger(__name__)


class OTGigaPose(GigaPose):
    """Train AE/DINO features with masked optimal-transport correspondence loss."""

    def __init__(self, *args, ot_loss_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        config = dict(ot_loss_config or {})
        self.ot_loss_config = {
            "feature_temperature": float(config.get("feature_temperature", 0.07)),
            "sinkhorn_iterations": int(config.get("sinkhorn_iterations", 30)),
            "correspondence_weight": float(config.get("correspondence_weight", 1.0)),
            "soft_patch_reprojection_weight": float(
                config.get("soft_patch_reprojection_weight", 0.25)
            ),
            "soft_affine_center_weight": float(
                config.get("soft_affine_center_weight", 0.25)
            ),
            "entropy_weight": float(config.get("entropy_weight", 0.0)),
            "reprojection_beta": float(config.get("reprojection_beta", 0.05)),
            "hard_match_confidence": float(
                config.get("hard_match_confidence", 0.05)
            ),
        }

    @staticmethod
    def _batch_size(batch) -> int:
        return int(batch.src_img.shape[0])

    def _log_metrics(
        self,
        split: str,
        metrics: dict[str, torch.Tensor],
        *,
        batch_size: int,
        prog_bar: tuple[str, ...] = (),
    ) -> None:
        for name, value in metrics.items():
            self.log(
                f"{split}/{name}",
                value,
                sync_dist=True,
                on_step=split == "train",
                on_epoch=split != "train",
                prog_bar=name in prog_bar,
                batch_size=batch_size,
            )

    def compute_ot_objective(self, batch, split: str):
        batch_size = self._batch_size(batch)
        src_features = self.ae_net(batch.src_img)
        tar_features = self.ae_net(batch.tar_img)
        outputs = ot_correspondence_losses(
            src_features=src_features,
            tar_features=tar_features,
            src_mask=batch.src_mask,
            tar_mask=batch.tar_mask,
            src_pts=batch.src_pts,
            tar_pts=batch.tar_pts,
            src_pose=batch.src_pose,
            tar_pose=batch.tar_pose,
            src_K=batch.src_K,
            tar_K=batch.tar_K,
            src_M=batch.src_M,
            tar_M=batch.tar_M,
            feature_temperature=self.ot_loss_config["feature_temperature"],
            sinkhorn_iterations=self.ot_loss_config["sinkhorn_iterations"],
            patch_size=int(self.ae_net.patch_size),
            correspondence_weight=self.ot_loss_config["correspondence_weight"],
            soft_patch_reprojection_weight=self.ot_loss_config[
                "soft_patch_reprojection_weight"
            ],
            soft_affine_center_weight=self.ot_loss_config[
                "soft_affine_center_weight"
            ],
            entropy_weight=self.ot_loss_config["entropy_weight"],
            reprojection_beta=self.ot_loss_config["reprojection_beta"],
            hard_match_confidence=self.ot_loss_config["hard_match_confidence"],
        )
        setattr(batch, "pred_src_pts", outputs.hard_src_pts.detach())
        setattr(batch, "pred_tar_pts", outputs.hard_tar_pts.detach())
        self._log_metrics(
            split,
            {
                "loss_ot_total": outputs.total,
                "loss_ot_correspondence": outputs.correspondence,
                "loss_ot_soft_patch_reprojection": outputs.soft_patch_reprojection,
                "loss_ot_soft_affine_center": outputs.soft_affine_center,
                "loss_ot_entropy": outputs.entropy,
                "monitor_ot_gt_probability": outputs.mean_gt_probability.detach(),
                "monitor_ot_gt_top1_accuracy": outputs.gt_top1_accuracy.detach(),
                "monitor_ot_gt_confident_mutual_accuracy": (
                    outputs.gt_confident_mutual_accuracy.detach()
                ),
                "monitor_ot_mean_confident_match_score": (
                    outputs.mean_confident_match_score.detach()
                ),
                "monitor_ot_valid_gt_pairs": outputs.valid_gt_pairs.detach(),
                "monitor_ot_valid_soft_rows": outputs.valid_soft_rows.detach(),
                "monitor_ot_mutual_matches_per_instance": (
                    outputs.mutual_matches_per_instance.detach()
                ),
                "monitor_ot_confident_mutual_matches_per_instance": (
                    outputs.confident_mutual_matches_per_instance.detach()
                ),
            },
            prog_bar=("loss_ot_total", "monitor_ot_gt_top1_accuracy"),
            batch_size=batch_size,
        )
        return outputs.total

    def training_step(self, batchs, idx_batch):
        if self.trainer.global_step < self.optim_config.warm_up_steps:
            self.warm_up_lr()
        elif self.trainer.global_step == self.optim_config.warm_up_steps:
            logger.info("Finished warm up, setting lr to %s", self.lr)

        total = None
        total_batch_size = 0
        for batch in batchs:
            if batch is None:
                continue
            total_batch_size += self._batch_size(batch)
            dataset_loss = self.compute_ot_objective(batch, "train")
            total = dataset_loss if total is None else total + dataset_loss
        if total is None:
            total = next(self.parameters()).sum() * 0.0
            total_batch_size = 1
        self.log(
            "train/loss",
            total,
            sync_dist=True,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=total_batch_size,
        )
        self.log(
            "total",
            total,
            sync_dist=True,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=total_batch_size,
        )
        return total

    def validation_step(self, batch, idx_batch):
        if batch is None:
            return None
        batch_size = self._batch_size(batch)
        total = self.compute_ot_objective(batch, "val")
        self.log_validation_keypoints(batch, idx_batch, "val", type_data="gt")
        self.log_validation_keypoints(batch, idx_batch, "val", type_data="pred")
        self.log(
            "val/loss",
            total,
            sync_dist=True,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=batch_size,
        )
        return total

    def log_validation_keypoints(self, batch, idx_batch, split, type_data="gt"):
        if idx_batch != 0 and idx_batch % self.log_interval != 0:
            return
        vis_pts = plot_keypoints_batch(batch, type_data=type_data)
        image_dir = osp.join(self.log_dir, "validation_images")
        os.makedirs(image_dir, exist_ok=True)
        sample_path = osp.join(
            image_dir,
            f"{split}_ot_{type_data}_step{int(self.global_step):06d}_"
            f"batch{idx_batch:04d}_rank{self.global_rank}.png",
        )
        save_tensor_to_image(vis_pts, sample_path)
        log_image(
            logger=self.logger,
            name=f"vis/{split}_ot_{type_data}_samples",
            path=sample_path,
            step=int(self.global_step),
        )
