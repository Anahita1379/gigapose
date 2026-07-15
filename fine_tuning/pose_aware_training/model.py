"""GigaPose subclass with pose-aware IST and soft-template objectives."""

from __future__ import annotations

import os
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F

from fine_tuning.early_stopping import pose_score
from src.libVis.torch import save_tensor_to_image
from src.models.gigaPose import GigaPose
from src.utils.batch import gather
from src.utils.logging import get_logger, log_image

from .losses import (
    pose_aware_ist_losses,
    soft_pose_aware_template_loss,
)
from .visualization import plot_scale_alignment_batch


logger = get_logger(__name__)


class PoseAwareGigaPose(GigaPose):
    """Drop-in GigaPose variant used only by the isolated training entry point."""

    def __init__(self, *args, pose_loss_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        config = dict(pose_loss_config or {})
        self.pose_loss_config = {
            "log_depth_weight": float(config.get("log_depth_weight", 1.0)),
            "instance_log_scale_weight": float(
                config.get("instance_log_scale_weight", 0.5)
            ),
            "scale_consistency_weight": float(
                config.get("scale_consistency_weight", 0.05)
            ),
            "inplane_weight": float(config.get("inplane_weight", 1.0)),
            "reprojection_weight": float(config.get("reprojection_weight", 0.0)),
            "anti_flip_weight": float(config.get("anti_flip_weight", 0.0)),
            "optimize_pose_monitor_errors": bool(
                config.get("optimize_pose_monitor_errors", False)
            ),
            "direct_translation_weight": float(
                config.get("direct_translation_weight", 0.05)
            ),
            "direct_depth_weight": float(config.get("direct_depth_weight", 0.05)),
            "direct_rotation_weight": float(
                config.get("direct_rotation_weight", 0.05)
            ),
            "direct_reprojection_weight": float(
                config.get("direct_reprojection_weight", 0.0)
            ),
            "info_nce_weight": float(config.get("info_nce_weight", 1.0)),
            "soft_template_weight": float(config.get("soft_template_weight", 0.25)),
            "log_depth_beta": float(config.get("log_depth_beta", 0.1)),
            "reprojection_beta": float(config.get("reprojection_beta", 0.05)),
            "direct_translation_beta": float(
                config.get("direct_translation_beta", 0.05)
            ),
            "direct_depth_beta": float(config.get("direct_depth_beta", 0.05)),
            "direct_rotation_beta": float(config.get("direct_rotation_beta", 0.05)),
            "anti_flip_margin": float(config.get("anti_flip_margin", 0.25)),
            "direct_translation_scale": float(
                config.get("direct_translation_scale", 1000.0)
            ),
            "direct_depth_scale": float(config.get("direct_depth_scale", 1000.0)),
            "softmax_temperature": float(config.get("softmax_temperature", 0.1)),
            "target_temperature_deg": float(
                config.get("target_temperature_deg", 15.0)
            ),
            "translation_to_mm": float(config.get("translation_to_mm", 1.0)),
        }

    @staticmethod
    def _labels(batch, device: torch.device) -> torch.Tensor:
        values = np.asarray(batch.infos.label).astype(np.int64)
        return torch.as_tensor(values, device=device)

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

    @staticmethod
    def _attach_ist_predictions_for_visualization(batch, predictions) -> None:
        src_pts = batch.src_pts.clone().long()
        tar_pts = batch.tar_pts.clone().long()
        pair_patch_valid = (
            (src_pts[..., 0] != -1)
            & (src_pts[..., 1] != -1)
            & (tar_pts[..., 0] != -1)
            & (tar_pts[..., 1] != -1)
        )
        pred_rel_scale = torch.full(
            pair_patch_valid.shape,
            float("nan"),
            dtype=predictions["scale"].dtype,
            device=predictions["scale"].device,
        )
        pred_rel_inplane = torch.full(
            (*pair_patch_valid.shape, 2),
            float("nan"),
            dtype=predictions["inplane"].dtype,
            device=predictions["inplane"].device,
        )
        if predictions["scale"].shape[0] == int(pair_patch_valid.sum()):
            pred_rel_scale[pair_patch_valid] = predictions["scale"]
            pred_rel_inplane[pair_patch_valid] = predictions["inplane"]
        setattr(batch, "pred_relScale", pred_rel_scale)
        setattr(batch, "pred_relInplane", pred_rel_inplane)

    def log_validation_ist_overlay(self, batch, idx_batch, split):
        """Use the isolated scale-diagnostic overlay for pose-aware runs."""
        if idx_batch != 0 and idx_batch % self.log_interval != 0:
            return
        vis_overlay = plot_scale_alignment_batch(batch)
        image_dir = osp.join(self.log_dir, "validation_images")
        os.makedirs(image_dir, exist_ok=True)
        sample_path = osp.join(
            image_dir,
            f"{split}_ist_overlay_step{int(self.global_step):06d}_"
            f"batch{idx_batch:04d}_rank{self.global_rank}.png",
        )
        save_tensor_to_image(vis_overlay, sample_path)
        log_image(
            logger=self.logger,
            name=f"vis/{split}_ist_overlay",
            path=sample_path,
            step=int(self.global_step),
        )

    def compute_pose_aware_ist_objective(self, batch, split: str):
        batch_size = self._batch_size(batch)
        predictions = self.ist_net(
            src_img=batch.src_img,
            tar_img=batch.tar_img,
            src_pts=batch.src_pts.clone().long(),
            tar_pts=batch.tar_pts.clone().long(),
        )
        self._attach_ist_predictions_for_visualization(batch, predictions)
        outputs = pose_aware_ist_losses(
            pred_scale=predictions["scale"],
            pred_inplane=predictions["inplane"],
            src_pts=batch.src_pts,
            tar_pts=batch.tar_pts,
            gt_scale=batch.relScale,
            gt_inplane=batch.relInplane,
            src_pose=batch.src_pose,
            tar_pose=batch.tar_pose,
            src_K=batch.src_K,
            tar_K=batch.tar_K,
            src_M=batch.src_M,
            tar_M=batch.tar_M,
            patch_size=int(self.ist_net.patch_size),
            log_depth_beta=self.pose_loss_config["log_depth_beta"],
            reprojection_beta=self.pose_loss_config["reprojection_beta"],
            direct_translation_scale=self.pose_loss_config[
                "direct_translation_scale"
            ],
            direct_depth_scale=self.pose_loss_config["direct_depth_scale"],
            direct_translation_beta=self.pose_loss_config[
                "direct_translation_beta"
            ],
            direct_depth_beta=self.pose_loss_config["direct_depth_beta"],
            direct_rotation_beta=self.pose_loss_config["direct_rotation_beta"],
            anti_flip_margin=self.pose_loss_config["anti_flip_margin"],
        )
        weighted_log_depth = (
            self.pose_loss_config["log_depth_weight"] * outputs.log_depth
        )
        weighted_instance_log_scale = (
            self.pose_loss_config["instance_log_scale_weight"]
            * outputs.instance_log_scale
        )
        weighted_scale_consistency = (
            self.pose_loss_config["scale_consistency_weight"]
            * outputs.scale_consistency
        )
        weighted_inplane = (
            self.pose_loss_config["inplane_weight"] * outputs.inplane
        )
        weighted_reprojection = (
            self.pose_loss_config["reprojection_weight"] * outputs.reprojection
        )
        weighted_anti_flip = (
            self.pose_loss_config["anti_flip_weight"] * outputs.anti_flip
        )
        total = (
            weighted_log_depth
            + weighted_instance_log_scale
            + weighted_scale_consistency
            + weighted_inplane
            + weighted_reprojection
            + weighted_anti_flip
        )
        direct_pose_total = (
            self.pose_loss_config["direct_translation_weight"]
            * outputs.direct_translation
            + self.pose_loss_config["direct_depth_weight"]
            * outputs.direct_depth
            + self.pose_loss_config["direct_rotation_weight"]
            * outputs.direct_rotation
            + self.pose_loss_config["direct_reprojection_weight"]
            * outputs.direct_reprojection
        )
        if self.pose_loss_config["optimize_pose_monitor_errors"]:
            total = total + direct_pose_total
        translation_error_mm = (
            outputs.translation_error
            * self.pose_loss_config["translation_to_mm"]
        )
        depth_error_mm = (
            outputs.depth_abs_error
            * self.pose_loss_config["translation_to_mm"]
        )
        monitor_pose_score = pose_score(
            translation_error_mm,
            outputs.rotation_error_deg,
        )
        self._log_metrics(
            split,
            {
                "loss_log_depth": outputs.log_depth,
                "loss_instance_log_scale": outputs.instance_log_scale,
                "loss_scale_consistency": outputs.scale_consistency,
                "loss_inplane": outputs.inplane,
                "loss_reprojection": outputs.reprojection,
                "loss_direct_translation": outputs.direct_translation,
                "loss_direct_depth": outputs.direct_depth,
                "loss_direct_rotation": outputs.direct_rotation,
                "loss_direct_reprojection": outputs.direct_reprojection,
                "loss_direct_pose_total": direct_pose_total,
                "loss_anti_flip": outputs.anti_flip,
                "loss_ist_pose_aware": total,
                # Human-readable metrics are detached for logging. Their
                # normalized direct-loss counterparts above are optionally
                # optimized when --optimize-pose-monitor-errors is enabled.
                "monitor_translation_error_mm": translation_error_mm.detach(),
                "monitor_depth_abs_error_mm": depth_error_mm.detach(),
                "monitor_rotation_error_deg": outputs.rotation_error_deg.detach(),
                "monitor_pose_score": monitor_pose_score.detach(),
                "monitor_reprojection_error_px": (
                    outputs.reprojection_error_px.detach()
                ),
                "monitor_scale_signed_log_bias": (
                    outputs.scale_signed_log_bias.detach()
                ),
                "monitor_scale_abs_log_error": (
                    outputs.scale_abs_log_error.detach()
                ),
                "monitor_scale_median_pred_gt_ratio": (
                    outputs.scale_median_ratio.detach()
                ),
                "monitor_scale_within_5pct": (
                    outputs.scale_within_5pct.detach()
                ),
                "monitor_scale_within_10pct": (
                    outputs.scale_within_10pct.detach()
                ),
                "monitor_scale_within_20pct": (
                    outputs.scale_within_20pct.detach()
                ),
                "monitor_scale_log_std": outputs.scale_log_std.detach(),
                "monitor_flip_closer_fraction": (
                    outputs.flip_closer_fraction.detach()
                ),
                "monitor_flip_margin": outputs.flip_margin.detach(),
                "valid_instances": outputs.valid_instances.detach(),
                "valid_patch_pairs": outputs.valid_patch_pairs.detach(),
            },
            prog_bar=(
                "loss_log_depth",
                "loss_instance_log_scale",
                "loss_inplane",
                "monitor_translation_error_mm",
                "monitor_rotation_error_deg",
                "monitor_pose_score",
            ),
            batch_size=batch_size,
        )
        return total

    def compute_retrieval_objective(self, batch, split: str):
        """Compute patch InfoNCE and pose-aware soft template selection once."""
        batch_size = self._batch_size(batch)
        src_features = self.ae_net(batch.src_img)
        tar_features = self.ae_net(batch.tar_img)

        src_selected = gather(src_features, batch.src_pts.clone().long())
        tar_selected = gather(tar_features, batch.tar_pts.clone().long())
        if src_selected.shape[0] == 0:
            info_nce = src_features.sum() * 0.0
            pos_similarity = info_nce.detach()
        else:
            labels = torch.arange(
                src_selected.shape[0],
                dtype=torch.long,
                device=src_features.device,
            )
            info_nce = self.training_loss.contrast_loss(
                src_selected,
                tar_selected,
                labels,
            )
            pos_similarity = F.cosine_similarity(
                src_selected, tar_selected, dim=1, eps=1e-8
            ).mean()

        soft_template, soft_metrics = soft_pose_aware_template_loss(
            src_features=src_features,
            tar_features=tar_features,
            src_masks=batch.src_mask,
            tar_masks=batch.tar_mask,
            src_rotations=batch.src_pose[:, :3, :3],
            tar_rotations=batch.tar_pose[:, :3, :3],
            labels=self._labels(batch, src_features.device),
            prediction_temperature=self.pose_loss_config["softmax_temperature"],
            target_temperature_rad=np.deg2rad(
                self.pose_loss_config["target_temperature_deg"]
            ),
        )
        total = (
            self.pose_loss_config["info_nce_weight"] * info_nce
            + self.pose_loss_config["soft_template_weight"] * soft_template
        )
        metrics = {
            "loss_infoNCE": info_nce,
            "loss_soft_template": soft_template,
            "loss_retrieval_total": total,
            "retrieval_positive_similarity": pos_similarity.detach(),
            **{name: value.detach() for name, value in soft_metrics.items()},
        }
        self._log_metrics(
            split,
            metrics,
            prog_bar=(
                "loss_infoNCE",
                "loss_soft_template",
                "soft_template_selected_rotation_deg",
            ),
            batch_size=batch_size,
        )
        return total

    def training_step(self, batchs, idx_batch):
        if self.trainer.global_step < self.optim_config.warm_up_steps:
            self.warm_up_lr()
        elif self.trainer.global_step == self.optim_config.warm_up_steps:
            logger.info("Finished warm up, setting lr to %s", self.lr)

        total = None
        total_batch_size = 0
        for idx_dataset, batch in enumerate(batchs):
            if batch is None:
                continue
            total_batch_size += self._batch_size(batch)
            dataset_loss = batch.src_img.sum() * 0.0
            if self.optim_config.nets_to_train in ("ist", "all"):
                dataset_loss = dataset_loss + self.compute_pose_aware_ist_objective(
                    batch, "train"
                )
            if self.optim_config.nets_to_train == "all":
                dataset_loss = dataset_loss + self.compute_retrieval_objective(
                    batch, "train"
                )
            total = dataset_loss if total is None else total + dataset_loss

        if total is None:
            # This is only reachable if every collated dataset batch was None.
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
        total = batch.src_img.sum() * 0.0
        if self.optim_config.nets_to_train in ("ist", "all"):
            total = total + self.compute_pose_aware_ist_objective(batch, "val")
            self.log_validation_keypoints(batch, idx_batch, "val", type_data="gt")
            self.log_validation_ist_overlay(batch, idx_batch, "val")
        if self.optim_config.nets_to_train == "all":
            total = total + self.compute_retrieval_objective(batch, "val")
        self.log(
            "val/loss",
            total,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return total
