"""GigaPose with explicit translation and optional SO(3) residual heads."""

from __future__ import annotations

import os.path as osp
import time

import numpy as np
import torch
import torch.nn.functional as F

from fine_tuning.early_stopping import pose_score
from fine_tuning.pose_aware_training.losses import pose_aware_ist_losses
from fine_tuning.pose_aware_training.model import PoseAwareGigaPose
from src.utils.logging import get_logger

from .geometry import (
    aggregate_ist_pose,
    apply_pose_residual,
    paired_instance_descriptor,
)
from .heads import PoseResidualHead
from .losses import residual_pose_losses


logger = get_logger(__name__)


class ResidualPoseGigaPose(PoseAwareGigaPose):
    """Isolated pose-aware model with a geometrically constrained pose head."""

    def __init__(self, *args, residual_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        config = dict(residual_config or {})
        self.residual_config = {
            "hidden_dim": int(config.get("hidden_dim", 256)),
            "residual_lr": float(config.get("residual_lr", 1e-4)),
            "train_ist": bool(config.get("train_ist", True)),
            "enable_rotation": bool(config.get("enable_rotation", False)),
            "apply_at_inference": bool(config.get("apply_at_inference", True)),
            "max_center_offset_px": float(
                config.get("max_center_offset_px", 56.0)
            ),
            "max_log_depth_residual": float(
                config.get("max_log_depth_residual", 0.5)
            ),
            "max_rotation_deg": float(config.get("max_rotation_deg", 20.0)),
            "center_weight": float(config.get("center_weight", 1.0)),
            "log_depth_weight": float(config.get("log_depth_weight", 1.0)),
            "translation_weight": float(config.get("translation_weight", 0.05)),
            "rotation_weight": float(config.get("rotation_weight", 0.25)),
            "regularization_weight": float(
                config.get("regularization_weight", 1e-3)
            ),
            "center_beta": float(config.get("center_beta", 0.05)),
            "log_depth_beta": float(config.get("log_depth_beta", 0.05)),
            "translation_beta": float(config.get("translation_beta", 0.05)),
            "rotation_beta": float(config.get("rotation_beta", 0.05)),
            "translation_unit": float(config.get("translation_unit", 1000.0)),
            "translation_to_mm": float(config.get("translation_to_mm", 1.0)),
        }
        descriptor_size = int(self.ist_net.regressor.descriptor_size)
        self.pose_residual_head = PoseResidualHead(
            descriptor_size=descriptor_size,
            hidden_dim=self.residual_config["hidden_dim"],
        )

    def warm_up_lr(self):
        for optimizer in self.trainer.optimizers:
            for group in optimizer.param_groups:
                target_lr = group.get("target_lr", group["lr"])
                group["lr"] = (
                    self.global_step
                    / float(self.optim_config.warm_up_steps)
                    * target_lr
                )
            if self.global_step % 50 == 0:
                logger.info(
                    "Step=%d, residual warm up lrs=%s",
                    self.global_step,
                    [group["lr"] for group in optimizer.param_groups],
                )

    def configure_optimizers(self):
        for parameter in self.ae_net.parameters():
            parameter.requires_grad = False
        for parameter in self.ist_net.parameters():
            parameter.requires_grad = self.residual_config["train_ist"]
        for parameter in self.pose_residual_head.parameters():
            parameter.requires_grad = True
        if not self.residual_config["enable_rotation"]:
            for parameter in self.pose_residual_head.rotation.parameters():
                parameter.requires_grad = False

        groups = []
        if self.optim_config.nets_to_train == "all":
            ae_parameters = self.ae_net.get_toUpdate_parameters()
            groups.append(
                {
                    "params": ae_parameters,
                    "lr": self.optim_config.ae_lr,
                    "target_lr": self.optim_config.ae_lr,
                    "name": "ae",
                }
            )
        if self.residual_config["train_ist"]:
            groups.append(
                {
                    "params": self.ist_net.parameters(),
                    "lr": self.optim_config.ist_lr,
                    "target_lr": self.optim_config.ist_lr,
                    "name": "ist",
                }
            )
        groups.append(
            {
                "params": [
                    parameter
                    for parameter in self.pose_residual_head.parameters()
                    if parameter.requires_grad
                ],
                "lr": self.residual_config["residual_lr"],
                "target_lr": self.residual_config["residual_lr"],
                "name": "residual",
            }
        )
        logger.info(
            "Residual optimizer groups: %s",
            [(group["name"], group["target_lr"]) for group in groups],
        )
        # The inherited training loop logs ``self.lr`` when warm-up finishes.
        self.lr = {group["name"]: group["target_lr"] for group in groups}
        return torch.optim.AdamW(
            groups,
            weight_decay=self.optim_config.weight_decay,
        )

    def _forward_ist_with_features(self, batch):
        src_features = self.ist_net.forward_by_chunk(batch.src_img)
        tar_features = self.ist_net.forward_by_chunk(batch.tar_img)
        pair_valid = (
            (batch.src_pts[..., 0] >= 0)
            & (batch.src_pts[..., 1] >= 0)
            & (batch.tar_pts[..., 0] >= 0)
            & (batch.tar_pts[..., 1] >= 0)
        )
        indices = pair_valid.nonzero(as_tuple=False)
        instance_ids = indices[:, 0]
        src_xy = batch.src_pts[pair_valid].long()
        tar_xy = batch.tar_pts[pair_valid].long()
        src_selected = src_features[
            instance_ids, :, src_xy[:, 1], src_xy[:, 0]
        ]
        tar_selected = tar_features[
            instance_ids, :, tar_xy[:, 1], tar_xy[:, 0]
        ]
        features = torch.cat([tar_selected, src_selected], dim=1)
        scale = self.ist_net.regressor.scale_predictor(features).squeeze(1)
        inplane = self.ist_net.regressor.inplane_predictor(features)
        if self.ist_net.regressor.normalize_output:
            inplane = F.normalize(inplane, dim=1)
        return {"scale": scale, "inplane": inplane}, src_features, tar_features

    def _base_pose_loss(self, outputs):
        config = self.pose_loss_config
        total = (
            config["log_depth_weight"] * outputs.log_depth
            + config["instance_log_scale_weight"] * outputs.instance_log_scale
            + config["scale_consistency_weight"] * outputs.scale_consistency
            + config["inplane_weight"] * outputs.inplane
            + config["reprojection_weight"] * outputs.reprojection
            + config["anti_flip_weight"] * outputs.anti_flip
        )
        direct_total = (
            config["direct_translation_weight"] * outputs.direct_translation
            + config["direct_depth_weight"] * outputs.direct_depth
            + config["direct_rotation_weight"] * outputs.direct_rotation
            + config["direct_reprojection_weight"] * outputs.direct_reprojection
        )
        if config["optimize_pose_monitor_errors"]:
            total = total + direct_total
        return total, direct_total

    def compute_pose_aware_ist_objective(self, batch, split: str):
        batch_size = self._batch_size(batch)
        predictions, src_features, tar_features = self._forward_ist_with_features(
            batch
        )
        self._attach_ist_predictions_for_visualization(batch, predictions)
        base_outputs = pose_aware_ist_losses(
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
        base_total, direct_total = self._base_pose_loss(base_outputs)

        baseline = aggregate_ist_pose(
            pred_scale=predictions["scale"],
            pred_inplane=predictions["inplane"],
            src_pts=batch.src_pts,
            tar_pts=batch.tar_pts,
            src_pose=batch.src_pose,
            src_K=batch.src_K,
            tar_K=batch.tar_K,
            src_M=batch.src_M,
            tar_M=batch.tar_M,
            patch_size=int(self.ist_net.patch_size),
        )
        descriptor, descriptor_valid = paired_instance_descriptor(
            src_features=src_features,
            tar_features=tar_features,
            src_pts=batch.src_pts,
            tar_pts=batch.tar_pts,
            baseline_pose=baseline.pose,
            tar_K=batch.tar_K,
            tar_M=batch.tar_M,
            translation_unit=self.residual_config["translation_unit"],
            patch_size=int(self.ist_net.patch_size),
        )
        residual = self.pose_residual_head(descriptor)
        valid = baseline.valid_instances & descriptor_valid
        raw_translation = torch.where(
            valid[:, None],
            residual["translation"],
            torch.zeros_like(residual["translation"]),
        )
        raw_rotation = torch.where(
            valid[:, None],
            residual["rotation"],
            torch.zeros_like(residual["rotation"]),
        )
        refined = apply_pose_residual(
            baseline_pose=baseline.pose,
            tar_K=batch.tar_K,
            tar_M=batch.tar_M,
            raw_translation=raw_translation,
            raw_rotation=raw_rotation,
            max_center_offset_px=self.residual_config["max_center_offset_px"],
            max_log_depth_residual=self.residual_config[
                "max_log_depth_residual"
            ],
            max_rotation_rad=np.deg2rad(
                self.residual_config["max_rotation_deg"]
            ),
            enable_rotation=self.residual_config["enable_rotation"],
        )
        crop_diagonal = (
            float(batch.tar_img.shape[-2] ** 2 + batch.tar_img.shape[-1] ** 2)
            ** 0.5
        )
        residual_outputs = residual_pose_losses(
            baseline_pose=baseline.pose,
            refined_pose=refined.pose,
            target_pose=batch.tar_pose,
            target_K=batch.tar_K,
            target_M=batch.tar_M,
            valid=valid,
            delta_uv_px=refined.delta_uv_px,
            delta_log_depth=refined.delta_log_depth,
            delta_rotvec=refined.delta_rotvec,
            crop_diagonal_px=crop_diagonal,
            translation_scale=self.residual_config["translation_unit"],
            center_beta=self.residual_config["center_beta"],
            log_depth_beta=self.residual_config["log_depth_beta"],
            translation_beta=self.residual_config["translation_beta"],
            rotation_beta=self.residual_config["rotation_beta"],
        )
        residual_total = (
            self.residual_config["center_weight"] * residual_outputs.center
            + self.residual_config["log_depth_weight"]
            * residual_outputs.log_depth
            + self.residual_config["translation_weight"]
            * residual_outputs.translation
            + self.residual_config["regularization_weight"]
            * residual_outputs.regularization
        )
        if self.residual_config["enable_rotation"]:
            residual_total = (
                residual_total
                + self.residual_config["rotation_weight"]
                * residual_outputs.rotation
            )
        total = base_total + residual_total

        to_mm = self.residual_config["translation_to_mm"]
        baseline_translation_error_mm = (
            residual_outputs.baseline_translation_error * to_mm
        )
        refined_translation_error_mm = (
            residual_outputs.refined_translation_error * to_mm
        )
        baseline_pose_score = pose_score(
            baseline_translation_error_mm,
            residual_outputs.baseline_rotation_error_deg,
        )
        refined_pose_score = pose_score(
            refined_translation_error_mm,
            residual_outputs.refined_rotation_error_deg,
        )
        self._log_metrics(
            split,
            {
                "loss_log_depth": base_outputs.log_depth,
                "loss_instance_log_scale": base_outputs.instance_log_scale,
                "loss_scale_consistency": base_outputs.scale_consistency,
                "loss_inplane": base_outputs.inplane,
                "loss_reprojection": base_outputs.reprojection,
                "loss_direct_pose_total": direct_total,
                "loss_ist_pose_aware": base_total,
                "loss_residual_center": residual_outputs.center,
                "loss_residual_log_depth": residual_outputs.log_depth,
                "loss_residual_translation": residual_outputs.translation,
                "loss_residual_rotation": residual_outputs.rotation,
                "loss_residual_regularization": residual_outputs.regularization,
                "loss_pose_residual_total": residual_total,
                "loss_combined": total,
                "monitor_baseline_translation_error_mm": (
                    baseline_translation_error_mm
                ).detach(),
                "monitor_refined_translation_error_mm": (
                    refined_translation_error_mm
                ).detach(),
                "monitor_baseline_depth_error_mm": (
                    residual_outputs.baseline_depth_error * to_mm
                ).detach(),
                "monitor_refined_depth_error_mm": (
                    residual_outputs.refined_depth_error * to_mm
                ).detach(),
                "monitor_baseline_rotation_error_deg": (
                    residual_outputs.baseline_rotation_error_deg.detach()
                ),
                "monitor_refined_rotation_error_deg": (
                    residual_outputs.refined_rotation_error_deg.detach()
                ),
                "monitor_baseline_pose_score": baseline_pose_score.detach(),
                "monitor_refined_pose_score": refined_pose_score.detach(),
                "monitor_baseline_center_error_px": (
                    residual_outputs.baseline_center_error_px.detach()
                ),
                "monitor_refined_center_error_px": (
                    residual_outputs.refined_center_error_px.detach()
                ),
                "monitor_translation_improvement_mm": (
                    (
                        residual_outputs.baseline_translation_error
                        - residual_outputs.refined_translation_error
                    )
                    * to_mm
                ).detach(),
                "monitor_depth_improvement_mm": (
                    (
                        residual_outputs.baseline_depth_error
                        - residual_outputs.refined_depth_error
                    )
                    * to_mm
                ).detach(),
                "monitor_center_improvement_px": (
                    residual_outputs.baseline_center_error_px
                    - residual_outputs.refined_center_error_px
                ).detach(),
                "monitor_center_offset_px": (
                    residual_outputs.center_offset_px.detach()
                ),
                "monitor_abs_log_depth_residual": (
                    residual_outputs.abs_log_depth_residual.detach()
                ),
                "monitor_rotation_residual_deg": (
                    residual_outputs.rotation_residual_deg.detach()
                ),
                "monitor_scale_abs_log_error": (
                    base_outputs.scale_abs_log_error.detach()
                ),
                "monitor_scale_median_pred_gt_ratio": (
                    base_outputs.scale_median_ratio.detach()
                ),
                "valid_instances": valid.sum().to(total.dtype).detach(),
                "valid_patch_pairs": base_outputs.valid_patch_pairs.detach(),
            },
            prog_bar=(
                "loss_pose_residual_total",
                "monitor_refined_translation_error_mm",
                "monitor_refined_depth_error_mm",
                "monitor_refined_rotation_error_deg",
                "monitor_refined_pose_score",
            ),
            batch_size=batch_size,
        )
        setattr(batch, "residual_baseline_pose", baseline.pose.detach())
        setattr(batch, "residual_refined_pose", refined.pose.detach())
        return total

    def _refine_inference_predictions(self, batch, predictions, dataset_name):
        if not self.residual_config["apply_at_inference"]:
            return predictions
        template_data = self.template_datas[dataset_name]
        device = batch.tar_img.device
        batch_size, hypotheses = predictions.id_src.shape[:2]
        labels = torch.as_tensor(
            np.asarray(batch.infos.label).astype(np.int64), device=device
        )
        target_features = self.ist_net.forward_by_chunk(batch.tar_img)
        sample_ids = torch.arange(batch_size, device=device)
        refined_poses = predictions.pred_poses.clone()
        for hypothesis in range(hypotheses):
            view_ids = predictions.id_src[:, hypothesis]
            source_features = template_data.ist_features[labels - 1][
                sample_ids, view_ids
            ]
            baseline_pose = predictions.pred_poses[:, hypothesis]
            descriptor, valid = paired_instance_descriptor(
                src_features=source_features,
                tar_features=target_features,
                src_pts=predictions.ransac_src_pts[:, hypothesis],
                tar_pts=predictions.ransac_tar_pts[:, hypothesis],
                baseline_pose=baseline_pose,
                tar_K=batch.tar_K,
                tar_M=batch.tar_M,
                translation_unit=self.residual_config["translation_unit"],
                patch_size=int(self.ist_net.patch_size),
            )
            residual = self.pose_residual_head(descriptor)
            raw_translation = torch.where(
                valid[:, None],
                residual["translation"],
                torch.zeros_like(residual["translation"]),
            )
            raw_rotation = torch.where(
                valid[:, None],
                residual["rotation"],
                torch.zeros_like(residual["rotation"]),
            )
            refined = apply_pose_residual(
                baseline_pose=baseline_pose,
                tar_K=batch.tar_K,
                tar_M=batch.tar_M,
                raw_translation=raw_translation,
                raw_rotation=raw_rotation,
                max_center_offset_px=self.residual_config[
                    "max_center_offset_px"
                ],
                max_log_depth_residual=self.residual_config[
                    "max_log_depth_residual"
                ],
                max_rotation_rad=np.deg2rad(
                    self.residual_config["max_rotation_deg"]
                ),
                enable_rotation=self.residual_config["enable_rotation"],
            )
            refined_poses[:, hypothesis] = refined.pose
        predictions.register_tensor("pred_poses", refined_poses)
        return predictions

    @torch.no_grad()
    def eval_retrieval(
        self,
        batch,
        idx_batch,
        dataset_name,
        sort_pred_by_inliers=True,
        save_outputs=True,
        log_retrieval=True,
    ):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        predictions = super().eval_retrieval(
            batch=batch,
            idx_batch=idx_batch,
            dataset_name=dataset_name,
            sort_pred_by_inliers=sort_pred_by_inliers,
            save_outputs=False,
            log_retrieval=log_retrieval,
        )
        predictions = self._refine_inference_predictions(
            batch, predictions, dataset_name
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if save_outputs:
            save_path = osp.join(self.log_dir, "predictions", f"{idx_batch}.npz")
            _, predictions = self.filter_and_save(
                predictions,
                test_list=batch.test_list,
                time=elapsed,
                save_path=save_path,
            )
        return predictions
