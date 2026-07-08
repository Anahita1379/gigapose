import os
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from einops import repeat
import pytorch_lightning as pl
from tqdm import tqdm
import pandas as pd
from src.utils.logging import get_logger, log_image
from src.utils.batch import BatchedData, gather
from src.utils.optimizer import HybridOptim
from torchvision.utils import save_image
from src.utils.time import Timer
from src.models.loss import cosine_similarity
from src.lib3d.torch import (
    cosSin,
    get_relative_scale_inplane,
    geodesic_distance,
)
from src.libVis.torch import (
    plot_Kabsch,
    plot_ist_alignment_batch,
    plot_keypoints_batch,
    save_tensor_to_image,
)
from src.models.poses import ObjectPoseRecovery
import src.megapose.utils.tensor_collection as tc
from src.utils.inout import save_predictions_from_batched_predictions

logger = get_logger(__name__)


class GigaPose(pl.LightningModule):
    def __init__(
        self,
        model_name,
        ae_net,
        ist_net,
        training_loss,
        testing_metric,
        optim_config,
        log_interval,
        log_dir,
        max_num_dets_per_forward=None,
        **kwargs,
    ):
        # define the network
        super().__init__()
        self.model_name = model_name
        self.ae_net = ae_net
        self.ist_net = ist_net
        self.training_loss = training_loss
        self.testing_metric = testing_metric

        self.max_num_dets_per_forward = max_num_dets_per_forward

        self.log_interval = log_interval
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(osp.join(self.log_dir, "predictions"), exist_ok=True)

        self.optim_config = optim_config
        self.optim_name = "AdamW"

        # for testing
        self.template_datas = {}
        self.pose_recovery = {}
        self.l2_loss = nn.MSELoss()
        self.timer = Timer()
        self.run_id = None
        self.template_datasets = None
        self.test_dataset_name = None

        logger.info("Initialize GigaPose done!")

    def warm_up_lr(self):
        for optim in self.trainer.optimizers:
            for idx_group, pg in enumerate(optim.param_groups):
                if len(optim.param_groups) > 1:
                    lr = self.lr["ae"] if idx_group == 0 else self.lr["ist"]
                    pg["lr"] = (
                        self.global_step / float(self.optim_config.warm_up_steps) * lr
                    )
                else:
                    pg["lr"] = (
                        self.global_step
                        / float(self.optim_config.warm_up_steps)
                        * self.lr
                    )
            if self.global_step % 50 == 0:
                logger.info(f"Step={self.global_step}, lr warm up: lr={pg['lr']}")

    def configure_optimizers(self):
        # define optimizer
        if self.optim_config.nets_to_train in ["ae", "all"]:
            logger.info("Optimizer for ae net")
            ae_optimizer = torch.optim.AdamW(
                self.ae_net.get_toUpdate_parameters(),
                self.optim_config.ae_lr,
                weight_decay=self.optim_config.weight_decay,
            )
            self.lr = self.optim_config.ae_lr

        if self.optim_config.nets_to_train in ["ist", "all"]:
            logger.info("Optimizer for ist net")
            ist_optimizer = torch.optim.AdamW(
                self.ist_net.parameters(),
                self.optim_config.ist_lr,
                weight_decay=self.optim_config.weight_decay,
            )
            self.lr = self.optim_config.ist_lr

        # disable gradient for non-updatable parameters
        if self.optim_config.nets_to_train != "all":
            if self.optim_config.nets_to_train == "ae":
                for param in self.ist_net.parameters():
                    param.requires_grad = False
            if self.optim_config.nets_to_train == "ist":
                for param in self.ae_net.parameters():
                    param.requires_grad = False
        else:
            self.lr = {
                "ae": self.optim_config.ae_lr,
                "ist": self.optim_config.ist_lr,
            }
        # combine optimizers
        if self.optim_config.nets_to_train == "all":
            optimizer = HybridOptim([ae_optimizer, ist_optimizer])
        elif self.optim_config.nets_to_train == "ae":
            optimizer = ae_optimizer
        elif self.optim_config.nets_to_train == "ist":
            optimizer = ist_optimizer
        else:
            raise NotImplementedError
        assert optimizer is not None
        return optimizer

    def move_to_device(self):
        self.ae_net.to(self.device)
        self.ist_net.to(self.device)
        logger.info(f"Moving models to {self.device} done!")

    def compute_contrastive_loss(self, batch, split):
        """
        Contrastive loss based on corresponding patches
        - Positive are patches from the same correspondence
        - Negative are patches from different correspondences
        """
        device = batch.src_img.device
        # get the query and ref features
        src_feat = self.ae_net(batch.src_img)
        tar_feat = self.ae_net(batch.tar_img)
        src_pts = getattr(batch, "src_pts").clone().long()
        tar_pts = getattr(batch, "tar_pts").clone().long()

        loss = {}
        # select the corresponding patches
        src_feat_ = gather(src_feat, src_pts)
        tar_feat_ = gather(tar_feat, tar_pts)
        label = torch.arange(src_feat_.shape[0], dtype=torch.long).to(device)
        loss["infoNCE"] = self.training_loss.contrast_loss(
            src_feat_,
            tar_feat_,
            label,
        )

        # monitor the similarity between query and ref
        with torch.no_grad():
            pos_sim = F.cosine_similarity(src_feat_, tar_feat_, dim=1, eps=1e-8)
            loss["pos_sim"] = pos_sim.mean()

            neg_sim = cosine_similarity(src_feat_, tar_feat_, normalize=True)
            loss["neg_sim"] = neg_sim.mean()

        for metric_name, metric_value in loss.items():
            name = f"{split}/{metric_name}"
            if metric_name == "infoNCE":
                prog_bar = True
            else:
                prog_bar = False
            self.log(
                name,
                metric_value,
                sync_dist=True,
                on_step=True,
                on_epoch=False,
                prog_bar=prog_bar,
            )
        return loss

    def compute_regression_loss(self, batch, split):
        """
        Contrastive loss based on corresponding patches
        - Positive are patches from the same correspondence
        - Negative are patches from different correspondences
        """
        # get the query and ref features
        num_patches = batch.src_pts.shape[1]
        H, W = np.sqrt(num_patches).astype(int), np.sqrt(num_patches).astype(int)

        loss = {}
        gt_relInplane = getattr(batch, "relInplane")
        gt_relScale = getattr(batch, "relScale")

        src_pts = getattr(batch, "src_pts").clone().long()
        tar_pts = getattr(batch, "tar_pts").clone().long()

        preds = self.ist_net(
            src_img=batch.src_img,
            tar_img=batch.tar_img,
            src_pts=src_pts,
            tar_pts=tar_pts,
        )
        src_patch_valid = torch.logical_and(src_pts[:, :, 0] != -1, src_pts[:, :, 1] != -1)
        tar_patch_valid = torch.logical_and(tar_pts[:, :, 0] != -1, tar_pts[:, :, 1] != -1)
        pair_patch_valid = torch.logical_and(src_patch_valid, tar_patch_valid)
        pred_rel_scale = torch.full(
            pair_patch_valid.shape,
            float("nan"),
            dtype=preds["scale"].dtype,
            device=preds["scale"].device,
        )
        pred_rel_inplane = torch.full(
            (*pair_patch_valid.shape, 2),
            float("nan"),
            dtype=preds["inplane"].dtype,
            device=preds["inplane"].device,
        )
        if preds["scale"].shape[0] == int(pair_patch_valid.sum()):
            pred_rel_scale[pair_patch_valid] = preds["scale"]
            pred_rel_inplane[pair_patch_valid] = preds["inplane"]
        setattr(batch, "pred_relScale", pred_rel_scale)
        setattr(batch, "pred_relInplane", pred_rel_inplane)
        num_patch_pairs = pair_patch_valid.sum()
        self.log(
            f"{split}/valid_patch_pairs",
            num_patch_pairs.float(),
            sync_dist=True,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
        if preds["inplane"].shape[0] != src_pts.shape[0]:
            gt_relInplane = repeat(gt_relInplane, "b -> b 1 H W", H=H, W=W)
            gt_relInplane = gather(gt_relInplane, src_pts).squeeze(1)

            gt_relScale = repeat(gt_relScale, "b -> b 1 H W", H=H, W=W)
            gt_relScale = gather(gt_relScale, src_pts).squeeze(1)

        if preds["inplane"].shape[0] == 0 or gt_relScale.numel() == 0:
            # Keep the zero loss connected to trainable IST parameters so
            # Lightning/DDP can still run backward on an all-skipped batch.
            zero = next(self.ist_net.parameters()).sum() * 0.0
            for metric_name in ("inp", "scale", "scale_err", "angle_err"):
                self.log(
                    f"{split}/{metric_name}",
                    zero.detach() if metric_name.endswith("err") else zero,
                    sync_dist=True,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=metric_name in ["inp", "scale"],
                )
            self.log(
                f"{split}/valid_regression_fraction",
                zero.detach(),
                sync_dist=True,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )
            self.log(
                f"{split}/invalid_regression_fraction",
                torch.ones_like(zero).detach(),
                sync_dist=True,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )
            logger.info(
                "Skipping %s regression batch with zero valid patch correspondences",
                split,
            )
            return {
                "inp": zero,
                "scale": zero,
                "scale_err": zero.detach(),
                "angle_err": zero.detach(),
            }

        gt_inplane_finite = torch.isfinite(gt_relInplane)
        gt_scale_finite = torch.isfinite(gt_relScale)
        gt_scale_positive = gt_relScale > 0
        pred_inplane_finite = torch.isfinite(preds["inplane"]).all(dim=1)
        pred_scale_finite = torch.isfinite(preds["scale"])
        if pred_scale_finite.ndim > 1:
            pred_scale_finite = pred_scale_finite.flatten(start_dim=1).all(dim=1)
        valid = (
            gt_inplane_finite
            & gt_scale_finite
            & gt_scale_positive
            & pred_inplane_finite
            & pred_scale_finite
        )
        valid_fraction = valid.float().mean()
        invalid_fraction = 1.0 - valid_fraction
        self.log(
            f"{split}/valid_regression_fraction",
            valid_fraction,
            sync_dist=True,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
        self.log(
            f"{split}/invalid_regression_fraction",
            invalid_fraction,
            sync_dist=True,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
        if not valid.all():
            num_invalid = int((~valid).sum().detach().cpu())
            total = int(valid.numel())
            reason_counts = {
                "gt_inplane_nan": int((~gt_inplane_finite).sum().detach().cpu()),
                "gt_scale_nan": int((~gt_scale_finite).sum().detach().cpu()),
                "gt_scale_nonpositive": int((gt_scale_finite & ~gt_scale_positive).sum().detach().cpu()),
                "pred_inplane_nan": int((~pred_inplane_finite).sum().detach().cpu()),
                "pred_scale_nan": int((~pred_scale_finite).sum().detach().cpu()),
            }
            logger.info(
                "Skipping %d/%d non-finite or invalid %s regression targets: %s",
                num_invalid,
                total,
                split,
                reason_counts,
            )
            gt_relInplane = gt_relInplane[valid]
            gt_relScale = gt_relScale[valid]
            preds["inplane"] = preds["inplane"][valid]
            preds["scale"] = preds["scale"][valid]

        if gt_relScale.numel() == 0:
            zero = (preds["inplane"].sum() + preds["scale"].sum()) * 0.0
            loss["inp"] = zero
            loss["scale"] = zero
            loss["scale_err"] = zero.detach()
            loss["angle_err"] = zero.detach()
            for metric_name, metric_value in loss.items():
                self.log(
                    f"{split}/{metric_name}",
                    metric_value,
                    sync_dist=True,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=metric_name in ["inp", "scale"],
                )
            return loss

        # it is simpler to use l2 loss for warm up to regress correct magnitudes
        if self.trainer.global_step < self.optim_config.warm_up_steps:
            loss["inp"] = self.l2_loss(
                preds["inplane"],
                cosSin(gt_relInplane),
            )
            loss["scale"] = self.l2_loss(preds["scale"], gt_relScale)
        else:
            loss["inp"] = self.training_loss.inplane_loss(
                preds["inplane"],
                cosSin(gt_relInplane),
            )
            loss["scale"] = self.training_loss.scale_loss(preds["scale"], gt_relScale)

        # Visualize the predicted inplane and scale
        with torch.no_grad():
            scale_err = torch.abs(preds["scale"].clone() - gt_relScale)
            loss["scale_err"] = scale_err.mean()

            angle_err = geodesic_distance(preds["inplane"], cosSin(gt_relInplane))
            loss["angle_err"] = torch.rad2deg(angle_err)

        for metric_name, metric_value in loss.items():
            name = f"{split}/{metric_name}"
            if metric_name in ["inp", "scale"]:
                prog_bar = True
            else:
                prog_bar = False
            self.log(
                name,
                metric_value,
                sync_dist=True,
                on_step=True,
                on_epoch=False,
                prog_bar=prog_bar,
            )
        return loss

    def training_step(self, batchs, idx_batch):
        if self.trainer.global_step < self.optim_config.warm_up_steps:
            self.warm_up_lr()
        elif self.trainer.global_step == self.optim_config.warm_up_steps:
            logger.info(f"Finished warm up, setting lr to {self.lr}")

        loss = 0
        times = {}

        for idx_dataset, batch in enumerate(batchs):
            if batch is None:
                continue
            if idx_batch % self.log_interval == 0:
                vis_pts = plot_keypoints_batch(batch)
                sample_path = f"{self.log_dir}/sample_rank{self.global_rank}.png"
                save_tensor_to_image(vis_pts, sample_path)
                log_image(
                    logger=self.logger,
                    name=f"vis/train_samples_{idx_dataset}",
                    path=sample_path,
                    step=int(self.global_step),
                )

            if self.optim_config.nets_to_train in ["ist", "all"]:
                self.timer.tic()
                loss_ = self.compute_regression_loss(batch, "train")
                loss += loss_["scale"] + loss_["inp"]
                times[f"scale_inp_{idx_dataset}"] = self.timer.toc()

            if self.optim_config.nets_to_train in ["ae", "all"]:
                self.timer.tic()
                loss_ = self.compute_contrastive_loss(batch, "train")
                loss += loss_["infoNCE"]
                times[f"infoNCE_{idx_dataset}"] = self.timer.toc()

        for time_name, time_value in times.items():
            self.log(
                time_name,
                time_value,
                sync_dist=True,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )

        self.log(
            "total",
            loss,
            sync_dist=True,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log(
            "train/loss",
            loss,
            sync_dist=True,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
        return loss

    def log_validation_keypoints(self, batch, idx_batch, split, type_data="gt"):
        if idx_batch != 0 and idx_batch % self.log_interval != 0:
            return
        vis_pts = plot_keypoints_batch(batch, type_data=type_data)
        image_dir = osp.join(self.log_dir, "validation_images")
        os.makedirs(image_dir, exist_ok=True)
        sample_path = osp.join(
            image_dir,
            f"{split}_{type_data}_step{int(self.global_step):06d}_batch{idx_batch:04d}_rank{self.global_rank}.png",
        )
        save_tensor_to_image(vis_pts, sample_path)
        log_image(
            logger=self.logger,
            name=f"vis/{split}_{type_data}_samples",
            path=sample_path,
            step=int(self.global_step),
        )

    def log_validation_ist_overlay(self, batch, idx_batch, split):
        if idx_batch != 0 and idx_batch % self.log_interval != 0:
            return
        vis_overlay = plot_ist_alignment_batch(batch)
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

    def validate_contrast_loss(self, batch, idx_batch, split):
        src_feat = self.ae_net(batch.src_img)
        tar_feat = self.ae_net(batch.tar_img)

        preds = self.testing_metric.val(
            src_feat=src_feat,
            tar_feat=tar_feat,
            src_mask=batch.src_mask,
            tar_mask=batch.tar_mask,
        )
        setattr(batch, "pred_src_pts", preds.src_pts)
        setattr(batch, "pred_tar_pts", preds.tar_pts)

        pred_valid = torch.logical_and(
            batch.pred_src_pts[:, :, 0] != -1,
            batch.pred_tar_pts[:, :, 0] != -1,
        )
        pred_match_counts = pred_valid.sum(dim=1).float()
        self.log(
            f"{split}/pred_matches",
            pred_match_counts.mean(),
            sync_dist=True,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        if hasattr(preds, "score"):
            self.log(
                f"{split}/match_score_max",
                preds.score.max(dim=1).values.mean(),
                sync_dist=True,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )
            self.log(
                f"{split}/match_score_mean",
                preds.score.mean(),
                sync_dist=True,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )

        # monitor the distance between the gt and the predicted matches for same target patches
        mask = torch.logical_and(
            batch.tar_pts[:, :, 1] != -1, batch.pred_tar_pts[:, :, 1] != -1
        )
        if mask.any():
            distance = (batch.tar_pts[mask] - batch.pred_tar_pts[mask]).norm(dim=1).mean()
        else:
            distance = batch.tar_pts.new_tensor(float("nan"))
        self.log(
            f"{split}/matching",
            distance,
            sync_dist=True,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        self.log_validation_keypoints(batch, idx_batch, split, type_data="pred")

    def validation_step(self, batch, idx_batch):
        if batch is None:
            return None
        loss = 0
        if self.optim_config.nets_to_train in ["ist", "all"]:
            loss_ = self.compute_regression_loss(batch, "val")
            loss += loss_["scale"] + loss_["inp"]
            self.log_validation_keypoints(batch, idx_batch, "val", type_data="gt")
            self.log_validation_ist_overlay(batch, idx_batch, "val")
        if self.optim_config.nets_to_train in ["ae", "all"]:
            _ = self.validate_contrast_loss(batch, idx_batch, "val")
        self.log_heavy_validation_cad_overlay(batch, idx_batch)
        self.log(
            "val/loss",
            loss,
            sync_dist=True,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return loss

    @torch.no_grad()
    def log_heavy_validation_cad_overlay(self, batch, idx_batch):
        """Run sparse full inference and render its top CAD pose on real RGB."""
        if not getattr(self, "heavy_validation_enabled", False):
            return
        interval = int(getattr(self, "heavy_validation_interval", 1000))
        step = int(self.global_step)
        if interval <= 0 or step == 0 or step % interval != 0:
            return
        if idx_batch != 0 or self.global_rank != 0:
            return

        max_images = max(1, int(getattr(self, "heavy_validation_images", 4)))
        selected_indices = []
        selected_frames = set()
        for index, row in batch.infos.reset_index(drop=True).iterrows():
            frame_key = (
                int(row.get("scene_id", -1)),
                int(row.get("view_id", row.get("im_id", -1))),
            )
            if frame_key in selected_frames:
                continue
            selected_frames.add(frame_key)
            selected_indices.append(index)
            if len(selected_indices) >= max_images:
                break
        if not selected_indices:
            return
        batch = batch[np.asarray(selected_indices, dtype=np.int64)]
        # IST's template backbone is trainable in IST-only runs. Do not reuse
        # template features cached by an earlier heavy-validation step.
        self.template_datas.pop(self.heavy_validation_dataset_name, None)
        self.pose_recovery.pop(self.heavy_validation_dataset_name, None)
        predictions = self.eval_retrieval(
            batch,
            idx_batch=idx_batch,
            dataset_name=self.heavy_validation_dataset_name,
            save_outputs=False,
            log_retrieval=False,
        )

        # Keep EGL/CAD dependencies out of the frequent lightweight path.
        import trimesh
        from PIL import Image, ImageDraw
        from fine_tuning.ac_geometry import InstanceRenderer, bbox_from_mask

        mesh = trimesh.load(self.heavy_validation_mesh_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
            raise ValueError(
                f"Could not load heavy-validation CAD: "
                f"{self.heavy_validation_mesh_path}"
            )
        translation_scale = (
            0.001 if 0.01 < float(np.linalg.norm(mesh.extents)) < 100.0 else 1.0
        )
        renderer = InstanceRenderer(mesh)
        panels = []
        try:
            count = batch.tar_full_img.shape[0]
            for index in range(count):
                rgb = (
                    batch.tar_full_img[index]
                    .detach()
                    .cpu()
                    .clamp(0, 1)
                    .permute(1, 2, 0)
                    .numpy()
                )
                image_height, image_width = (
                    batch.tar_image_size[index].detach().cpu().numpy().astype(int)
                )
                image_top, image_left = (
                    batch.tar_image_offset[index]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(int)
                )
                padded_image = Image.fromarray(np.uint8(rgb * 255))
                pose = predictions.pred_poses[index, 0].detach().cpu().numpy().copy()
                pose[:3, 3] *= translation_scale
                K = batch.tar_K[index].detach().cpu().numpy()
                segmentation, _ = renderer.render(
                    [pose],
                    K,
                    width=padded_image.width,
                    height=padded_image.height,
                )
                image = padded_image.crop(
                    (
                        image_left,
                        image_top,
                        image_left + image_width,
                        image_top + image_height,
                    )
                )
                mask = (
                    segmentation[
                        image_top : image_top + image_height,
                        image_left : image_left + image_width,
                    ]
                    == 1
                )
                base = np.asarray(image, dtype=np.float32).copy()
                base[mask] = (
                    0.58 * base[mask]
                    + 0.42 * np.array([0, 255, 80], dtype=np.float32)
                )
                overlay = Image.fromarray(np.uint8(np.clip(base, 0, 255)))
                draw = ImageDraw.Draw(overlay)
                x, y, width, height = bbox_from_mask(mask)
                if width > 0 and height > 0:
                    draw.rectangle(
                        (x, y, x + width - 1, y + height - 1),
                        outline=(0, 255, 80),
                        width=3,
                    )
                panel = Image.new("RGB", (image.width * 2, image.height))
                panel.paste(image, (0, 0))
                panel.paste(overlay, (image.width, 0))
                panels.append(
                    torch.from_numpy(np.asarray(panel).copy())
                    .permute(2, 0, 1)
                    .float()
                    / 255.0
                )
        finally:
            renderer.close()

        if panels:
            image_dir = osp.join(self.log_dir, "validation_images")
            os.makedirs(image_dir, exist_ok=True)
            sample_path = osp.join(
                image_dir,
                f"val_heavy_cad_step{int(self.global_step):06d}_"
                f"rank{self.global_rank}.png",
            )
            save_tensor_to_image(torch.stack(panels), sample_path, nrow=1)
            log_image(
                logger=self.logger,
                name="vis/val_heavy_cad_overlay",
                path=sample_path,
                step=int(self.global_step),
            )

    def set_template_data(self, dataset_name):
        logger.info("Initializing template data ...")
        self.timer.tic()
        template_dataset = self.template_datasets[dataset_name]
        names = ["rgb", "mask", "K", "M", "poses", "ae_features", "ist_features"]
        template_data = {name: BatchedData(None) for name in names}

        for idx in tqdm(range(len(template_dataset))):
            for name in names:
                if name in ["ae_features", "ist_features"]:
                    continue
                if name == "rgb":
                    templates = template_dataset[idx].rgb.to(self.device)
                    if self.max_num_dets_per_forward is None:
                        template_data[name].append(templates)

                    ae_features = self.ae_net(templates)
                    template_data["ae_features"].append(ae_features)

                    ist_features = self.ist_net.forward_by_chunk(templates)
                    template_data["ist_features"].append(ist_features)
                else:
                    tmp = getattr(template_dataset[idx], name)
                    template_data[name].append(tmp.to(self.device))
        if self.max_num_dets_per_forward is not None:
            names.remove("rgb")
        for name in names:
            template_data[name].stack()
            template_data[name] = template_data[name].data

        self.template_datas[dataset_name] = tc.PandasTensorCollection(
            infos=pd.DataFrame(), **template_data
        )
        self.pose_recovery[dataset_name] = ObjectPoseRecovery(
            template_K=template_data["K"],
            template_Ms=template_data["M"],
            template_poses=template_data["poses"],
        )
        num_obj = len(template_data["K"])
        onboarding_time = self.timer.toc() / num_obj
        self.timer.reset()
        logger.info(f"Init {dataset_name} done! Avg time={onboarding_time} s/object")

    def filter_and_save(
        self,
        predictions,
        test_list,
        time,
        save_path,
        keep_only_testing_instances=True,
    ):
        labels = np.asarray(predictions.infos.label).astype(np.int32)
        assert len(np.unique(labels)) == len(np.unique(test_list.infos.obj_id))
        detection_times = []
        if keep_only_testing_instances:
            selected_idxs = []
            for idx, id in enumerate(test_list.infos.obj_id):
                num_inst = test_list.infos.inst_count[idx]
                idx_inst = labels == id
                pred_inst = predictions[idx_inst.tolist()]

                selected_idx = torch.argsort(pred_inst.scores[:, 0], descending=True)
                selected_idx = selected_idx[:num_inst].cpu().numpy()
                selected_idx = np.arange(len(labels))[idx_inst][selected_idx]
                selected_idxs.extend(selected_idx.tolist())
                detection_times.extend(
                    [test_list.infos.detection_time[idx] for _ in range(num_inst)]
                )
        predictions = predictions[selected_idxs]

        detection_times = np.array(detection_times)
        detection_times = torch.from_numpy(detection_times).to(
            predictions.scores.device
        )
        time = torch.ones_like(detection_times) * time
        predictions.register_tensor("detection_time", detection_times)
        predictions.register_tensor("time", time)

        scene_id = np.asarray(predictions.infos.scene_id).astype(np.int32)
        im_id = np.asarray(predictions.infos.view_id).astype(np.int32)
        label = np.asarray(predictions.infos.label).astype(np.int32)

        np.savez(
            save_path,
            scene_id=scene_id,
            im_id=im_id,
            object_id=label,
            time=predictions.time.cpu().numpy(),
            detection_time=predictions.detection_time.cpu().numpy(),
            poses=predictions.pred_poses.cpu().numpy(),
            scores=predictions.scores.cpu().numpy(),
        )
        return selected_idxs, predictions

    def vis_retrieval(
        self, template_data, batch, selected_idxs, predictions, idx_batch
    ):
        device = template_data.rgb.device
        idx_sample = torch.arange(0, predictions.id_src.shape[0], device=device)
        tar_label_np = np.asarray(predictions.infos.label).astype(np.int32)
        tar_label = torch.from_numpy(tar_label_np).to(device)

        src_imgs = template_data.rgb[tar_label - 1]
        src_masks = template_data.mask[tar_label - 1]
        tar_img = batch.tar_img[selected_idxs]
        tar_mask = batch.tar_mask[selected_idxs]
        pred_imgs = []
        for idx_k in range(self.testing_metric.k):
            batch = tc.PandasTensorCollection(
                infos=pd.DataFrame(),
                src_img=src_imgs[idx_sample, predictions.id_src[:, idx_k]].clone(),
                src_mask=src_masks[idx_sample, predictions.id_src[:, idx_k]].clone(),
                tar_img=tar_img,
                tar_mask=tar_mask,
                src_pts=predictions.ransac_src_pts[:, idx_k],
                tar_pts=predictions.ransac_tar_pts[:, idx_k],
            )
            keypoint_img = plot_keypoints_batch(batch, concate_input_in_pred=False)
            wrap_img = plot_Kabsch(batch, predictions.M[:, idx_k])
            pred_img = torch.cat([keypoint_img, wrap_img], dim=3)
            pred_imgs.append(pred_img)
        pred_imgs = torch.cat(pred_imgs, dim=0)
        return pred_imgs

    def eval_retrieval(
        self,
        batch,
        idx_batch,
        dataset_name,
        sort_pred_by_inliers=True,
        save_outputs=True,
        log_retrieval=True,
    ):
        torch.cuda.empty_cache()
        # prepare template data
        if dataset_name not in self.template_datas:
            self.set_template_data(dataset_name)

        template_data = self.template_datas[dataset_name]
        pose_recovery = self.pose_recovery[dataset_name]
        times = {"neighbor_search": None, "final_step": None}

        B, C, H, W = batch.tar_img.shape
        device = batch.tar_img.device

        # if low_memory_mode, two detections are forward at a time
        list_idx_sample = []
        if self.max_num_dets_per_forward is not None:
            for start_idx in np.arange(0, B, self.max_num_dets_per_forward):
                end_idx = min(start_idx + self.max_num_dets_per_forward, B)
                idx_sample_ = torch.arange(start_idx, end_idx, device=device)
                list_idx_sample.append(idx_sample_)
        else:
            idx_sample = torch.arange(0, B, device=device)
            list_idx_sample.append(idx_sample)

        for idx_sub_batch, idx_sample in enumerate(list_idx_sample):
            # compute target features
            tar_ae_features = self.ae_net(batch.tar_img[idx_sample])
            tar_label_np = np.asarray(
                batch.infos.label[idx_sample.cpu().numpy()]
            ).astype(np.int32)
            tar_label = torch.from_numpy(tar_label_np).to(device)

            # template data
            src_ae_features = template_data.ae_features[tar_label - 1]
            src_masks = template_data.mask[tar_label - 1]

            # Step 1: Nearest neighbor search
            self.timer.tic()
            predictions_ = self.testing_metric.test(
                src_feats=src_ae_features,
                tar_feat=tar_ae_features,
                src_masks=src_masks,
                tar_mask=batch.tar_mask[idx_sample],
                max_batch_size=None,
            )
            predictions_.infos = batch.infos
            if idx_sub_batch == 0:
                predictions = predictions_
            else:
                predictions.cat_df(predictions_)

        # Step 2: Find affine transforms
        num_patches = predictions.src_pts.shape[2]
        k = self.testing_metric.k
        pred_scales = torch.zeros(B, k, num_patches, device=device)
        pred_cosSin_inplanes = torch.zeros(B, k, num_patches, 2, device=device)

        self.timer.tic()
        for idx_k in range(k):
            idx_sample = torch.arange(0, B, device=device)
            idx_views = [idx_sample, predictions.id_src[:, idx_k]]

            tar_label_np = np.asarray(batch.infos.label).astype(np.int32)
            tar_label = torch.from_numpy(tar_label_np).to(device)

            src_ist_features = template_data.ist_features[tar_label - 1]
            tar_ist_features = self.ist_net.forward_by_chunk(batch.tar_img[idx_sample])

            if self.max_num_dets_per_forward is not None:
                (
                    pred_scales[:, idx_k],
                    pred_cosSin_inplanes[:, idx_k],
                ) = self.ist_net.inference_by_chunk(
                    src_feat=src_ist_features[idx_views],
                    tar_feat=tar_ist_features,
                    src_pts=predictions.src_pts[:, idx_k],
                    tar_pts=predictions.tar_pts[:, idx_k],
                    max_batch_size=self.max_num_dets_per_forward,
                )
            else:
                (
                    pred_scales[:, idx_k],
                    pred_cosSin_inplanes[:, idx_k],
                ) = self.ist_net.inference(
                    src_feat=src_ist_features[idx_views],
                    tar_feat=tar_ist_features,
                    src_pts=predictions.src_pts[:, idx_k],
                    tar_pts=predictions.tar_pts[:, idx_k],
                )

        predictions.register_tensor("relScale", pred_scales)
        predictions.register_tensor(
            "relInplane",
            pred_cosSin_inplanes,
        )
        times["neighbor_search"] = self.timer.toc()
        self.timer.reset()

        self.timer.tic()
        predictions = pose_recovery.forward_ransac(predictions=predictions)
        # sort the predictions by the number of inliers for each detection
        score = torch.sum(predictions.ransac_scores, dim=2) / num_patches
        predictions.register_tensor("scores", score)
        if sort_pred_by_inliers:
            order = torch.argsort(score, dim=1, descending=True)
            for k, v in predictions._tensors.items():
                if k in ["infos", "meta"]:
                    continue
                predictions.register_tensor(k, v[idx_sample[:, None], order])

        # calculate prediction
        pred_poses = self.pose_recovery[dataset_name].forward_recovery(
            tar_label=tar_label,
            tar_K=batch.tar_K,
            tar_M=batch.tar_M,
            pred_src_views=predictions.id_src,
            pred_M=predictions.M.clone(),
        )
        predictions.register_tensor("pred_poses", pred_poses)

        times["final_step"] = self.timer.toc()
        self.timer.reset()
        total_time = sum(times.values())

        if save_outputs:
            save_path = osp.join(self.log_dir, "predictions", f"{idx_batch}.npz")
            selected_idxs, predictions = self.filter_and_save(
                predictions,
                test_list=batch.test_list,
                time=total_time,
                save_path=save_path,
            )
        else:
            selected_idxs = list(range(B))

        if (
            log_retrieval
            and idx_batch % self.log_interval == 0
            and self.max_num_dets_per_forward is None
        ):
            vis_img = self.vis_retrieval(
                template_data=template_data,
                batch=batch,
                selected_idxs=selected_idxs,
                predictions=predictions,
                idx_batch=idx_batch,
            )
            sample_path = f"{self.log_dir}/retrieved_sample_rank{self.global_rank}_{idx_batch}.png"
            save_image(
                vis_img,
                sample_path,
                nrow=predictions.id_src.shape[0],
            )
            log_image(
                logger=self.logger,
                name=f"{dataset_name}",
                path=sample_path,
                step=int(self.global_step),
            )
        return predictions

    @torch.no_grad()
    def test_step(self, batch, idx_batch):
        self.eval_retrieval(
            batch,
            idx_batch=idx_batch,
            dataset_name=self.test_dataset_name,
        )
        return 0

    def on_test_epoch_end(self):
        if self.global_rank == 0:
            prediction_dir = osp.join(self.log_dir, "predictions")
            save_predictions_from_batched_predictions(
                prediction_dir,
                dataset_name=self.test_dataset_name,
                model_name=self.model_name,
                run_id=self.run_id,
                is_refined=False,
            )
