"""Standalone training for GigaPose translation/rotation residual heads."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import warnings

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from fine_tuning.heavy_validation import (
    HeavyValidationCallback,
    build_fixed_heavy_loader,
)
from fine_tuning.train import LossPrintCallback, parse_devices, parse_int_list
from fine_tuning.train_val import configure_logger
from src.utils.logging import get_logger
from src.utils.weight import load_checkpoint

from .callbacks import ResidualMetricHistoryCallback


logger = get_logger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="assettocorsa_new_dataset")
    parser.add_argument(
        "--root-dir", type=Path, default=Path("gigaPose_datasets/datasets")
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-split", default="train_pbr_web_gsam_clean")
    parser.add_argument("--validation-split", default="val_pbr_web_gsam_clean")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--ist-lr", type=float, default=2e-6)
    parser.add_argument("--residual-lr", type=float, default=1e-4)
    parser.add_argument("--ae-lr", type=float, default=1e-6)
    parser.add_argument("--nets-to-train", choices=("ist", "all"), default="ist")
    parser.add_argument(
        "--train-ist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Jointly update IST. Use --no-train-ist for residual-head-only training.",
    )
    parser.add_argument(
        "--ae-train-mode",
        choices=("all", "last-block", "last-blocks", "block-offsets", "norm"),
        default="last-blocks",
    )
    parser.add_argument("--ae-train-last-n-blocks", type=int, default=1)
    parser.add_argument("--ae-train-block-offsets", default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--run-name", default="assettocorsa_residual_pose")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument(
        "--logger", choices=("tensorboard", "wandb", "none"), default="wandb"
    )
    parser.add_argument("--wandb-project", default="gigapose")
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--log-every-n-steps", type=int, default=1)
    parser.add_argument("--print-loss-every", type=int, default=50)
    parser.add_argument("--metric-csv-every", type=int, default=50)
    parser.add_argument("--match-sim-threshold", type=float, default=None)
    parser.add_argument("--match-patch-threshold", type=float, default=None)

    base = parser.add_argument_group("base pose-aware IST objective")
    base.add_argument("--base-log-depth-weight", type=float, default=1.0)
    base.add_argument("--instance-log-scale-weight", type=float, default=0.5)
    base.add_argument("--scale-consistency-weight", type=float, default=0.05)
    base.add_argument("--inplane-weight", type=float, default=0.75)
    base.add_argument("--base-reprojection-weight", type=float, default=0.0)
    base.add_argument("--anti-flip-weight", type=float, default=0.0)
    base.add_argument("--anti-flip-margin", type=float, default=0.25)
    base.add_argument("--base-log-depth-beta", type=float, default=0.05)
    base.add_argument("--base-reprojection-beta", type=float, default=0.05)
    base.add_argument("--info-nce-weight", type=float, default=1.0)
    base.add_argument("--soft-template-weight", type=float, default=0.25)

    residual = parser.add_argument_group("explicit residual pose objective")
    residual.add_argument("--residual-hidden-dim", type=int, default=256)
    residual.add_argument(
        "--rotation-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the optional bounded three-axis SO(3) correction head.",
    )
    residual.add_argument("--residual-center-weight", type=float, default=1.0)
    residual.add_argument("--residual-log-depth-weight", type=float, default=1.0)
    residual.add_argument("--residual-translation-weight", type=float, default=0.05)
    residual.add_argument("--residual-rotation-weight", type=float, default=0.25)
    residual.add_argument("--residual-regularization-weight", type=float, default=1e-3)
    residual.add_argument("--residual-center-beta", type=float, default=0.05)
    residual.add_argument("--residual-log-depth-beta", type=float, default=0.05)
    residual.add_argument("--residual-translation-beta", type=float, default=0.05)
    residual.add_argument("--residual-rotation-beta", type=float, default=0.05)
    residual.add_argument("--max-center-offset-px", type=float, default=56.0)
    residual.add_argument("--max-log-depth-residual", type=float, default=0.5)
    residual.add_argument("--max-rotation-deg", type=float, default=20.0)
    residual.add_argument(
        "--pose-translation-unit", choices=("mm", "m"), default="mm"
    )
    residual.add_argument("--best-residual-checkpoints", type=int, default=3)

    heavy = parser.add_argument_group("optional end-to-end heavy validation")
    heavy.add_argument(
        "--heavy-validation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    heavy.add_argument("--heavy-validation-interval", type=int, default=1000)
    heavy.add_argument("--heavy-validation-images", type=int, default=4)
    heavy.add_argument("--heavy-validation-seed", type=int, default=20260707)
    heavy.add_argument("--heavy-validation-mesh", type=Path, default=None)
    return parser.parse_args()


def make_dataset_config(
    cfg,
    args: argparse.Namespace,
    split_name: str,
    *,
    augment: bool,
    retain_heavy_visuals: bool = False,
):
    dataset_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg.data.train.dataloader, resolve=True)
    )
    dataset_cfg._target_ = (
        "fine_tuning.pose_aware_training.dataset."
        "PoseAwareAssettoCorsaFineTuneSet"
    )
    dataset_cfg.root_dir = str(args.root_dir.resolve())
    dataset_cfg.dataset_name = args.dataset_name
    dataset_cfg.split_name = split_name
    dataset_cfg.batch_size = args.batch_size
    dataset_cfg.depth_scale = 1.0
    dataset_cfg.template_config.dir = str((args.root_dir / "templates").resolve())
    dataset_cfg.template_config.scale_factor = 1.0
    dataset_cfg.transforms.rgb_augmentation = augment
    dataset_cfg.retain_heavy_visuals = retain_heavy_visuals
    return dataset_cfg


def validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    positive = {
        "validation interval": args.validation_interval,
        "checkpoint interval": args.checkpoint_interval,
        "residual hidden dim": args.residual_hidden_dim,
        "max center offset": args.max_center_offset_px,
        "max log-depth residual": args.max_log_depth_residual,
        "max rotation": args.max_rotation_deg,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"Values must be positive: {invalid}")
    weights = {
        "base log-depth": args.base_log_depth_weight,
        "instance log-scale": args.instance_log_scale_weight,
        "scale consistency": args.scale_consistency_weight,
        "inplane": args.inplane_weight,
        "base reprojection": args.base_reprojection_weight,
        "anti-flip": args.anti_flip_weight,
        "residual center": args.residual_center_weight,
        "residual log-depth": args.residual_log_depth_weight,
        "residual translation": args.residual_translation_weight,
        "residual rotation": args.residual_rotation_weight,
        "residual regularization": args.residual_regularization_weight,
    }
    negative = [name for name, value in weights.items() if value < 0]
    if negative:
        raise ValueError(f"Loss weights must be nonnegative: {negative}")
    if not any(
        value > 0
        for value in (
            args.residual_center_weight,
            args.residual_log_depth_weight,
            args.residual_translation_weight,
        )
    ):
        raise ValueError("At least one translation-residual loss must be positive")
    if args.best_residual_checkpoints < 0:
        raise ValueError("--best-residual-checkpoints must be nonnegative")
    if (
        args.heavy_validation
        and args.heavy_validation_interval % args.validation_interval != 0
    ):
        raise ValueError(
            "--heavy-validation-interval must be divisible by "
            "--validation-interval"
        )


def main() -> None:
    warnings.filterwarnings(
        "ignore", message="TypedStorage is deprecated.*", category=UserWarning
    )
    args = parse_args()
    validate_args(args)
    pl.seed_everything(args.seed)

    with initialize_config_dir(
        version_base=None,
        config_dir=str((REPO_ROOT / "configs").resolve()),
    ):
        cfg = compose(config_name="train")
    OmegaConf.set_struct(cfg, False)
    output_dir = (
        REPO_ROOT / "gigaPose_datasets" / "results" / args.run_name
    ).resolve()
    os.makedirs(output_dir, exist_ok=True)
    cfg.save_dir = str(output_dir)
    cfg.name_exp = args.run_name
    cfg.machine.batch_size = args.batch_size
    cfg.machine.num_workers = args.num_workers
    cfg.machine.trainer.devices = parse_devices(args.devices, args.device)
    cfg.machine.trainer.max_steps = args.max_steps
    cfg.machine.trainer.max_epochs = -1
    cfg.machine.trainer.val_check_interval = args.validation_interval
    cfg.machine.trainer.check_val_every_n_epoch = None
    cfg.machine.trainer.num_sanity_val_steps = 2
    cfg.machine.trainer.log_every_n_steps = args.log_every_n_steps
    cfg.callback.checkpoint.dirpath = str(output_dir / "checkpoints")
    cfg.callback.checkpoint.every_n_train_steps = args.checkpoint_interval
    configure_logger(cfg, args, output_dir)

    cfg.model._target_ = (
        "fine_tuning.residual_pose_training.model.ResidualPoseGigaPose"
    )
    cfg.model.log_dir = str(output_dir)
    cfg.model.optim_config.nets_to_train = args.nets_to_train
    cfg.model.optim_config.ist_lr = args.ist_lr
    cfg.model.optim_config.ae_lr = args.ae_lr
    cfg.model.ae_net.train_mode = args.ae_train_mode
    cfg.model.ae_net.train_last_n_blocks = args.ae_train_last_n_blocks
    cfg.model.ae_net.train_block_offsets = parse_int_list(
        args.ae_train_block_offsets
    )
    unit = 1000.0 if args.pose_translation_unit == "mm" else 1.0
    cfg.model.pose_loss_config = {
        "log_depth_weight": args.base_log_depth_weight,
        "instance_log_scale_weight": args.instance_log_scale_weight,
        "scale_consistency_weight": args.scale_consistency_weight,
        "inplane_weight": args.inplane_weight,
        "reprojection_weight": args.base_reprojection_weight,
        "anti_flip_weight": args.anti_flip_weight,
        "anti_flip_margin": args.anti_flip_margin,
        "optimize_pose_monitor_errors": False,
        "log_depth_beta": args.base_log_depth_beta,
        "reprojection_beta": args.base_reprojection_beta,
        "direct_translation_scale": unit,
        "direct_depth_scale": unit,
        "info_nce_weight": args.info_nce_weight,
        "soft_template_weight": args.soft_template_weight,
        "translation_to_mm": 1.0 if args.pose_translation_unit == "mm" else 1000.0,
    }
    cfg.model.residual_config = {
        "hidden_dim": args.residual_hidden_dim,
        "residual_lr": args.residual_lr,
        "train_ist": args.train_ist,
        "enable_rotation": args.rotation_residual,
        "apply_at_inference": True,
        "max_center_offset_px": args.max_center_offset_px,
        "max_log_depth_residual": args.max_log_depth_residual,
        "max_rotation_deg": args.max_rotation_deg,
        "center_weight": args.residual_center_weight,
        "log_depth_weight": args.residual_log_depth_weight,
        "translation_weight": args.residual_translation_weight,
        "rotation_weight": args.residual_rotation_weight,
        "regularization_weight": args.residual_regularization_weight,
        "center_beta": args.residual_center_beta,
        "log_depth_beta": args.residual_log_depth_beta,
        "translation_beta": args.residual_translation_beta,
        "rotation_beta": args.residual_rotation_beta,
        "translation_unit": unit,
        "translation_to_mm": 1.0 if args.pose_translation_unit == "mm" else 1000.0,
    }
    if args.match_sim_threshold is not None:
        cfg.model.testing_metric.sim_threshold = args.match_sim_threshold
    if args.match_patch_threshold is not None:
        cfg.model.testing_metric.patch_threshold = args.match_patch_threshold

    train_dataset = instantiate(
        make_dataset_config(cfg, args, args.train_split, augment=True)
    )
    validation_dataset = instantiate(
        make_dataset_config(cfg, args, args.validation_split, augment=False)
    )
    train_loader = DataLoader(
        train_dataset.web_dataloader.datapipeline,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn,
    )
    validation_loader = DataLoader(
        validation_dataset.web_dataloader.datapipeline,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=validation_dataset.collate_fn,
    )

    heavy_callback = None
    heavy_template_dataset = None
    if args.heavy_validation:
        heavy_dataset = instantiate(
            make_dataset_config(
                cfg,
                args,
                args.validation_split,
                augment=False,
                retain_heavy_visuals=True,
            )
        )
        heavy_dir = output_dir / "heavy_validation"
        heavy_loader, selected_keys = build_fixed_heavy_loader(
            dataset=heavy_dataset,
            num_frames=args.heavy_validation_images,
            seed=args.heavy_validation_seed,
            selection_path=heavy_dir / "fixed_selection.json",
        )
        template_cfg = make_dataset_config(
            cfg, args, args.validation_split, augment=False
        )
        template_cfg._target_ = "src.dataloader.template.TemplateSet"
        heavy_template_dataset = instantiate(template_cfg)
        mesh_path = (
            args.heavy_validation_mesh
            or args.root_dir / args.dataset_name / "models" / "obj_000001.ply"
        ).resolve()
        heavy_callback = HeavyValidationCallback(
            loader=heavy_loader,
            dataset_name=args.dataset_name,
            interval=args.heavy_validation_interval,
            output_dir=heavy_dir,
            mesh_path=mesh_path,
        )
        logger.info("Fixed heavy-validation keys: %s", selected_keys)

    model = instantiate(cfg.model)
    load_checkpoint(model, args.checkpoint, checkpoint_key="state_dict")
    model.heavy_validation_enabled = False
    model.template_datasets = (
        {args.dataset_name: heavy_template_dataset}
        if heavy_template_dataset is not None
        else {}
    )

    trainer = instantiate(cfg.machine.trainer)
    if args.print_loss_every > 0:
        trainer.callbacks.append(LossPrintCallback(args.print_loss_every))
    trainer.callbacks.append(
        ResidualMetricHistoryCallback(
            output_dir / "pose_metrics.csv",
            train_interval=args.metric_csv_every,
        )
    )
    if args.best_residual_checkpoints > 0:
        trainer.callbacks.append(
            ModelCheckpoint(
                dirpath=str(output_dir / "checkpoints"),
                filename="best-residual-step{step:06d}",
                monitor="val/monitor_refined_translation_error_mm",
                mode="min",
                save_top_k=args.best_residual_checkpoints,
                save_last=False,
                auto_insert_metric_name=False,
                verbose=True,
            )
        )
    if heavy_callback is not None:
        trainer.callbacks.append(heavy_callback)

    logger.info("Residual-pose outputs: %s", output_dir)
    logger.info(
        "Translation residual: center=%.3g log-depth=%.3g metric=%.3g; "
        "rotation enabled=%s weight=%.3g",
        args.residual_center_weight,
        args.residual_log_depth_weight,
        args.residual_translation_weight,
        args.rotation_residual,
        args.residual_rotation_weight,
    )
    trainer.fit(
        model,
        train_dataloaders=[train_loader],
        val_dataloaders=validation_loader,
    )


if __name__ == "__main__":
    main()
