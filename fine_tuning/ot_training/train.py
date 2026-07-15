"""Standalone OT/soft-correspondence AE fine-tuning entry point.

This module is intentionally separate from ``fine_tuning.train`` and
``fine_tuning.pose_aware_training.train``. It trains AE/DINO features using a
masked Sinkhorn optimal-transport correspondence objective.
"""

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

from fine_tuning.early_stopping import (
    add_early_stopping_args,
    configure_early_stopping,
    validate_early_stopping_args,
)
from fine_tuning.train import LossPrintCallback, parse_devices, parse_int_list
from fine_tuning.train_val import configure_logger
from src.utils.logging import get_logger
from src.utils.weight import load_checkpoint


logger = get_logger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="assettocorsa")
    parser.add_argument(
        "--root-dir", type=Path, default=Path("gigaPose_datasets/datasets")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("gigaPose_datasets/pretrained/gigaPose_v1.ckpt"),
    )
    parser.add_argument("--train-split", default="train_pbr_web")
    parser.add_argument("--validation-split", default="val_pbr_web")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--ae-lr", type=float, default=1e-6)
    parser.add_argument(
        "--ae-train-mode",
        choices=("all", "last-block", "last-blocks", "block-offsets"),
        default="last-blocks",
        help=(
            "Which AE/DINOv2 parameters to update. Start with last-blocks or "
            "block-offsets; full all is slower and easier to overfit."
        ),
    )
    parser.add_argument("--ae-train-last-n-blocks", type=int, default=1)
    parser.add_argument(
        "--ae-train-block-offsets",
        default=None,
        help=(
            "Comma-separated offsets from the final DINO block for "
            "--ae-train-mode block-offsets. Example: '2' trains the "
            "penultimate block; '2,3' trains the two before last."
        ),
    )
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--run-name", default="assettocorsa_ot_ae")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument(
        "--logger",
        choices=("tensorboard", "wandb", "none"),
        default="tensorboard",
    )
    parser.add_argument("--wandb-project", default="gigapose")
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--log-every-n-steps", type=int, default=1)
    parser.add_argument("--print-loss-every", type=int, default=50)

    ot = parser.add_argument_group("optimal transport losses")
    ot.add_argument("--feature-temperature", type=float, default=0.07)
    ot.add_argument("--sinkhorn-iterations", type=int, default=30)
    ot.add_argument("--correspondence-weight", type=float, default=1.0)
    ot.add_argument("--soft-patch-reprojection-weight", type=float, default=0.25)
    ot.add_argument("--soft-affine-center-weight", type=float, default=0.25)
    ot.add_argument(
        "--entropy-weight",
        type=float,
        default=0.0,
        help=(
            "Optional entropy penalty on the transport matrix. Positive values "
            "make OT sharper; keep 0 initially."
        ),
    )
    ot.add_argument("--reprojection-beta", type=float, default=0.05)
    ot.add_argument(
        "--hard-match-confidence",
        type=float,
        default=0.05,
        help=(
            "Minimum row-conditional OT probability for validation hard "
            "matches. Only confident mutual nearest matches are visualized."
        ),
    )
    ot.add_argument(
        "--best-ot-checkpoints",
        type=int,
        default=3,
        help=(
            "Save this many additional checkpoints ranked by validation GT "
            "top-1 OT correspondence accuracy. Set 0 to disable."
        ),
    )
    add_early_stopping_args(
        parser,
        default_monitor="val/monitor_ot_gt_top1_accuracy",
        default_mode="max",
        default_patience=12,
        default_min_delta=0.001,
        default_start_step=2000,
    )
    return parser.parse_args()


def make_dataset_config(
    cfg,
    args: argparse.Namespace,
    split_name: str,
    *,
    augment: bool,
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
    dataset_cfg.retain_heavy_visuals = False
    return dataset_cfg


def validate_args(args: argparse.Namespace) -> None:
    validate_early_stopping_args(args)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {args.checkpoint}")
    if args.validation_interval <= 0:
        raise ValueError("--validation-interval must be positive")
    if args.checkpoint_interval <= 0:
        raise ValueError("--checkpoint-interval must be positive")
    if args.ae_lr <= 0:
        raise ValueError("--ae-lr must be positive")
    if args.feature_temperature <= 0:
        raise ValueError("--feature-temperature must be positive")
    if args.sinkhorn_iterations <= 0:
        raise ValueError("--sinkhorn-iterations must be positive")
    if args.ae_train_last_n_blocks <= 0:
        raise ValueError("--ae-train-last-n-blocks must be positive")
    offsets = parse_int_list(args.ae_train_block_offsets)
    if offsets is not None and any(offset <= 0 for offset in offsets):
        raise ValueError("--ae-train-block-offsets values must be positive")
    if not 0.0 <= args.hard_match_confidence <= 1.0:
        raise ValueError("--hard-match-confidence must be in [0, 1]")
    if args.best_ot_checkpoints < 0:
        raise ValueError("--best-ot-checkpoints must be nonnegative")
    nonnegative = {
        "correspondence": args.correspondence_weight,
        "soft patch reprojection": args.soft_patch_reprojection_weight,
        "soft affine center": args.soft_affine_center_weight,
        "entropy": args.entropy_weight,
    }
    invalid = [name for name, value in nonnegative.items() if value < 0]
    if invalid:
        raise ValueError(f"Loss weights must be nonnegative: {invalid}")
    if not any(value > 0 for value in nonnegative.values()):
        raise ValueError("At least one OT loss weight must be positive")


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="TypedStorage is deprecated.*",
        category=UserWarning,
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
    if args.early_stopping:
        cfg.callback.checkpoint.save_last = False
    configure_logger(cfg, args, output_dir)

    cfg.model._target_ = "fine_tuning.ot_training.model.OTGigaPose"
    cfg.model.log_dir = str(output_dir)
    cfg.model.optim_config.nets_to_train = "ae"
    cfg.model.optim_config.ae_lr = args.ae_lr
    cfg.model.ae_net.train_mode = args.ae_train_mode
    cfg.model.ae_net.train_last_n_blocks = args.ae_train_last_n_blocks
    cfg.model.ae_net.train_block_offsets = parse_int_list(args.ae_train_block_offsets)
    cfg.model.ot_loss_config = {
        "feature_temperature": args.feature_temperature,
        "sinkhorn_iterations": args.sinkhorn_iterations,
        "correspondence_weight": args.correspondence_weight,
        "soft_patch_reprojection_weight": args.soft_patch_reprojection_weight,
        "soft_affine_center_weight": args.soft_affine_center_weight,
        "entropy_weight": args.entropy_weight,
        "reprojection_beta": args.reprojection_beta,
        "hard_match_confidence": args.hard_match_confidence,
    }

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

    model = instantiate(cfg.model)
    load_checkpoint(model, args.checkpoint, checkpoint_key="state_dict")
    model.heavy_validation_enabled = False
    model.template_datasets = {}

    trainer = instantiate(cfg.machine.trainer)
    if args.print_loss_every > 0:
        trainer.callbacks.append(LossPrintCallback(args.print_loss_every))
    if args.best_ot_checkpoints > 0:
        trainer.callbacks.append(
            ModelCheckpoint(
                dirpath=str(output_dir / "checkpoints"),
                filename="best-ot-step{step:06d}",
                monitor="val/monitor_ot_gt_top1_accuracy",
                mode="max",
                save_top_k=args.best_ot_checkpoints,
                save_last=False,
                auto_insert_metric_name=False,
                verbose=True,
            )
        )
    configure_early_stopping(
        trainer,
        args,
        output_dir,
        logger=logger,
    )

    logger.info("OT AE training outputs: %s", output_dir)
    logger.info(
        "AE train mode: %s last_n=%s block_offsets=%s",
        args.ae_train_mode,
        args.ae_train_last_n_blocks,
        args.ae_train_block_offsets,
    )
    logger.info(
        "OT losses: correspondence=%.3g soft-patch=%.3g soft-affine=%.3g "
        "entropy=%.3g",
        args.correspondence_weight,
        args.soft_patch_reprojection_weight,
        args.soft_affine_center_weight,
        args.entropy_weight,
    )
    logger.info(
        "Validation hard matches: mutual-only confidence>=%.3g",
        args.hard_match_confidence,
    )

    trainer.fit(
        model,
        train_dataloaders=[train_loader],
        val_dataloaders=validation_loader,
    )


if __name__ == "__main__":
    main()
