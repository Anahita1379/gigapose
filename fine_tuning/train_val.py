"""Fine-tune GigaPose with isolated light and heavy validation loaders.

This alternate entry point leaves ``fine_tuning.train`` unchanged. Ordinary
training and light validation remain crop-only. Optional heavy validation uses
a separate fixed map-style loader and a DDP-safe rank-zero callback.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from fine_tuning.early_stopping import (
    add_early_stopping_args,
    configure_early_stopping,
    validate_early_stopping_args,
)
from fine_tuning.heavy_validation import (
    HeavyValidationCallback,
    build_fixed_heavy_loader,
)
from fine_tuning.train import LossPrintCallback, parse_devices, parse_int_list
from src.utils.logging import get_logger
from src.utils.weight import load_checkpoint


logger = get_logger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--ist-lr", type=float, default=1e-5)
    parser.add_argument("--ae-lr", type=float, default=1e-6)
    parser.add_argument("--nets-to-train", choices=("ist", "ae", "all"), default="ist")
    parser.add_argument(
        "--ae-train-mode",
        choices=("all", "last-block", "last-blocks", "block-offsets", "norm"),
        default="all",
    )
    parser.add_argument("--ae-train-last-n-blocks", type=int, default=1)
    parser.add_argument("--ae-train-block-offsets", default=None)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument(
        "--devices",
        default=None,
        help="GPU ids such as '0', '0,1,2,3', or 'all'.",
    )
    parser.add_argument("--run-name", default="assettocorsa_ist_finetune")
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
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--match-sim-threshold", type=float, default=None)
    parser.add_argument("--match-patch-threshold", type=float, default=None)
    parser.add_argument(
        "--heavy-validation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--heavy-validation-interval", type=int, default=1000)
    parser.add_argument("--heavy-validation-images", type=int, default=4)
    parser.add_argument("--heavy-validation-seed", type=int, default=20260707)
    parser.add_argument("--heavy-validation-mesh", type=Path, default=None)
    add_early_stopping_args(
        parser,
        default_monitor=None,
        default_mode="min",
        default_patience=16,
        default_min_delta=1e-4,
        default_start_step=2000,
    )
    args = parser.parse_args()
    if args.early_stopping_monitor is None:
        args.early_stopping_monitor = (
            "val/matching" if args.nets_to_train == "ae" else "val/loss"
        )
    return args


def make_dataset_config(
    cfg,
    args: argparse.Namespace,
    split_name: str,
    augment: bool,
    retain_heavy_visuals: bool = False,
):
    dataset_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg.data.train.dataloader, resolve=True)
    )
    dataset_cfg._target_ = "fine_tuning.dataloader.AssettoCorsaFineTuneSet"
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


def configure_logger(cfg, args: argparse.Namespace, output_dir: Path) -> None:
    if args.logger == "tensorboard":
        cfg.machine.trainer.logger = OmegaConf.create(
            {
                "_target_": "pytorch_lightning.loggers.TensorBoardLogger",
                "save_dir": str(output_dir),
                "name": "tensorboard",
                "version": "",
            }
        )
    elif args.logger == "wandb":
        cfg.machine.trainer.logger = OmegaConf.create(
            {
                "_target_": "pytorch_lightning.loggers.WandbLogger",
                "project": args.wandb_project,
                "save_dir": str(output_dir),
                "offline": args.wandb_offline,
                "name": args.run_name,
            }
        )
    else:
        cfg.machine.trainer.logger = False


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="TypedStorage is deprecated.*",
        category=UserWarning,
    )
    args = parse_args()
    validate_early_stopping_args(args)
    pl.seed_everything(args.seed)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {args.checkpoint}")
    if (
        args.heavy_validation
        and args.heavy_validation_interval % args.validation_interval != 0
    ):
        raise ValueError(
            "--heavy-validation-interval must be divisible by "
            "--validation-interval because the heavy callback runs after a "
            "light-validation pass."
        )

    with initialize_config_dir(
        version_base=None, config_dir=str((REPO_ROOT / "configs").resolve())
    ):
        cfg = compose(config_name="train")
    OmegaConf.set_struct(cfg, False)

    output_dir = (
        REPO_ROOT / "gigaPose_datasets" / "results" / args.run_name
    ).resolve()
    devices = parse_devices(args.devices, args.device)
    cfg.save_dir = str(output_dir)
    cfg.name_exp = args.run_name
    cfg.machine.batch_size = args.batch_size
    cfg.machine.num_workers = args.num_workers
    cfg.machine.trainer.devices = devices
    cfg.machine.trainer.max_steps = args.max_steps
    cfg.machine.trainer.max_epochs = -1
    cfg.machine.trainer.val_check_interval = args.validation_interval
    cfg.machine.trainer.check_val_every_n_epoch = None
    cfg.machine.trainer.num_sanity_val_steps = 2
    cfg.machine.trainer.log_every_n_steps = args.log_every_n_steps
    cfg.model.log_dir = str(output_dir)
    cfg.model.optim_config.nets_to_train = args.nets_to_train
    cfg.model.optim_config.ist_lr = args.ist_lr
    cfg.model.optim_config.ae_lr = args.ae_lr
    cfg.model.ae_net.train_mode = args.ae_train_mode
    cfg.model.ae_net.train_last_n_blocks = args.ae_train_last_n_blocks
    cfg.model.ae_net.train_block_offsets = parse_int_list(
        args.ae_train_block_offsets
    )
    if args.match_sim_threshold is not None:
        cfg.model.testing_metric.sim_threshold = args.match_sim_threshold
    if args.match_patch_threshold is not None:
        cfg.model.testing_metric.patch_threshold = args.match_patch_threshold
    cfg.callback.checkpoint.dirpath = str(output_dir / "checkpoints")
    cfg.callback.checkpoint.every_n_train_steps = args.checkpoint_interval
    if args.early_stopping:
        cfg.callback.checkpoint.save_last = False
    configure_logger(cfg, args, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # These two loaders are always crop-only, even when heavy validation is on.
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

    heavy_loader = None
    heavy_template_dataset = None
    heavy_callback = None
    mesh_path = (
        args.heavy_validation_mesh
        or args.root_dir / args.dataset_name / "models" / "obj_000001.ply"
    ).resolve()
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
    # Disable the older in-validation-step heavy path. This entry point uses
    # only the independent callback and loader above.
    model.heavy_validation_enabled = False
    model.template_datasets = (
        {args.dataset_name: heavy_template_dataset}
        if heavy_template_dataset is not None
        else {}
    )

    trainer = instantiate(cfg.machine.trainer)
    if args.print_loss_every > 0:
        trainer.callbacks.append(LossPrintCallback(args.print_loss_every))
    configure_early_stopping(
        trainer,
        args,
        output_dir,
        logger=logger,
    )
    if heavy_callback is not None:
        trainer.callbacks.append(heavy_callback)

    logger.info(
        "Fine-tuning %s on %s; outputs: %s",
        args.nets_to_train,
        args.dataset_name,
        output_dir,
    )
    logger.info("Checkpoints: %s", output_dir / "checkpoints")
    logger.info("Light validation images: %s", output_dir / "validation_images")
    logger.info(
        "Camera preprocessing: symmetric pad to 2064x760; front/rear outer "
        "258 px per side invalid (75%% horizontal region retained)"
    )
    if heavy_callback is not None:
        logger.info(
            "Dedicated heavy validation: every %d steps; outputs: %s",
            args.heavy_validation_interval,
            output_dir / "heavy_validation",
        )
        logger.info("Heavy validation mesh: %s", mesh_path)
    else:
        logger.info("Dedicated heavy validation: disabled")
    if args.logger == "tensorboard":
        logger.info("TensorBoard: tensorboard --logdir %s", output_dir / "tensorboard")
    elif args.logger == "wandb":
        logger.info(
            "Weights & Biases run: project=%s name=%s",
            args.wandb_project,
            args.run_name,
        )

    trainer.fit(
        model,
        train_dataloaders=[train_loader],
        val_dataloaders=validation_loader,
    )


if __name__ == "__main__":
    main()
