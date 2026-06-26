"""Fine-tune pretrained GigaPose on configurable Assetto Corsa splits."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.utils.logging import get_logger
from src.utils.weight import load_checkpoint


logger = get_logger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_devices(value: str | None, legacy_device: int | None) -> list[int]:
    if value is None:
        return [0 if legacy_device is None else legacy_device]
    value = value.strip().lower()
    if value in {"all", "auto"}:
        count = torch.cuda.device_count()
        if count < 1:
            raise RuntimeError("No CUDA devices are available.")
        return list(range(count))
    devices = [part.strip() for part in value.split(",") if part.strip()]
    if not devices:
        raise ValueError("--devices must be 'all' or a comma-separated GPU list.")
    return [int(device) for device in devices]


class LossPrintCallback(pl.Callback):
    """Print compact train/validation metrics while Lightning also logs them."""

    def __init__(self, every_n_steps: int) -> None:
        self.every_n_steps = every_n_steps

    @staticmethod
    def _as_float(value):
        if hasattr(value, "detach"):
            return float(value.detach().cpu())
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _format_metrics(self, trainer: pl.Trainer, prefixes: tuple[str, ...]) -> str:
        parts = []
        for name, value in sorted(trainer.callback_metrics.items()):
            if name != "total" and any(name.startswith(prefix) for prefix in prefixes):
                scalar = self._as_float(value)
                if scalar is not None:
                    parts.append(f"{name}={scalar:.5f}")
        return " ".join(parts)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.every_n_steps <= 0 or not trainer.is_global_zero:
            return
        if trainer.global_step == 0 or trainer.global_step % self.every_n_steps != 0:
            return
        parts = []
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        scalar_loss = self._as_float(loss)
        if scalar_loss is not None:
            parts.append(f"loss={scalar_loss:.5f}")
        metrics = self._format_metrics(trainer, ("train/",))
        if metrics:
            parts.append(metrics)
        metrics = " ".join(parts)
        if metrics:
            logger.info("step=%d %s", trainer.global_step, metrics)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        metrics = self._format_metrics(trainer, ("val/",))
        if metrics:
            logger.info("validation step=%d %s", trainer.global_step, metrics)


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
        "--device",
        type=int,
        default=None,
        help="Legacy single-GPU option. Prefer --devices.",
    )
    parser.add_argument(
        "--devices",
        default=None,
        help="GPU ids to use, e.g. '0', '0,1,2,3', or 'all'.",
    )
    parser.add_argument("--run-name", default="assettocorsa_ist_finetune")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument(
        "--logger",
        choices=("tensorboard", "wandb", "none"),
        default="tensorboard",
        help="Experiment logger to use for losses and validation images.",
    )
    parser.add_argument("--wandb-project", default="gigapose")
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--log-every-n-steps", type=int, default=1)
    parser.add_argument(
        "--print-loss-every",
        type=int,
        default=50,
        help="Print compact loss metrics every N optimizer steps; use 0 to disable.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Save a checkpoint every N optimizer steps.",
    )
    return parser.parse_args()


def make_dataset_config(cfg, args: argparse.Namespace, split_name: str, augment: bool):
    dataset_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg.data.train.dataloader, resolve=True)
    )
    dataset_cfg._target_ = "fine_tuning.dataloader.AssettoCorsaFineTuneSet"
    dataset_cfg.root_dir = str(args.root_dir.resolve())
    dataset_cfg.dataset_name = args.dataset_name
    dataset_cfg.split_name = split_name
    dataset_cfg.batch_size = args.batch_size
    # Prepared depths and poses are both in millimeters.
    dataset_cfg.depth_scale = 1.0
    dataset_cfg.template_config.dir = str((args.root_dir / "templates").resolve())
    dataset_cfg.template_config.scale_factor = 1.0
    dataset_cfg.transforms.rgb_augmentation = augment
    return dataset_cfg


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {args.checkpoint}")

    with initialize_config_dir(
        version_base=None, config_dir=str((REPO_ROOT / "configs").resolve())
    ):
        cfg = compose(config_name="train")
    OmegaConf.set_struct(cfg, False)

    output_dir = (REPO_ROOT / "gigaPose_datasets" / "results" / args.run_name).resolve()
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
    cfg.callback.checkpoint.dirpath = str(output_dir / "checkpoints")
    cfg.callback.checkpoint.every_n_train_steps = args.checkpoint_interval

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

    os.makedirs(output_dir, exist_ok=True)
    trainer = instantiate(cfg.machine.trainer)
    if args.print_loss_every > 0:
        trainer.callbacks.append(LossPrintCallback(args.print_loss_every))

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
    # Initialize weights only. Passing ckpt_path to trainer.fit would also
    # restore the old optimizer and global step, which is not desired here.
    load_checkpoint(model, args.checkpoint, checkpoint_key="state_dict")
    logger.info(
        "Fine-tuning %s on %s; outputs: %s",
        args.nets_to_train,
        args.dataset_name,
        output_dir,
    )
    logger.info("Checkpoints: %s", output_dir / "checkpoints")
    logger.info("Validation images: %s", output_dir / "validation_images")
    if args.logger == "tensorboard":
        logger.info("TensorBoard: tensorboard --logdir %s", output_dir / "tensorboard")
    elif args.logger == "wandb":
        logger.info("Weights & Biases run: project=%s name=%s", args.wandb_project, args.run_name)
    trainer.fit(
        model,
        train_dataloaders=[train_loader],
        val_dataloaders=validation_loader,
    )


if __name__ == "__main__":
    main()
