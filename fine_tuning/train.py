"""Fine-tune pretrained GigaPose on configurable Assetto Corsa splits."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

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
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--run-name", default="assettocorsa_ist_finetune")
    parser.add_argument("--seed", type=int, default=2023)
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
    cfg.save_dir = str(output_dir)
    cfg.name_exp = args.run_name
    cfg.machine.batch_size = args.batch_size
    cfg.machine.num_workers = args.num_workers
    cfg.machine.trainer.devices = [args.device]
    cfg.machine.trainer.max_steps = args.max_steps
    cfg.machine.trainer.max_epochs = -1
    cfg.machine.trainer.val_check_interval = args.validation_interval
    cfg.machine.trainer.check_val_every_n_epoch = None
    cfg.machine.trainer.num_sanity_val_steps = 2
    cfg.model.log_dir = str(output_dir)
    cfg.model.optim_config.nets_to_train = args.nets_to_train
    cfg.model.optim_config.ist_lr = args.ist_lr
    cfg.model.optim_config.ae_lr = args.ae_lr
    cfg.callback.checkpoint.dirpath = str(output_dir / "checkpoints")

    os.makedirs(output_dir, exist_ok=True)
    trainer = instantiate(cfg.machine.trainer)

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
    trainer.fit(
        model,
        train_dataloaders=[train_loader],
        val_dataloaders=validation_loader,
    )


if __name__ == "__main__":
    main()
