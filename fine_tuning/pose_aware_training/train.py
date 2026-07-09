"""Standalone pose-aware GigaPose fine-tuning entry point.

This module does not modify or replace ``fine_tuning.train`` or
``fine_tuning.train_val``. Run it with:

    python -m fine_tuning.pose_aware_training.train ...
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
from torch.utils.data import DataLoader

from fine_tuning.heavy_validation import (
    HeavyValidationCallback,
    build_fixed_heavy_loader,
)
from fine_tuning.train import LossPrintCallback, parse_devices, parse_int_list
from fine_tuning.train_val import configure_logger
from src.utils.logging import get_logger
from src.utils.weight import load_checkpoint

from .callbacks import PoseMetricHistoryCallback


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
    parser.add_argument("--ist-lr", type=float, default=1e-5)
    parser.add_argument("--ae-lr", type=float, default=1e-6)
    parser.add_argument(
        "--nets-to-train",
        choices=("ist", "all"),
        default="ist",
        help=(
            "'ist' optimizes pose-aware IST losses. 'all' additionally trains "
            "AE retrieval with InfoNCE and soft pose-aware template selection."
        ),
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
    parser.add_argument("--run-name", default="assettocorsa_pose_aware")
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
    parser.add_argument("--metric-csv-every", type=int, default=50)
    parser.add_argument("--match-sim-threshold", type=float, default=None)
    parser.add_argument("--match-patch-threshold", type=float, default=None)

    losses = parser.add_argument_group("pose-aware losses")
    losses.add_argument("--log-depth-weight", type=float, default=1.0)
    losses.add_argument("--inplane-weight", type=float, default=1.0)
    losses.add_argument("--reprojection-weight", type=float, default=0.1)
    losses.add_argument(
        "--optimize-pose-monitor-errors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When enabled, add normalized direct translation/depth/full-rotation "
            "pose losses to the optimized objective. By default these quantities "
            "are logged only."
        ),
    )
    losses.add_argument("--direct-translation-weight", type=float, default=0.05)
    losses.add_argument("--direct-depth-weight", type=float, default=0.05)
    losses.add_argument("--direct-rotation-weight", type=float, default=0.05)
    losses.add_argument(
        "--direct-reprojection-weight",
        type=float,
        default=0.0,
        help=(
            "Extra weight for the same normalized center-reprojection loss used "
            "by --reprojection-weight. Keep at 0 unless you intentionally want "
            "to emphasize this term when direct pose losses are enabled."
        ),
    )
    losses.add_argument("--info-nce-weight", type=float, default=1.0)
    losses.add_argument("--soft-template-weight", type=float, default=0.25)
    losses.add_argument("--log-depth-beta", type=float, default=0.1)
    losses.add_argument("--reprojection-beta", type=float, default=0.05)
    losses.add_argument("--direct-translation-beta", type=float, default=0.05)
    losses.add_argument("--direct-depth-beta", type=float, default=0.05)
    losses.add_argument("--direct-rotation-beta", type=float, default=0.05)
    losses.add_argument(
        "--direct-translation-scale",
        type=float,
        default=None,
        help=(
            "Normalization scale in scene-gt translation units. Defaults to "
            "1000 for mm datasets and 1.0 for metre datasets."
        ),
    )
    losses.add_argument(
        "--direct-depth-scale",
        type=float,
        default=None,
        help=(
            "Normalization scale in scene-gt depth units. Defaults to 1000 "
            "for mm datasets and 1.0 for metre datasets."
        ),
    )
    losses.add_argument("--softmax-temperature", type=float, default=0.1)
    losses.add_argument("--target-temperature-deg", type=float, default=15.0)
    losses.add_argument(
        "--pose-translation-unit",
        choices=("mm", "m"),
        default="mm",
        help="Unit used by cam_t_m2c in scene_gt. Assetto datasets use mm.",
    )

    heavy = parser.add_argument_group("optional dedicated heavy validation")
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
        raise FileNotFoundError(f"Pretrained checkpoint not found: {args.checkpoint}")
    if args.validation_interval <= 0:
        raise ValueError("--validation-interval must be positive")
    if (
        args.heavy_validation
        and args.heavy_validation_interval % args.validation_interval != 0
    ):
        raise ValueError(
            "--heavy-validation-interval must be divisible by "
            "--validation-interval"
        )
    nonnegative = {
        "log-depth": args.log_depth_weight,
        "inplane": args.inplane_weight,
        "reprojection": args.reprojection_weight,
        "direct translation": args.direct_translation_weight,
        "direct depth": args.direct_depth_weight,
        "direct rotation": args.direct_rotation_weight,
        "direct reprojection": args.direct_reprojection_weight,
        "InfoNCE": args.info_nce_weight,
        "soft-template": args.soft_template_weight,
    }
    invalid = [name for name, value in nonnegative.items() if value < 0]
    if invalid:
        raise ValueError(f"Loss weights must be nonnegative: {invalid}")
    positive = {
        "direct translation scale": args.direct_translation_scale,
        "direct depth scale": args.direct_depth_scale,
    }
    invalid_positive = [
        name for name, value in positive.items() if value is not None and value <= 0
    ]
    if invalid_positive:
        raise ValueError(f"Scales must be positive: {invalid_positive}")


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
    configure_logger(cfg, args, output_dir)

    cfg.model._target_ = (
        "fine_tuning.pose_aware_training.model.PoseAwareGigaPose"
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
    default_translation_scale = 1000.0 if args.pose_translation_unit == "mm" else 1.0
    default_depth_scale = 1000.0 if args.pose_translation_unit == "mm" else 1.0
    cfg.model.pose_loss_config = {
        "log_depth_weight": args.log_depth_weight,
        "inplane_weight": args.inplane_weight,
        "reprojection_weight": args.reprojection_weight,
        "optimize_pose_monitor_errors": args.optimize_pose_monitor_errors,
        "direct_translation_weight": args.direct_translation_weight,
        "direct_depth_weight": args.direct_depth_weight,
        "direct_rotation_weight": args.direct_rotation_weight,
        "direct_reprojection_weight": args.direct_reprojection_weight,
        "info_nce_weight": args.info_nce_weight,
        "soft_template_weight": args.soft_template_weight,
        "log_depth_beta": args.log_depth_beta,
        "reprojection_beta": args.reprojection_beta,
        "direct_translation_beta": args.direct_translation_beta,
        "direct_depth_beta": args.direct_depth_beta,
        "direct_rotation_beta": args.direct_rotation_beta,
        "direct_translation_scale": (
            args.direct_translation_scale
            if args.direct_translation_scale is not None
            else default_translation_scale
        ),
        "direct_depth_scale": (
            args.direct_depth_scale
            if args.direct_depth_scale is not None
            else default_depth_scale
        ),
        "softmax_temperature": args.softmax_temperature,
        "target_temperature_deg": args.target_temperature_deg,
        "translation_to_mm": (
            1.0 if args.pose_translation_unit == "mm" else 1000.0
        ),
    }
    if args.match_sim_threshold is not None:
        cfg.model.testing_metric.sim_threshold = args.match_sim_threshold
    if args.match_patch_threshold is not None:
        cfg.model.testing_metric.patch_threshold = args.match_patch_threshold

    train_dataset = instantiate(
        make_dataset_config(
            cfg, args, args.train_split, augment=True
        )
    )
    validation_dataset = instantiate(
        make_dataset_config(
            cfg, args, args.validation_split, augment=False
        )
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
        PoseMetricHistoryCallback(
            output_dir / "pose_metrics.csv",
            train_interval=args.metric_csv_every,
        )
    )
    if heavy_callback is not None:
        trainer.callbacks.append(heavy_callback)

    logger.info("Pose-aware training mode: %s", args.nets_to_train)
    logger.info("Outputs: %s", output_dir)
    logger.info("Metric history: %s", output_dir / "pose_metrics.csv")
    logger.info(
        "Optimized IST losses: log-depth=%.3g inplane=%.3g reprojection=%.3g",
        args.log_depth_weight,
        args.inplane_weight,
        args.reprojection_weight,
    )
    if args.nets_to_train == "all":
        logger.info(
            "Optimized retrieval losses: InfoNCE=%.3g soft-template=%.3g",
            args.info_nce_weight,
            args.soft_template_weight,
        )
    if args.optimize_pose_monitor_errors:
        logger.info(
            "Direct pose losses enabled: translation=%.3g depth=%.3g "
            "rotation=%.3g reprojection-extra=%.3g",
            args.direct_translation_weight,
            args.direct_depth_weight,
            args.direct_rotation_weight,
            args.direct_reprojection_weight,
        )
    else:
        logger.info(
            "Translation, depth, full rotation, and raw pixel reprojection "
            "errors are monitoring-only metrics and do not contribute gradients."
        )

    trainer.fit(
        model,
        train_dataloaders=[train_loader],
        val_dataloaders=validation_loader,
    )


if __name__ == "__main__":
    main()
