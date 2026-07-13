"""Run end-to-end inference with a trained residual-pose checkpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from fine_tuning.train import parse_devices
from src.utils.logging import get_logger
from src.utils.weight import load_checkpoint


logger = get_logger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--root-dir", type=Path, default=Path("gigaPose_datasets/datasets")
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--max-num-dets-per-forward", type=int, default=4)
    parser.add_argument("--match-sim-threshold", type=float, default=0.5)
    parser.add_argument("--match-patch-threshold", type=float, default=3.0)
    parser.add_argument("--residual-hidden-dim", type=int, default=256)
    parser.add_argument(
        "--apply-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply the trained residual heads. Use --no-apply-residual to "
            "produce a matched baseline from the same checkpoint."
        ),
    )
    parser.add_argument(
        "--rotation-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Must match the setting used during training.",
    )
    parser.add_argument("--max-center-offset-px", type=float, default=56.0)
    parser.add_argument("--max-log-depth-residual", type=float, default=0.5)
    parser.add_argument("--max-rotation-deg", type=float, default=20.0)
    parser.add_argument(
        "--pose-translation-unit", choices=("mm", "m"), default="mm"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    with initialize_config_dir(
        version_base=None,
        config_dir=str((REPO_ROOT / "configs").resolve()),
    ):
        cfg = compose(config_name="test")
    OmegaConf.set_struct(cfg, False)

    output_dir = (
        REPO_ROOT / "gigaPose_datasets" / "results" / args.run_name
    ).resolve()
    os.makedirs(output_dir, exist_ok=True)
    cfg.save_dir = str(output_dir)
    cfg.name_exp = args.run_name
    cfg.test_dataset_name = args.dataset_name
    cfg.run_id = args.run_name
    cfg.machine.batch_size = args.batch_size
    cfg.machine.num_workers = args.num_workers
    cfg.machine.trainer.devices = parse_devices(args.devices, args.device)
    cfg.machine.trainer.logger = False
    cfg.model._target_ = (
        "fine_tuning.residual_pose_training.model.ResidualPoseGigaPose"
    )
    cfg.model.log_dir = str(output_dir)
    cfg.model.testing_metric.sim_threshold = args.match_sim_threshold
    cfg.model.testing_metric.patch_threshold = args.match_patch_threshold
    unit = 1000.0 if args.pose_translation_unit == "mm" else 1.0
    cfg.model.residual_config = {
        "hidden_dim": args.residual_hidden_dim,
        "enable_rotation": args.rotation_residual,
        "apply_at_inference": args.apply_residual,
        "max_center_offset_px": args.max_center_offset_px,
        "max_log_depth_residual": args.max_log_depth_residual,
        "max_rotation_deg": args.max_rotation_deg,
        "translation_unit": unit,
        "translation_to_mm": 1.0 if args.pose_translation_unit == "mm" else 1000.0,
    }

    trainer = instantiate(cfg.machine.trainer)
    model = instantiate(cfg.model)
    load_checkpoint(model, args.checkpoint, checkpoint_key="state_dict")

    test_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg.data.test.dataloader, resolve=True)
    )
    test_cfg.root_dir = str(args.root_dir.resolve())
    test_cfg.dataset_name = args.dataset_name
    test_cfg.batch_size = args.batch_size
    test_cfg.depth_scale = 1.0
    test_cfg.template_config.dir = str((args.root_dir / "templates").resolve())
    test_cfg.template_config.scale_factor = 1.0
    test_cfg.load_gt = False
    test_dataset = instantiate(test_cfg)
    test_loader = DataLoader(
        test_dataset.web_dataloader.datapipeline,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=test_dataset.collate_fn,
    )

    template_cfg = OmegaConf.create(OmegaConf.to_container(test_cfg, resolve=True))
    template_cfg._target_ = "src.dataloader.template.TemplateSet"
    template_dataset = instantiate(template_cfg)
    model.template_datasets = {args.dataset_name: template_dataset}
    model.test_dataset_name = args.dataset_name
    model.max_num_dets_per_forward = args.max_num_dets_per_forward
    model.run_id = args.run_name
    prediction_dir = Path(model.log_dir) / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for prediction_file in prediction_dir.glob("*.npz"):
        prediction_file.unlink()
    model.log_interval = max(1, len(test_loader) // 30)
    logger.info("Residual inference outputs: %s", output_dir)
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
