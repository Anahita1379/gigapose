"""Train the lag-masked transformer using frozen DINO/RGB/mask/CAD evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader

from tracking.rgb_self_recovery.sliding_window.gated_candidate_transformer import train as base
from tracking.rgb_self_recovery.sliding_window.gated_candidate_transformer.dataset import (
    collection_statistics,
)

from .dataset import VisualTrainingWindowDataset, visual_statistics
from .model import FORMAT, FrozenVisualFusionTransformer


def parser():
    value = base.parser()
    value.description = __doc__
    return value


def _checkpoint(model, args, epoch, metrics, train, validation):
    return {
        "format": FORMAT,
        "model_state": model.state_dict(),
        "model_config": model.config(),
        "epoch": epoch,
        "validation_metrics": metrics,
        "candidate_feature_names": train.candidate_feature_names,
        "frame_feature_names": train.frame_feature_names,
        "saved_candidates": train.candidate_count,
        "train_runs": sorted(train.runs),
        "validation_runs": sorted(validation.runs),
        "frozen_visual_encoder": True,
        "visual_feature_manifests": train.visual_manifests,
        "arguments": {
            key: [str(item) for item in value] if isinstance(value, list)
            else str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }


def main() -> None:
    args = parser().parse_args()
    if not 0 <= args.attention_future_lag < args.window_length:
        raise ValueError("--attention-future-lag must be in [0, --window-length)")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU")
        args.device = "cpu"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_data = VisualTrainingWindowDataset(
        args.data, window_length=args.window_length, stride=args.window_stride
    )
    validation_data = VisualTrainingWindowDataset(
        args.validation_data,
        window_length=args.window_length,
        stride=args.validation_stride,
    )
    train, validation = train_data.collection, validation_data.collection
    overlap = base._validate_collections(train, validation, args.allow_run_overlap)
    if (
        train.candidate_visual_dim != validation.candidate_visual_dim
        or train.frame_visual_dim != validation.frame_visual_dim
    ):
        raise ValueError("Train/validation visual dimensions differ")
    signature_keys = ("checkpoint", "mesh_scale", "center_mesh")
    train_signature = tuple(train.visual_manifests[0].get(key) for key in signature_keys)
    validation_signature = tuple(
        validation.visual_manifests[0].get(key) for key in signature_keys
    )
    if train_signature != validation_signature:
        raise ValueError(
            "Train/validation sidecars use different frozen encoders or CAD conventions"
        )
    statistics = {
        key: torch.from_numpy(value)
        for key, value in {
            **collection_statistics(train),
            **visual_statistics(train),
        }.items()
    }
    model = FrozenVisualFusionTransformer(
        train.candidate_dim,
        train.frame_dim,
        train.candidate_count,
        candidate_visual_dim=train.candidate_visual_dim,
        frame_visual_dim=train.frame_visual_dim,
        window_length=args.window_length,
        model_dim=args.model_dim,
        heads=args.heads,
        feedforward_dim=args.feedforward_dim,
        cross_layers=args.cross_layers,
        dropout=args.dropout,
        attention_future_lag=args.attention_future_lag,
        use_translation_fixed_lag=not args.disable_translation_fixed_lag,
        translation_fixed_lag_layers=args.translation_fixed_lag_layers,
        translation_position_scale_m=args.translation_position_scale_m,
        translation_relative_scale_m=args.translation_relative_scale_m,
        maximum_translation_refinement_m=args.maximum_translation_refinement_m,
        statistics=statistics,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loaders = {
        "train": DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"),
        ),
        "validation": DataLoader(
            validation_data, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"),
        ),
    }
    history, best_loss, best_pose, stale = [], float("inf"), float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = base._epoch(model, loaders["train"], args.device, args, optimizer)
        validation_metrics = base._epoch(model, loaders["validation"], args.device, args)
        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in validation_metrics.items()})
        history.append(row)
        payload = _checkpoint(model, args, epoch, validation_metrics, train, validation)
        torch.save(payload, args.output_dir / "last.ckpt")
        if validation_metrics["loss"] < best_loss - args.min_delta:
            best_loss, stale = validation_metrics["loss"], 0
            torch.save(payload, args.output_dir / "best.ckpt")
            torch.save(payload, args.output_dir / "best_loss.ckpt")
        else:
            stale += 1
        if validation_metrics["pose_score"] < best_pose:
            best_pose = validation_metrics["pose_score"]
            torch.save(payload, args.output_dir / "best_pose.ckpt")
        print(
            f"epoch={epoch:03d} train={train_metrics['loss']:.4f} "
            f"val={validation_metrics['loss']:.4f} "
            f"pose={validation_metrics['pose_score']:.4f} "
            f"oracle_acc={validation_metrics['oracle_accuracy']:.3f} "
            f"orientation_acc={validation_metrics['orientation_accuracy']:.3f} "
            f"stale={stale}/{args.patience}", flush=True,
        )
        if stale >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break
    with (args.output_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    report = {
        "format": "rgb_self_recovery_frozen_visual_fusion_training_v2",
        "best_validation_loss": best_loss,
        "best_validation_pose_score": best_pose,
        "pose_score_definition": (
            "selected translation RMSE / translation_scale_m + selected rotation "
            "RMSE / rotation_scale_deg; includes learned translation fixed lag"
        ),
        "epochs_completed": len(history),
        "train_frames": sum(len(bundle) for bundle in train.bundles),
        "validation_frames": sum(len(bundle) for bundle in validation.bundles),
        "train_windows": len(train_data),
        "validation_windows": len(validation_data),
        "train_runs": sorted(train.runs),
        "validation_runs": sorted(validation.runs),
        "run_overlap": overlap,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_visual_encoder_parameters_in_transformer": 0,
        "best_loss_checkpoint": str(args.output_dir / "best_loss.ckpt"),
        "best_checkpoint_alias": str(args.output_dir / "best.ckpt"),
        "recommended_pose_checkpoint": str(args.output_dir / "best_pose.ckpt"),
    }
    (args.output_dir / "run_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
