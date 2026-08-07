"""Train a causal GRU to select recovery candidates or GigaPose fallback."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import CandidateBundle, SequenceClipDataset, feature_statistics
from .model import GRUCandidateSelector


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--data", type=Path, required=True)
    value.add_argument("--validation-data", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--clip-length", type=int, default=8)
    value.add_argument("--clip-stride", type=int, default=1)
    value.add_argument("--validation-stride", type=int, default=8)
    value.add_argument("--candidate-hidden", type=int, default=96)
    value.add_argument("--gru-hidden", type=int, default=128)
    value.add_argument("--gru-layers", type=int, default=1)
    value.add_argument("--dropout", type=float, default=0.1)
    value.add_argument("--epochs", type=int, default=100)
    value.add_argument("--batch-size", type=int, default=16)
    value.add_argument("--num-workers", type=int, default=4)
    value.add_argument("--learning-rate", type=float, default=2e-4)
    value.add_argument("--weight-decay", type=float, default=1e-4)
    value.add_argument("--expected-cost-weight", type=float, default=0.1)
    value.add_argument("--fallback-label-weight", type=float, default=1.0)
    value.add_argument("--max-oracle-cost", type=float, default=20.0)
    value.add_argument("--gradient-clip", type=float, default=5.0)
    value.add_argument("--patience", type=int, default=10)
    value.add_argument("--min-delta", type=float, default=1e-4)
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=20260803)
    value.add_argument("--allow-run-overlap", action="store_true")
    value.add_argument("--overwrite", action="store_true")
    return value


def _validate_splits(train: CandidateBundle, validation: CandidateBundle, allow_overlap: bool):
    for key in ("candidate_feature_names", "frame_feature_names", "saved_candidates"):
        if train.manifest.get(key) != validation.manifest.get(key):
            raise ValueError(f"Train/validation candidate schemas differ: {key}")
    train_runs = set(train.arrays["sequence_run"].tolist())
    validation_runs = set(validation.arrays["sequence_run"].tolist())
    overlap = sorted(train_runs & validation_runs)
    if overlap and not allow_overlap:
        raise ValueError(
            "Train/validation source_run overlap would leak temporal appearance: "
            f"{overlap}. Split by complete run or pass --allow-run-overlap only for a diagnostic."
        )
    return train_runs, validation_runs, overlap


def _step(model, batch, device, args, optimizer=None):
    candidate = batch["candidate_features"].to(device)
    frame = batch["frame_features"].to(device)
    candidate_valid = batch["candidate_valid"].to(device)
    frame_valid = batch["frame_valid"].to(device)
    labels = batch["labels"].to(device)
    costs = batch["oracle_costs"].to(device)
    logits, _ = model(candidate, frame, candidate_valid)
    flat = frame_valid.reshape(-1)
    logits_flat = logits.reshape(-1, logits.shape[-1])[flat]
    labels_flat = labels.reshape(-1)[flat]
    costs_flat = costs.reshape(-1, costs.shape[-1])[flat]
    candidate_flat = candidate_valid.reshape(-1, candidate_valid.shape[-1])[flat]
    ce = F.cross_entropy(logits_flat, labels_flat, reduction="none")
    sample_weight = torch.where(
        labels_flat == 0,
        torch.full_like(ce, args.fallback_label_weight),
        torch.ones_like(ce),
    )
    classification = (ce * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)
    probabilities = logits_flat.softmax(dim=-1)
    safe_costs = torch.where(
        candidate_flat,
        costs_flat.clamp(max=args.max_oracle_cost),
        torch.zeros_like(costs_flat),
    )
    expected_cost = (probabilities * safe_costs).sum(dim=-1).mean()
    loss = classification + args.expected_cost_weight * expected_cost
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
    selected = logits_flat.argmax(dim=-1)
    selected_cost = costs_flat.gather(1, selected[:, None]).squeeze(1)
    return {
        "loss": float(loss.detach()),
        "classification_loss": float(classification.detach()),
        "expected_cost": float(expected_cost.detach()),
        "oracle_accuracy": float((selected == labels_flat).float().mean()),
        "selected_cost": float(selected_cost.mean()),
        "selected_recovery_fraction": float((selected != 0).float().mean()),
        "oracle_recovery_fraction": float((labels_flat != 0).float().mean()),
        "frames": int(len(labels_flat)),
    }


def _epoch(model, loader, device, args, optimizer=None):
    model.train(optimizer is not None)
    totals = {}
    frames = 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for batch in loader:
            metrics = _step(model, batch, device, args, optimizer)
            count = metrics.pop("frames")
            frames += count
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * count
    return {key: value / max(frames, 1) for key, value in totals.items()}


def _checkpoint(model, args, epoch, metrics, train_bundle, validation_bundle):
    return {
        "format": "rgb_self_recovery_gru_selector_v1",
        "model_state": model.state_dict(),
        "model_config": model.config(),
        "epoch": epoch,
        "validation_metrics": metrics,
        "candidate_feature_names": train_bundle.manifest["candidate_feature_names"],
        "frame_feature_names": train_bundle.manifest["frame_feature_names"],
        "saved_candidates": train_bundle.manifest["saved_candidates"],
        "baseline_candidate_index": 0,
        "train_candidate_checkpoint": train_bundle.manifest.get("checkpoint"),
        "validation_candidate_checkpoint": validation_bundle.manifest.get("checkpoint"),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }


def load_selector(path: Path, device: str = "cpu"):
    payload = torch.load(Path(path), map_location=device)
    if payload.get("format") != "rgb_self_recovery_gru_selector_v1":
        raise ValueError(f"Unsupported GRU selector checkpoint: {path}")
    config = payload["model_config"]
    model = GRUCandidateSelector(**config)
    model.load_state_dict(payload["model_state"])
    return model.to(device).eval(), payload


def main() -> None:
    args = parser().parse_args()
    if args.clip_length < 1 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Clip length, epochs, and batch size must be positive")
    if min(args.expected_cost_weight, args.fallback_label_weight) < 0:
        raise ValueError("Loss weights must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU.")
        args.device = "cpu"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_data = SequenceClipDataset(
        args.data, clip_length=args.clip_length, stride=args.clip_stride
    )
    validation_data = SequenceClipDataset(
        args.validation_data,
        clip_length=args.clip_length,
        stride=args.validation_stride,
    )
    train_bundle = train_data.bundle
    validation_bundle = validation_data.bundle
    train_runs, validation_runs, overlap = _validate_splits(
        train_bundle, validation_bundle, args.allow_run_overlap
    )
    statistics = {
        key: torch.from_numpy(value)
        for key, value in feature_statistics(train_bundle).items()
    }
    model = GRUCandidateSelector(
        train_bundle.candidate_dim,
        train_bundle.frame_dim,
        candidate_hidden=args.candidate_hidden,
        gru_hidden=args.gru_hidden,
        gru_layers=args.gru_layers,
        dropout=args.dropout,
        statistics=statistics,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    history = []
    best = float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = _epoch(model, train_loader, args.device, args, optimizer)
        validation_metrics = _epoch(model, validation_loader, args.device, args)
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(row)
        torch.save(
            _checkpoint(
                model, args, epoch, validation_metrics, train_bundle, validation_bundle
            ),
            args.output_dir / "last.ckpt",
        )
        if validation_metrics["loss"] < best - args.min_delta:
            best = validation_metrics["loss"]
            stale = 0
            torch.save(
                _checkpoint(
                    model, args, epoch, validation_metrics, train_bundle, validation_bundle
                ),
                args.output_dir / "best.ckpt",
            )
        else:
            stale += 1
        print(
            f"epoch={epoch:03d} train={train_metrics['loss']:.5f} "
            f"val={validation_metrics['loss']:.5f} "
            f"oracle_acc={validation_metrics['oracle_accuracy']:.3f} "
            f"selected_cost={validation_metrics['selected_cost']:.3f} "
            f"recovery={validation_metrics['selected_recovery_fraction']:.3f} "
            f"stale={stale}/{args.patience}",
            flush=True,
        )
        if stale >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    fields = list(history[0])
    with (args.output_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    report = {
        "format": "rgb_self_recovery_gru_training_v1",
        "best_validation_loss": best,
        "epochs_completed": len(history),
        "train_runs": sorted(train_runs),
        "validation_runs": sorted(validation_runs),
        "run_overlap": overlap,
        "train_frames": len(train_bundle),
        "validation_frames": len(validation_bundle),
        "train_clips": len(train_data),
        "validation_clips": len(validation_data),
        "recommended_checkpoint": str(args.output_dir / "best.ckpt"),
    }
    (args.output_dir / "run_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
