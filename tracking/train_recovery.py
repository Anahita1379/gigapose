"""Train the optional SE(3) correction, confidence, and ranking head."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tracking.recovery import FEATURE_NAMES, PoseRecoveryHead, bounded_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--groups-per-batch", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--rotation-weight", type=float, default=1.0)
    parser.add_argument("--translation-weight", type=float, default=1.0)
    parser.add_argument("--confidence-weight", type=float, default=0.25)
    parser.add_argument("--quality-weight", type=float, default=0.20)
    parser.add_argument("--ranking-weight", type=float, default=0.20)
    parser.add_argument("--ranking-margin", type=float, default=0.10)
    parser.add_argument("--max-rotation-deg", type=float, default=180.0)
    parser.add_argument("--max-translation-m", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _group_batches(
    groups: np.ndarray,
    selected_groups: np.ndarray,
    groups_per_batch: int,
    rng: np.random.Generator,
):
    values = selected_groups.copy()
    rng.shuffle(values)
    for start in range(0, len(values), groups_per_batch):
        group_chunk = values[start : start + groups_per_batch]
        yield np.flatnonzero(np.isin(groups, group_chunk))


def _loss(
    output: dict[str, torch.Tensor],
    rotation_target: torch.Tensor,
    translation_target: torch.Tensor,
    confidence_target: torch.Tensor,
    quality_target: torch.Tensor,
    groups: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    max_rotation = math.radians(args.max_rotation_deg)
    predicted_rotation = bounded_vector(output["rotation_raw"], max_rotation)
    predicted_translation = bounded_vector(
        output["translation_raw"], args.max_translation_m
    )
    rotation_loss = nn.functional.smooth_l1_loss(
        predicted_rotation / max(max_rotation, 1e-6),
        rotation_target / max(max_rotation, 1e-6),
    )
    translation_loss = nn.functional.smooth_l1_loss(
        predicted_translation / max(args.max_translation_m, 1e-6),
        translation_target / max(args.max_translation_m, 1e-6),
    )
    confidence_loss = nn.functional.binary_cross_entropy_with_logits(
        output["confidence_logit"], confidence_target
    )
    quality_loss = nn.functional.smooth_l1_loss(
        output["quality"], torch.log1p(quality_target)
    )
    ranking_terms = []
    for group in groups.unique():
        indices = torch.nonzero(groups == group, as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        target = quality_target[indices]
        prediction = output["quality"][indices]
        differences = target[:, None] - target[None, :]
        better, worse = torch.where(differences < -1e-5)
        if better.numel():
            ranking_terms.append(
                nn.functional.relu(
                    args.ranking_margin
                    + prediction[better]
                    - prediction[worse]
                ).mean()
            )
    ranking_loss = (
        torch.stack(ranking_terms).mean()
        if ranking_terms
        else output["quality"].sum() * 0.0
    )
    total = (
        args.rotation_weight * rotation_loss
        + args.translation_weight * translation_loss
        + args.confidence_weight * confidence_loss
        + args.quality_weight * quality_loss
        + args.ranking_weight * ranking_loss
    )
    return total, {
        "rotation": float(rotation_loss.detach()),
        "translation": float(translation_loss.detach()),
        "confidence": float(confidence_loss.detach()),
        "quality": float(quality_loss.detach()),
        "ranking": float(ranking_loss.detach()),
        "total": float(total.detach()),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    payload = np.load(args.data)
    names = tuple(str(value) for value in payload["feature_names"])
    if names != FEATURE_NAMES:
        raise ValueError("Input feature definition differs from tracking.recovery.")
    arrays = {name: payload[name] for name in payload.files}
    unique_groups = np.unique(arrays["group_ids"])
    rng.shuffle(unique_groups)
    validation_count = max(1, int(round(len(unique_groups) * args.validation_fraction)))
    validation_groups = unique_groups[:validation_count]
    training_groups = unique_groups[validation_count:]
    if training_groups.size == 0:
        raise ValueError("Not enough groups for a train/validation split.")
    training_mask = np.isin(arrays["group_ids"], training_groups)
    feature_mean = arrays["features"][training_mask].mean(axis=0)
    feature_std = arrays["features"][training_mask].std(axis=0)
    feature_std = np.maximum(feature_std, 1e-6)
    normalized_features = (arrays["features"] - feature_mean) / feature_std

    tensors = {
        "features": torch.as_tensor(normalized_features, dtype=torch.float32),
        "rotation": torch.as_tensor(arrays["rotation_targets"], dtype=torch.float32),
        "translation": torch.as_tensor(arrays["translation_targets"], dtype=torch.float32),
        "confidence": torch.as_tensor(arrays["confidence_targets"], dtype=torch.float32),
        "quality": torch.as_tensor(arrays["quality_targets"], dtype=torch.float32),
        "groups": torch.as_tensor(arrays["group_ids"], dtype=torch.long),
    }
    model = PoseRecoveryHead(len(FEATURE_NAMES), args.hidden_dim).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    stale = 0
    history = []

    def run_groups(selected: np.ndarray, training: bool) -> float:
        model.train(training)
        epoch_losses = []
        batches = _group_batches(
            arrays["group_ids"],
            selected,
            args.groups_per_batch,
            rng if training else np.random.default_rng(0),
        )
        for indices in batches:
            index = torch.as_tensor(indices, dtype=torch.long)
            batch = {
                key: value[index].to(args.device) for key, value in tensors.items()
            }
            with torch.set_grad_enabled(training):
                output = model(batch["features"])
                loss, _ = _loss(
                    output,
                    batch["rotation"],
                    batch["translation"],
                    batch["confidence"],
                    batch["quality"],
                    batch["groups"],
                    args,
                )
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
            epoch_losses.append(float(loss.detach()))
        return float(np.mean(epoch_losses))

    def save(name: str, epoch: int, validation_loss: float) -> None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_config": {
                    "hidden_dim": args.hidden_dim,
                    "max_rotation_rad": math.radians(args.max_rotation_deg),
                    "max_translation_m": args.max_translation_m,
                },
                "feature_names": FEATURE_NAMES,
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "epoch": epoch,
                "validation_loss": validation_loss,
                "arguments": vars(args),
            },
            args.output_dir / name,
        )

    for epoch in range(1, args.epochs + 1):
        train_loss = run_groups(training_groups, True)
        val_loss = run_groups(validation_groups, False)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        save("last.ckpt", epoch, val_loss)
        improved = val_loss < best_loss - args.min_delta
        if improved:
            best_loss, stale = val_loss, 0
            save("best.ckpt", epoch, val_loss)
        else:
            stale += 1
        print(
            f"epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f} "
            f"best={best_loss:.6f} stale={stale}/{args.patience}"
        )
        if stale >= args.patience:
            print("Early stopping: validation recovery objective stopped improving.")
            break
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    report = {
        "best_validation_loss": best_loss,
        "epochs_completed": len(history),
        "training_groups": int(training_groups.size),
        "validation_groups": int(validation_groups.size),
        "best_checkpoint": str(args.output_dir / "best.ckpt"),
    }
    (args.output_dir / "run_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
