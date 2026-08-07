"""Run the gated candidate transformer with fixed-lag safety decoding."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tracking.geometry import rotation_error_deg, so3_exp
from tracking.io import pose_to_csv_row, write_tracking_csv

from .dataset import BundleCollection, FixedLagWindowDataset
from .model import load_transformer
from .orientation import AdaptiveFixedLagOrientationGuard


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Exported candidate bundle")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fixed-lag", type=int, default=4)
    parser.add_argument("--orientation-gate-weight", type=float, default=1.0)
    parser.add_argument("--minimum-accept-probability", type=float, default=0.50)
    parser.add_argument("--minimum-candidate-probability", type=float, default=0.05)
    parser.add_argument("--minimum-translation-trust", type=float, default=0.0,
                        help="Optional learned fallback; zero disables it.")
    parser.add_argument("--minimum-rotation-trust", type=float, default=0.0,
                        help="Optional learned fallback; zero disables it.")
    parser.add_argument("--max-translation-uncertainty-m", type=float, default=20.0)
    parser.add_argument("--max-rotation-uncertainty-deg", type=float, default=90.0)
    parser.add_argument("--max-correction-translation-m", type=float, default=20.0)
    parser.add_argument("--max-correction-rotation-deg", type=float, default=45.0)
    parser.add_argument("--soft-rotation-jump-deg", type=float, default=45.0)
    parser.add_argument("--hard-rotation-jump-deg", type=float, default=90.0)
    parser.add_argument("--flip-jump-deg", type=float, default=135.0)
    parser.add_argument("--provisional-confirmation-frames", type=int, default=2)
    parser.add_argument("--stable-after-frames", type=int, default=4)
    parser.add_argument("--stable-confirmation-frames", type=int, default=5)
    parser.add_argument("--flip-consistency-deg", type=float, default=45.0)
    parser.add_argument("--minimum-orientation-probability", type=float, default=0.50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    suffix = values.shape[3:]
    expanded = indices[..., None]
    for _ in suffix:
        expanded = expanded.unsqueeze(-1)
    return values.gather(2, expanded.expand(*indices.shape, 1, *suffix)).squeeze(2)


def _validate_fixed_lag(fixed_lag: int, model_args: dict) -> int:
    window_length = int(model_args["window_length"])
    if not 0 <= fixed_lag < window_length:
        raise ValueError("--fixed-lag must be in [0, trained window length)")
    trained_attention_lag = int(model_args["attention_future_lag"])
    if fixed_lag != trained_attention_lag:
        raise ValueError(
            f"--fixed-lag={fixed_lag} differs from the checkpoint's "
            f"attention_future_lag={trained_attention_lag}; retrain with the "
            "desired lag to avoid train/inference attention mismatch"
        )
    return trained_attention_lag


def _decode_window(
    batch: dict[str, torch.Tensor],
    output: dict[str, torch.Tensor],
    args: argparse.Namespace,
    translation_scale_m: float,
    rotation_scale_deg: float,
) -> dict[str, torch.Tensor]:
    valid = batch["candidate_valid"].bool()
    orientation_log_probability = torch.nn.functional.logsigmoid(output["orientation_logits"])
    logits = output["candidate_logits"] + args.orientation_gate_weight * orientation_log_probability
    logits = logits.masked_fill(~valid, -1e9)
    probability = torch.softmax(logits, dim=-1)
    selected = logits.argmax(dim=-1)

    candidate_pose = _gather(batch["poses"], selected)
    residual = _gather(output["residual"], selected)
    fixed_lag_translation = (
        output["translation_fixed_lag_gate"][..., None]
        * output["translation_fixed_lag_delta"]
    )
    uncertainty = _gather(output["log_sigma"], selected)
    translation_trust_probability = torch.sigmoid(
        output["translation_trust_logits"].gather(2, selected[..., None]).squeeze(-1)
    )
    rotation_trust_probability = torch.sigmoid(
        output["rotation_trust_logits"].gather(2, selected[..., None]).squeeze(-1)
    )
    candidate_probability = probability.gather(2, selected[..., None]).squeeze(-1)
    orientation_probability = torch.sigmoid(
        output["orientation_logits"].gather(2, selected[..., None]).squeeze(-1)
    )
    translation_sigma = uncertainty[..., 0].exp() * translation_scale_m
    rotation_sigma = uncertainty[..., 1].exp() * rotation_scale_deg
    accept_probability = torch.sigmoid(output["abstention_logits"])

    corrected = candidate_pose.clone()
    # Residual heads are trained directly in metres and radians. The scales are
    # normalization constants for loss/uncertainty only.
    initial_translation_correction = residual[..., :3]
    translation_correction = initial_translation_correction + fixed_lag_translation
    rotation_correction = residual[..., 3:]
    translation_norm = torch.linalg.vector_norm(translation_correction, dim=-1)
    rotation_norm_deg = torch.rad2deg(torch.linalg.vector_norm(rotation_correction, dim=-1))
    translation_correction_allowed = (
        translation_norm <= args.max_correction_translation_m
    )
    rotation_correction_allowed = (
        rotation_norm_deg <= args.max_correction_rotation_deg
    )
    correction_allowed = translation_correction_allowed & rotation_correction_allowed
    corrected[..., :3, 3] += (
        translation_correction * translation_correction_allowed[..., None]
    )

    flat_pose = corrected.reshape(-1, 4, 4)
    flat_rotvec = rotation_correction.reshape(-1, 3)
    flat_allowed = rotation_correction_allowed.reshape(-1)
    for index in torch.nonzero(flat_allowed, as_tuple=False).flatten().tolist():
        delta = torch.as_tensor(
            so3_exp(flat_rotvec[index].detach().cpu().numpy()),
            device=corrected.device, dtype=corrected.dtype,
        )
        flat_pose[index, :3, :3] = flat_pose[index, :3, :3] @ delta

    # Preserve the learned proposal before any abstention/trust fallback.  This
    # is diagnostic only and lets held-out evaluation measure whether fallback
    # decisions were actually correct; inference never uses ground truth.
    proposed = corrected.clone()

    abstained = (
        (accept_probability < args.minimum_accept_probability)
        | (candidate_probability < args.minimum_candidate_probability)
        | (translation_sigma > args.max_translation_uncertainty_m)
        | (rotation_sigma > args.max_rotation_uncertainty_deg)
    )
    # Candidate zero is the unmodified GigaPose measurement in the exported bundles.
    baseline = batch["poses"][:, :, 0]
    corrected = torch.where(abstained[..., None, None], baseline, corrected)
    orientation_probability = torch.where(abstained, torch.ones_like(orientation_probability), orientation_probability)
    translation_fallback = abstained | (translation_trust_probability < args.minimum_translation_trust)
    rotation_fallback = abstained | (
        rotation_trust_probability < args.minimum_rotation_trust
    )
    # Translation and rotation fallback independently to candidate-zero
    # GigaPose. Adaptive orientation and fixed-lag handling run afterward.
    corrected[..., :3, 3] = torch.where(
        translation_fallback[..., None], baseline[..., :3, 3], corrected[..., :3, 3]
    )
    corrected[..., :3, :3] = torch.where(
        rotation_fallback[..., None, None],
        baseline[..., :3, :3],
        corrected[..., :3, :3],
    )
    orientation_probability = torch.where(
        rotation_fallback,
        torch.ones_like(orientation_probability),
        orientation_probability,
    )
    return {
        "pose": corrected,
        "proposed_pose": proposed,
        "selected": selected,
        "candidate_probability": candidate_probability,
        "orientation_probability": orientation_probability,
        "accept_probability": accept_probability,
        "translation_trust_probability": translation_trust_probability,
        "translation_fallback": translation_fallback,
        "rotation_trust_probability": rotation_trust_probability,
        "rotation_fallback": rotation_fallback,
        "translation_sigma_m": translation_sigma,
        "rotation_sigma_deg": rotation_sigma,
        "translation_correction_m": translation_norm,
        "initial_translation_correction_m": torch.linalg.vector_norm(
            initial_translation_correction, dim=-1
        ),
        "fixed_lag_translation_correction_m": torch.linalg.vector_norm(
            fixed_lag_translation, dim=-1
        ),
        "fixed_lag_translation_gate": output["translation_fixed_lag_gate"],
        "rotation_correction_deg": rotation_norm_deg,
        "translation_correction_allowed": translation_correction_allowed,
        "rotation_correction_allowed": rotation_correction_allowed,
        "correction_allowed": correction_allowed,
        "abstained": abstained,
    }


def _scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf8")
    return value


def _summary(rows: list[dict]) -> dict:
    def stats(key: str) -> dict:
        values = np.asarray([row[key] for row in rows if row.get(key) is not None], dtype=float)
        return {
            "count": int(values.size),
            "mean": float(values.mean()) if values.size else None,
            "median": float(np.median(values)) if values.size else None,
            "p90": float(np.percentile(values, 90)) if values.size else None,
        }

    return {
        "format": "gated_candidate_transformer_inference_v5",
        "rows": len(rows),
        "abstention_fraction": float(np.mean([row["abstained"] for row in rows])) if rows else None,
        "learned_translation_fallback_fraction": float(
            np.mean([row["translation_fallback"] for row in rows])
        ) if rows else None,
        "learned_rotation_fallback_fraction": float(
            np.mean([row["rotation_fallback"] for row in rows])
        ) if rows else None,
        "temporal_orientation_fallback_fraction": float(
            np.mean([row["temporal_orientation_fallback"] for row in rows])
        ) if rows else None,
        "confirmed_flip_count": int(sum(row["flip_confirmed"] for row in rows)),
        "fixed_lag_translation_correction_m": stats(
            "fixed_lag_translation_correction_m"
        ),
        "baseline_translation_error_m": stats("baseline_translation_error_m"),
        "final_translation_error_m": stats("final_translation_error_m"),
        "oracle_translation_fallback_error_m": stats("oracle_translation_fallback_error_m"),
        "baseline_rotation_error_deg": stats("baseline_rotation_error_deg"),
        "final_rotation_error_deg": stats("final_rotation_error_deg"),
        "oracle_rotation_fallback_error_deg": stats(
            "oracle_rotation_fallback_error_deg"
        ),
    }


def main() -> None:
    args = _args()
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {args.output_dir}; pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint_header = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint_header.get("format") == (
        "rgb_self_recovery_frozen_visual_fusion_transformer_v2"
    ):
        from tracking.rgb_self_recovery.sliding_window.frozen_visual_fusion_transformer.dataset import (
            VisualFixedLagWindowDataset,
        )
        from tracking.rgb_self_recovery.sliding_window.frozen_visual_fusion_transformer.model import (
            load_transformer as load_visual_transformer,
        )

        model, checkpoint = load_visual_transformer(args.checkpoint, device)
        dataset_class = VisualFixedLagWindowDataset
        model_name = "frozen_visual_fusion_transformer"
    else:
        model, checkpoint = load_transformer(args.checkpoint, device)
        dataset_class = FixedLagWindowDataset
        model_name = "gated_candidate_transformer"
    model_args = checkpoint["model_config"]
    train_args = checkpoint.get("arguments", {})
    window_length = int(model_args["window_length"])
    trained_attention_lag = _validate_fixed_lag(args.fixed_lag, model_args)

    collection = BundleCollection([args.data])
    if collection.candidate_dim != int(model_args["candidate_dim"]):
        raise ValueError("Candidate feature dimension does not match checkpoint")
    if collection.frame_dim != int(model_args["frame_dim"]):
        raise ValueError("Frame feature dimension does not match checkpoint")
    dataset = dataset_class(
        args.data, window_length=window_length, fixed_lag=args.fixed_lag
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    translation_scale = float(train_args.get("translation_scale_m", 1.0))
    rotation_scale = float(train_args.get("rotation_scale_deg", 20.0))

    predictions: dict[int, dict] = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            tensor_batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            output = model(
                tensor_batch["candidate_features"], tensor_batch["frame_features"],
                tensor_batch["candidate_valid"], tensor_batch["frame_valid"],
                candidate_visual_features=tensor_batch.get("candidate_visual_features"),
                frame_visual_features=tensor_batch.get("frame_visual_features"),
                candidate_poses=tensor_batch["poses"],
                time_s=tensor_batch["time_s"],
            )
            decoded = _decode_window(tensor_batch, output, args, translation_scale, rotation_scale)
            for item in range(tensor_batch["global_indices"].shape[0]):
                target = dataset.target_position
                global_index = int(tensor_batch["global_indices"][item, target].item())
                future = []
                for position in range(target, window_length):
                    if bool(tensor_batch["frame_valid"][item, position]):
                        future.append((
                            decoded["pose"][item, position].cpu().numpy(),
                            float(decoded["orientation_probability"][item, position]),
                        ))
                predictions[global_index] = {
                    "pose": decoded["pose"][item, target].cpu().numpy(),
                    "proposed_pose": decoded["proposed_pose"][item, target].cpu().numpy(),
                    "future": future,
                    **{
                        key: _scalar(value[item, target].cpu().numpy())
                        for key, value in decoded.items()
                        if key not in {"pose", "proposed_pose"}
                    },
                }

    guard = AdaptiveFixedLagOrientationGuard(
        soft_start_deg=args.soft_rotation_jump_deg,
        hard_limit_deg=args.hard_rotation_jump_deg,
        flip_min_deg=args.flip_jump_deg,
        provisional_confirmation=args.provisional_confirmation_frames,
        stable_after_frames=args.stable_after_frames,
        stable_confirmation=args.stable_confirmation_frames,
        consistency_deg=args.flip_consistency_deg,
        minimum_orientation_probability=args.minimum_orientation_probability,
    )
    csv_rows, diagnostics = [], []
    last_segment = None
    for global_index in sorted(predictions):
        bundle = collection.bundles[0]
        local = global_index
        arrays = bundle.arrays
        segment = str(_scalar(arrays["segment_id"][local]))
        if segment != last_segment:
            guard.reset()
            last_segment = segment
        prediction = predictions[global_index]
        timestamp = float(arrays["time_s"][local])
        decision = guard.resolve(
            prediction["pose"],
            time_s=timestamp,
            future_poses=[item[0] for item in prediction["future"]],
            future_orientation_probabilities=[item[1] for item in prediction["future"]],
        )
        final_pose = decision.pose
        proposed_pose = prediction["proposed_pose"]
        baseline_pose = np.asarray(arrays["poses"][local, 0], dtype=float)
        ground_truth = np.asarray(arrays["ground_truth_pose"][local], dtype=float)

        row = pose_to_csv_row(
            scene_id=int(arrays["scene_id"][local]), im_id=int(arrays["im_id"][local]),
            track_id=0, obj_id=1, pose_m=final_pose,
            confidence=float(prediction["candidate_probability"] * prediction["accept_probability"]),
            mode=model_name, source=model_name,
            elapsed_s=0.0,
        )
        csv_rows.append(row)
        diagnostics.append({
            "global_index": global_index,
            "scene_id": int(arrays["scene_id"][local]),
            "im_id": int(arrays["im_id"][local]),
            "instance_id": 0,
            "segment_id": segment,
            "source_run": str(_scalar(arrays["sequence_run"][local])),
            "camera_id": str(_scalar(arrays["sequence_camera"][local])),
            "selected_candidate": int(prediction["selected"]),
            "candidate_probability": float(prediction["candidate_probability"]),
            "orientation_probability": float(prediction["orientation_probability"]),
            "accept_probability": float(prediction["accept_probability"]),
            "translation_trust_probability": float(prediction["translation_trust_probability"]),
            "translation_fallback": bool(prediction["translation_fallback"]),
            "rotation_trust_probability": float(prediction["rotation_trust_probability"]),
            "rotation_fallback": bool(prediction["rotation_fallback"]),
            "translation_uncertainty_m": float(prediction["translation_sigma_m"]),
            "rotation_uncertainty_deg": float(prediction["rotation_sigma_deg"]),
            "translation_correction_m": float(prediction["translation_correction_m"]),
            "initial_translation_correction_m": float(
                prediction["initial_translation_correction_m"]
            ),
            "fixed_lag_translation_correction_m": float(
                prediction["fixed_lag_translation_correction_m"]
            ),
            "fixed_lag_translation_gate": float(
                prediction["fixed_lag_translation_gate"]
            ),
            "rotation_correction_deg": float(prediction["rotation_correction_deg"]),
            "translation_correction_allowed": bool(
                prediction["translation_correction_allowed"]
            ),
            "rotation_correction_allowed": bool(
                prediction["rotation_correction_allowed"]
            ),
            "correction_allowed": bool(prediction["correction_allowed"]),
            "abstained": bool(prediction["abstained"]),
            "temporal_orientation_fallback": decision.temporal_fallback,
            "flip_confirmed": decision.flip_confirmed,
            "flip_confirmation_required": decision.required_confirmation,
            "temporal_rotation_jump_deg": decision.rotation_jump_deg,
            "stable_orientation_frames": decision.stable_frames,
            "baseline_translation_error_m": float(np.linalg.norm(baseline_pose[:3, 3] - ground_truth[:3, 3])),
            "proposed_translation_error_m": float(np.linalg.norm(proposed_pose[:3, 3] - ground_truth[:3, 3])),
            "final_translation_error_m": float(np.linalg.norm(final_pose[:3, 3] - ground_truth[:3, 3])),
            "oracle_translation_fallback_error_m": float(min(
                np.linalg.norm(final_pose[:3, 3] - ground_truth[:3, 3]),
                np.linalg.norm(baseline_pose[:3, 3] - ground_truth[:3, 3]),
            )),
            "baseline_rotation_error_deg": float(rotation_error_deg(baseline_pose, ground_truth)),
            "proposed_rotation_error_deg": float(rotation_error_deg(proposed_pose, ground_truth)),
            "final_rotation_error_deg": float(rotation_error_deg(final_pose, ground_truth)),
            "oracle_rotation_fallback_error_deg": float(min(
                rotation_error_deg(final_pose, ground_truth),
                rotation_error_deg(baseline_pose, ground_truth),
            )),
        })

    write_tracking_csv(args.output_dir / "tracked_predictions.csv", csv_rows)
    with (args.output_dir / "diagnostics.jsonl").open("w") as handle:
        for row in diagnostics:
            handle.write(json.dumps(row) + "\n")
    report = _summary(diagnostics)
    report.update({
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "fixed_lag": args.fixed_lag,
        "attention_future_lag": trained_attention_lag,
        "model": model_name,
        "checkpoint_format": checkpoint.get("format"),
    })
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
