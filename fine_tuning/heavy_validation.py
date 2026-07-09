"""Dedicated fixed-subset heavy validation for GigaPose fine-tuning.

This module deliberately stays separate from the ordinary training and
light-validation loaders. Only rank zero loads full RGB frames, runs the full
retrieval/IST/RANSAC/pose-recovery path, and renders CAD overlays. Other DDP
ranks wait at synchronized boundaries and receive a shared failure flag.
"""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset
import trimesh

from fine_tuning.ac_geometry import InstanceRenderer, bbox_from_mask
from src.utils.logging import get_logger, log_image


logger = get_logger(__name__)

COLORS = [
    (0, 255, 80),
    (20, 140, 255),
    (255, 90, 30),
    (210, 40, 255),
    (255, 210, 20),
]


def build_fixed_heavy_loader(
    dataset,
    num_frames: int,
    seed: int,
    selection_path: Path,
) -> tuple[DataLoader, list[str]]:
    """Create a one-batch map-style loader over deterministic validation keys."""
    if num_frames <= 0:
        raise ValueError("Heavy validation requires at least one frame.")

    scene_dataset = dataset.web_dataloader.web_scene_dataset
    frame_index = scene_dataset.frame_index.reset_index(drop=True)
    if len(frame_index) == 0:
        raise RuntimeError("The heavy-validation source split is empty.")

    if selection_path.is_file():
        selection = json.loads(selection_path.read_text())
        selected_keys = [str(key) for key in selection["keys"]]
        key_to_index = {
            str(row.key): int(index) for index, row in frame_index.iterrows()
        }
        missing = [key for key in selected_keys if key not in key_to_index]
        if missing:
            raise KeyError(
                f"{len(missing)} saved heavy-validation keys are absent from "
                f"the current split: {missing[:5]}"
            )
        selected_indices = [key_to_index[key] for key in selected_keys]
    else:
        count = min(num_frames, len(frame_index))
        rng = np.random.default_rng(seed)
        selected_indices = sorted(
            int(index)
            for index in rng.choice(len(frame_index), size=count, replace=False)
        )
        selected_keys = [
            str(frame_index.iloc[index].key) for index in selected_indices
        ]
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            selection_path.parent.mkdir(parents=True, exist_ok=True)
            selection_path.write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "requested_frames": num_frames,
                        "keys": selected_keys,
                    },
                    indent=2,
                )
            )

    # Heavy validation must retain every annotated car in the selected frames.
    dataset.batch_size = 1_000_000
    subset = Subset(scene_dataset, selected_indices)
    loader = DataLoader(
        subset,
        batch_size=len(selected_indices),
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    return loader, selected_keys


def _distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _broadcast_failure(failed: torch.Tensor) -> torch.Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast(failed, src=0)
    return failed


def _translation_scale(mesh: trimesh.Trimesh) -> float:
    extent = float(np.linalg.norm(mesh.extents))
    return 0.001 if 0.01 < extent < 100.0 else 1.0


class HeavyValidationCallback(pl.Callback):
    """Run a sparse, rank-zero-only, full pose/CAD visual audit."""

    def __init__(
        self,
        loader: DataLoader,
        dataset_name: str,
        interval: int,
        output_dir: Path,
        mesh_path: Path,
    ) -> None:
        super().__init__()
        if interval <= 0:
            raise ValueError("Heavy-validation interval must be positive.")
        self.loader = loader
        self.dataset_name = dataset_name
        self.interval = interval
        self.output_dir = Path(output_dir)
        self.mesh_path = Path(mesh_path)
        self.last_step = -1

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module) -> None:
        step = int(trainer.global_step)
        should_run = step > 0 and step % self.interval == 0
        if not should_run or self.last_step == step:
            return
        self.last_step = step

        _distributed_barrier()
        failed = torch.zeros(1, dtype=torch.int32, device=pl_module.device)
        if trainer.is_global_zero:
            try:
                self._run_rank_zero(pl_module, step)
            except Exception:
                failed.fill_(1)
                logger.exception("Heavy validation failed at step %d", step)

        failed = _broadcast_failure(failed)
        _distributed_barrier()
        if int(failed.item()) != 0:
            raise RuntimeError(
                f"Heavy validation failed at step {step}; see the rank-zero "
                "traceback above."
            )

    @torch.inference_mode()
    def _run_rank_zero(self, model, step: int) -> None:
        batch = next(iter(self.loader))
        if batch is None or len(batch) == 0:
            raise RuntimeError("The fixed heavy-validation batch is empty.")

        full_rgb = batch.tar_full_img.cpu()
        image_sizes = batch.tar_image_size.cpu()
        image_offsets = batch.tar_image_offset.cpu()
        for name in ("tar_full_img", "tar_image_size", "tar_image_offset"):
            batch.delete_tensor(name)
        batch.to(model.device)

        # IST template features must reflect the current trainable IST state.
        model.template_datas.pop(self.dataset_name, None)
        model.pose_recovery.pop(self.dataset_name, None)
        was_training = model.training
        model.eval()
        try:
            predictions = model.eval_retrieval(
                batch,
                idx_batch=0,
                dataset_name=self.dataset_name,
                save_outputs=False,
                log_retrieval=False,
            )
        finally:
            model.train(was_training)

        mesh = trimesh.load(self.mesh_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
            raise ValueError(f"Could not load heavy-validation CAD: {self.mesh_path}")
        scale_translation = _translation_scale(mesh)

        frame_to_indices: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, row in batch.infos.reset_index(drop=True).iterrows():
            frame_key = (
                int(row["scene_id"]),
                int(row.get("view_id", row.get("im_id", -1))),
            )
            frame_to_indices[frame_key].append(index)

        renderer = InstanceRenderer(mesh)
        panels: list[Image.Image] = []
        report: list[dict[str, Any]] = []
        try:
            for frame_key, indices in frame_to_indices.items():
                source_index = indices[0]
                padded_rgb = (
                    full_rgb[source_index]
                    .clamp(0, 1)
                    .permute(1, 2, 0)
                    .numpy()
                )
                padded_image = Image.fromarray(np.uint8(padded_rgb * 255))
                poses = []
                scores = []
                for index in indices:
                    pose = (
                        predictions.pred_poses[index, 0]
                        .detach()
                        .cpu()
                        .numpy()
                        .copy()
                    )
                    pose[:3, 3] *= scale_translation
                    poses.append(pose)
                    scores.append(
                        float(predictions.scores[index, 0].detach().cpu())
                    )

                K = batch.tar_K[source_index].detach().cpu().numpy()
                segmentation, _ = renderer.render(
                    poses, K, padded_image.width, padded_image.height
                )

                original_height, original_width = (
                    image_sizes[source_index].numpy().astype(int)
                )
                top, left = image_offsets[source_index].numpy().astype(int)
                crop_box = (
                    left,
                    top,
                    left + original_width,
                    top + original_height,
                )
                original = padded_image.crop(crop_box)
                segmentation = segmentation[
                    top : top + original_height,
                    left : left + original_width,
                ]

                overlay_array = np.asarray(original, dtype=np.float32).copy()
                for render_id in range(1, len(indices) + 1):
                    mask = segmentation == render_id
                    color = np.asarray(
                        COLORS[(render_id - 1) % len(COLORS)], dtype=np.float32
                    )
                    overlay_array[mask] = (
                        0.58 * overlay_array[mask] + 0.42 * color
                    )
                overlay = Image.fromarray(
                    np.uint8(np.clip(overlay_array, 0, 255))
                )
                draw = ImageDraw.Draw(overlay)
                for render_id, score in enumerate(scores, start=1):
                    x, y, width, height = bbox_from_mask(
                        segmentation == render_id
                    )
                    if width <= 0 or height <= 0:
                        continue
                    color = COLORS[(render_id - 1) % len(COLORS)]
                    draw.rectangle(
                        (x, y, x + width - 1, y + height - 1),
                        outline=color,
                        width=3,
                    )
                    draw.text(
                        (x + 3, max(0, y - 15)),
                        f"pred {render_id}: {score:.3f}",
                        fill=color,
                        stroke_width=2,
                        stroke_fill=(0, 0, 0),
                    )

                panel = Image.new(
                    "RGB", (original.width * 2, original.height), (0, 0, 0)
                )
                panel.paste(original, (0, 0))
                panel.paste(overlay, (original.width, 0))
                panels.append(panel)
                report.append(
                    {
                        "scene_id": frame_key[0],
                        "view_id": frame_key[1],
                        "instances": len(indices),
                        "scores": scores,
                    }
                )
        finally:
            renderer.close()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        max_width = max(panel.width for panel in panels)
        total_height = sum(panel.height for panel in panels)
        contact_sheet = Image.new("RGB", (max_width, total_height), (0, 0, 0))
        y = 0
        for panel in panels:
            contact_sheet.paste(panel, (0, y))
            y += panel.height

        image_path = self.output_dir / f"heavy_cad_step{step:06d}.jpg"
        report_path = self.output_dir / f"heavy_cad_step{step:06d}.json"
        contact_sheet.save(image_path, quality=94)
        report_path.write_text(json.dumps(report, indent=2))
        log_image(
            logger=model.logger,
            name="vis/val_heavy_cad_overlay",
            path=str(image_path),
            step=step,
        )
        logger.info("Heavy validation step %d: wrote %s", step, image_path)
