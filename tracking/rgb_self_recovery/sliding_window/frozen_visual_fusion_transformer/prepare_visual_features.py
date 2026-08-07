"""Cache frozen frame and candidate RGB/mask/CAD embeddings in a candidate bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from tracking.io import WebDatasetSequence
from tracking.rendering import CADRenderer
from tracking.rgb_self_recovery.render_inputs import (
    decode_render_channels,
    render_candidate_channels,
)
from tracking.rgb_self_recovery.sliding_window.dino_matching_unet.inference import (
    load_predictor,
)
from tracking.rgb_self_recovery.sliding_window.gru_selector.dataset import (
    CandidateBundle,
    resolve_bundle,
)


FORMAT = "rgb_self_recovery_frozen_visual_features_v1"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    value.add_argument("--data", type=Path, required=True, help="Candidate bundle directory")
    value.add_argument("--dataset-dir", type=Path, help="Defaults to bundle manifest dataset_dir")
    value.add_argument("--split", help="Defaults to bundle manifest split")
    value.add_argument("--checkpoint", type=Path, help="Frozen DINO matching U-Net checkpoint")
    value.add_argument("--mesh", type=Path)
    value.add_argument("--mesh-scale", type=float, default=1.0)
    value.add_argument("--center-mesh", action="store_true")
    value.add_argument("--device", default="cuda")
    value.add_argument("--output", type=Path)
    value.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    value.add_argument("--overwrite", action="store_true")
    return value


def _resolve(args, bundle: CandidateBundle) -> tuple[Path, str, Path, Path]:
    manifest = bundle.manifest
    dataset = args.dataset_dir or Path(manifest["dataset_dir"])
    split = args.split or str(manifest.get("split", "test"))
    checkpoint = args.checkpoint or Path(manifest["checkpoint"])
    mesh = args.mesh or dataset / "models" / "obj_000001.ply"
    for path in (dataset, checkpoint, mesh):
        if not path.exists():
            raise FileNotFoundError(path)
    return dataset, split, checkpoint, mesh


def main() -> None:
    args = parser().parse_args()
    bundle = CandidateBundle(args.data)
    data_path, _ = resolve_bundle(args.data)
    output = args.output or data_path.with_name("visual_features.npz")
    visual_manifest = output.with_name("visual_manifest.json")
    if (output.exists() or visual_manifest.exists()) and not args.overwrite:
        raise FileExistsError(f"Visual sidecar exists: {output}; pass --overwrite")
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU", flush=True)
        device = "cpu"
    dataset_dir, split, checkpoint, mesh = _resolve(args, bundle)
    predictor = load_predictor(checkpoint, device)
    model = predictor.model.eval()
    if not hasattr(model, "extract_visual_embeddings_encoded"):
        raise TypeError("Checkpoint model does not expose DINO matching embeddings")
    renderer = CADRenderer(mesh, mesh_scale=args.mesh_scale, center_mesh=args.center_mesh)

    arrays = bundle.arrays
    keys = {
        (int(scene), int(image)): index
        for index, (scene, image) in enumerate(zip(arrays["scene_id"], arrays["im_id"]))
    }
    if len(keys) != len(bundle):
        raise ValueError("Candidate bundle contains duplicate scene_id/im_id rows")
    frame_values: list[np.ndarray | None] = [None] * len(bundle)
    candidate_values: list[np.ndarray | None] = [None] * len(bundle)
    started = time.perf_counter()
    source = WebDatasetSequence(dataset_dir, split, load_depth=False)
    written = 0
    try:
        for frame in source:
            index = keys.get((int(frame.scene_id), int(frame.im_id)))
            if index is None:
                continue
            if len(frame.detections) != 1:
                raise ValueError(
                    f"Expected one detection for {frame.scene_id}/{frame.im_id}"
                )
            rgb, observed, crop, _ = predictor.observation(frame, frame.detections[0])
            valid_indices = np.flatnonzero(arrays["valid"][index])
            K_crop = crop.transform_intrinsics(frame.K)
            encoded = np.stack([
                render_candidate_channels(
                    renderer,
                    arrays["poses"][index, candidate_index],
                    K_crop,
                    crop.output_size,
                )[0]
                for candidate_index in valid_indices
            ])
            rendered = torch.from_numpy(decode_render_channels(encoded)).to(device)
            with torch.inference_mode():
                image_encoding = model.encode_image(rgb, observed)
                features = model.extract_visual_embeddings_encoded(
                    image_encoding, rendered
                )
            frame_embedding = features["frame_visual_embedding"][0].float().cpu().numpy()
            valid_embedding = features["candidate_visual_embedding"].float().cpu().numpy()
            candidate_embedding = np.zeros(
                (bundle.candidate_count, valid_embedding.shape[1]), dtype=np.float32
            )
            candidate_embedding[valid_indices] = valid_embedding
            frame_values[index] = frame_embedding
            candidate_values[index] = candidate_embedding
            written += 1
            if written % 25 == 0:
                print(f"cached {written}/{len(bundle)} frames", flush=True)
    finally:
        renderer.close()
    missing = [index for index, value in enumerate(frame_values) if value is None]
    if missing:
        examples = [
            [int(arrays["scene_id"][i]), int(arrays["im_id"][i])] for i in missing[:10]
        ]
        raise RuntimeError(f"Could not find {len(missing)} candidate frames; examples={examples}")

    dtype = np.float16 if args.storage_dtype == "float16" else np.float32
    frame_features = np.asarray(frame_values, dtype=dtype)
    candidate_features = np.asarray(candidate_values, dtype=dtype)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        scene_id=arrays["scene_id"],
        im_id=arrays["im_id"],
        frame_visual_features=frame_features,
        candidate_visual_features=candidate_features,
    )
    report = {
        "format": FORMAT,
        "data": str(data_path),
        "dataset_dir": str(dataset_dir),
        "split": split,
        "checkpoint": str(checkpoint),
        "checkpoint_frozen": True,
        "mesh": str(mesh),
        "mesh_scale": args.mesh_scale,
        "center_mesh": args.center_mesh,
        "frame_count": len(bundle),
        "candidate_count": bundle.candidate_count,
        "frame_visual_dim": int(frame_features.shape[-1]),
        "candidate_visual_dim": int(candidate_features.shape[-1]),
        "storage_dtype": args.storage_dtype,
        "elapsed_s": time.perf_counter() - started,
    }
    visual_manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
