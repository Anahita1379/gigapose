# Assetto development evaluation

This folder evaluates raw GigaPose and all six sliding-window RGB
self-recovery backbones without changing GigaPose's dataset or inference code.

The prepared runs are deliberately separate:

- `rgb_recovery_val_putnam_rear`: 164 ordered rear-camera frames from
  `20260730_putnam_rain_1opp_2lap_farRear`.
- `rgb_recovery_val_laguna_front`: 933 ordered front-camera frames from
  `20260730_laguna2026_fog_1opp_2laps_mostlyFront`.

These runs were already used for validation/checkpoint selection. They are
useful development diagnostics, but they are not an unbiased final benchmark.
Do not merge their timelines: they use different tracks and cameras.

## 1. Prepare and validate

The folders are already generated under `gigaPose_datasets/datasets`. Rebuild
them only if the raw export changes:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.assetto_validation_eval.prepare \
  --overwrite

python3 -m tracking.rgb_self_recovery.sliding_window.assetto_validation_eval.validate \
  --dataset-dir gigaPose_datasets/datasets/rgb_recovery_val_putnam_rear

python3 -m tracking.rgb_self_recovery.sliding_window.assetto_validation_eval.validate \
  --dataset-dir gigaPose_datasets/datasets/rgb_recovery_val_laguna_front
```

Preparation creates the WebDataset shards, ground-truth poses, masks, test
targets, aligned CAD models, per-dataset CNOS detections, frame timestamps, and
template links. The images are not shade-cropped; this matches the training
dataset's default preparation setting.

## 2. Generate GigaPose predictions

The launcher defaults to the Assetto residual checkpoint previously used in
this repository:

`gigaPose_datasets/results/assettocorsa_ot2block_pose_aware_ist_translation_residual/checkpoints/best-residual-step010000.ckpt`

Run both datasets sequentially on GPU 0:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.assetto_validation_eval.run_gigapose \
  --devices 0 \
  --batch-size 16 \
  --num-workers 2
```

Here `--devices 0` is the CUDA device index. To expose one physical GPU and
always address it as device zero, prefix the command with
`CUDA_VISIBLE_DEVICES=GPU_INDEX`. To test a different GigaPose checkpoint, pass
`--checkpoint PATH`.

The command writes:

- `gigaPose_datasets/results/gigapose_dev_rgb_recovery_val_putnam_rear/`
- `gigaPose_datasets/results/gigapose_dev_rgb_recovery_val_laguna_front/`
- `gigaPose_datasets/results/gigapose_dev_manifest.json`

The following commands accept either a result directory or its exact
`MultiHypothesis.csv`, so the long generated CSV filename is not needed.

## 3. Run the six recovery models

```bash
MODELS=gigaPose_datasets/results/rgb_self_recovery_backbone_comparison
DEV=gigaPose_datasets/results/rgb_self_recovery_assetto_dev_eval

python3 -m tracking.rgb_self_recovery.sliding_window.assetto_validation_eval.run_suite \
  --dataset-dir gigaPose_datasets/datasets/rgb_recovery_val_putnam_rear \
  --predictions gigaPose_datasets/results/gigapose_dev_rgb_recovery_val_putnam_rear \
  --models-root "$MODELS" \
  --output-root "$DEV/putnam_rear/runs" \
  --device cuda \
  --save-overlays

python3 -m tracking.rgb_self_recovery.sliding_window.assetto_validation_eval.run_suite \
  --dataset-dir gigaPose_datasets/datasets/rgb_recovery_val_laguna_front \
  --predictions gigaPose_datasets/results/gigapose_dev_rgb_recovery_val_laguna_front \
  --models-root "$MODELS" \
  --output-root "$DEV/laguna_front/runs" \
  --device cuda \
  --save-overlays
```

The six model names are `cnn_multicheckpoint`, `dino_frozen_multicheckpoint`, `dino_last_block`,
`matching_unet`, `dino_matching_unet`, and
`dino_matching_unet_last_block`. Add
`--models cnn_multicheckpoint,dino_frozen_multicheckpoint` to run only a subset.
Add `--skip-missing` if some checkpoints have not been trained yet. For a quick
integration test, add `--max-frames 10` and use a different output root.

The evaluation launcher uses **soft orientation reranking** for every model by
default. It penalizes abrupt disagreement with the previous accepted pose and
rank-0 GigaPose pose, but never removes a candidate. Use
`--orientation-gate-mode hard` for the old rejection behavior, or
`--orientation-gate-mode off` to measure learned ranking without orientation
guidance. The launcher also defaults to `--state-iou-confidence-weight 0.5`:
silhouette IoU can keep a visually aligned, low-verifier result uncertain
instead of lost, while `verifier_confidence` remains available unchanged in
`candidate_diagnostics.csv`. Continuous window smoothing is re-rendered and
re-scored; a smoothed endpoint is rejected when visual cost rises by more than
`0.02` or silhouette IoU drops by more than `0.02`.

Recovery abstention is also enabled by default. The runner independently
scores the unmodified rank-0 GigaPose pose. If the learned recovery has verifier
confidence below `0.05`, makes a jump larger than `1 m` or `45 deg`, and has no
support from an established trajectory, the output retains rank-0 GigaPose and
is explicitly marked `lost`. This is a degradation guard, not a claim that the
fallback pose is correct. Use `--runner-arg=--no-recovery-abstention` only for
an intentional ablation.

Sequence handling is enabled by default. Frames are grouped by
`source_run + camera_id`, ordered by `source_frame`, and temporal state resets
when source frames are nonconsecutive or timestamps differ by more than 0.5
seconds. Override those limits with `--sequence-max-frame-gap` and
`--sequence-max-time-gap-s`.

For controlled temporal ablations, use three separate output roots:

- `--window-size 5`: full sequence-aware five-frame optimization;
- `--window-size 1`: original sequential tracker without the added window;
- `--window-size 1 --per-frame`: independent same-frame recovery with no
  previous track, optical flow, candidate beam, or window history.

## 4. Evaluate and compare

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.assetto_validation_eval.evaluate_suite \
  --dataset-dir gigaPose_datasets/datasets/rgb_recovery_val_putnam_rear \
  --gigapose-predictions gigaPose_datasets/results/gigapose_dev_rgb_recovery_val_putnam_rear \
  --runs-root "$DEV/putnam_rear/runs" \
  --output-dir "$DEV/putnam_rear/comparison"

python3 -m tracking.rgb_self_recovery.sliding_window.assetto_validation_eval.evaluate_suite \
  --dataset-dir gigaPose_datasets/datasets/rgb_recovery_val_laguna_front \
  --gigapose-predictions gigaPose_datasets/results/gigapose_dev_rgb_recovery_val_laguna_front \
  --runs-root "$DEV/laguna_front/runs" \
  --output-dir "$DEV/laguna_front/comparison"
```

Each comparison includes raw GigaPose as the baseline and reports paired
translation/rotation improvement for every recovery model. It writes
`summary.json`, per-frame metrics, distance-binned translation and rotation
plots, and CAD silhouette overlays. Positive paired improvement means the
recovery model reduced error relative to raw GigaPose on the same frame.

Use the Putnam and Laguna summaries side by side. A model should be considered
better only if it improves errors on both camera/track conditions without a
large loss in prediction coverage.
