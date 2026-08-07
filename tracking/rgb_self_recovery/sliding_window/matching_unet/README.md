# Lightweight matching U-Net

This package is an isolated fourth RGB self-recovery backbone. It does not
modify the CNN, DINO, dataset generator, five-frame optimizer, or common runner.
It consumes the exact same generated `train` and `validation` shards, so no
dataset regeneration or split change is needed.

## Architecture

The model has separate four-level encoders for RGB plus observed mask and the
five-channel CAD render. At every level it fuses image features, CAD features,
their absolute difference, and their product. A three-level U-Net decoder
preserves spatial disagreement and predicts a 64-by-64 center heatmap for the
default 128-by-128 crop. Differentiable soft-argmax converts that heatmap into
the two-dimensional center correction.

The globally pooled deepest and decoded features predict log-depth correction,
SO(3) rotation correction, confidence, and quality. The default model has about
2.98 million trainable parameters, versus about 3.64 million in the current CNN.
The trainer retains all existing regression, ranking, and anti-flip losses and
adds a Gaussian heatmap KL loss.

## Train on the identical split

```bash
export DATA="$PWD/gigaPose_datasets/results/rgb_self_recovery_assetto_window_data"
export MODELS="$PWD/gigaPose_datasets/results/rgb_self_recovery_backbone_comparison"

python3 -m tracking.rgb_self_recovery.sliding_window.matching_unet.train \
  --data "$DATA/train" \
  --validation-data "$DATA/validation" \
  --output-dir "$MODELS/matching_unet" \
  --epochs 200 \
  --batch-size 8 \
  --anti-flip-training \
  --heatmap-weight 0.10 \
  --heatmap-sigma-px 4 \
  --patience 10 \
  --min-delta 1e-4 \
  --logger wandb \
  --wandb-project rgb-self-recovery-backbones \
  --run-name matching-unet \
  --device cuda
```

Start with batch size 8 because the decoder retains spatial feature maps. Raise
it only after checking GPU memory. As with the other isolated trainers, use
`best_pose.ckpt` first; `best_loss.ckpt`, `best_rotation.ckpt`,
`best_center.ckpt`, `best_depth.ckpt`, `best.ckpt`, and `last.ckpt` are also
written.

## Run five-frame inference

Use the same predictions, prepared held-out sequence, mesh, mask settings, and
window arguments used for CNN and DINO. Only the module, checkpoint, and output
directory should differ:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.matching_unet.run \
  --predictions "$GIGAPOSE_PREDICTIONS" \
  --dataset-dir "$EVAL_DATA" \
  --split test \
  --checkpoint "$MODELS/matching_unet/best_pose.ckpt" \
  --mesh "$EVAL_DATA/models/obj_000001.ply" \
  --output-dir "$EVAL_RUNS/matching_unet" \
  --window-size 5 \
  --save-overlays \
  --device cuda
```

Pass the same additional flags used by the other inference runs, including
`--prediction-translation-unit`, mask options, mesh alignment options, and
association configuration. The output remains `tracked_predictions.csv`, so it
can be added to the existing evaluator:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.evaluate_backbones \
  --dataset-dir "$EVAL_DATA" \
  --split test \
  --model cnn="$EVAL_RUNS/cnn" \
  --model dino_frozen="$EVAL_RUNS/dino_frozen" \
  --model dino_last_block="$EVAL_RUNS/dino_last_block" \
  --model matching_unet="$EVAL_RUNS/matching_unet" \
  --baseline cnn \
  --mesh "$EVAL_DATA/models/obj_000001.ply" \
  --distance-bins-m 0,20,40,60,80,100,120 \
  --max-overlays 120 \
  --output-dir "$EVAL_RUNS/backbone_comparison_with_unet" \
  --overwrite
```

Select the final model using held-out GigaPose translation/rotation errors and
overlays. Lower training or heatmap loss alone is not sufficient evidence.
