# DINO patch matching U-Net (model 5)

This isolated model combines the three useful evidence sources without editing
the existing CNN, pooled-DINO, matching-U-Net, or five-frame pipeline:

```text
RGB + observed mask -> four-level CNN pyramid -----------+
Frozen DINOv2 RGB -> 16x16 spatial patch features -------+-> matching U-Net
Five-channel candidate CAD render -> geometry pyramid ---+       |
                                                               center heatmap
                                                               pose/ranking head
                                                                      |
                                                        five-frame optimization
```

For the default DINOv2-S/14 input of 224-by-224, 14-by-14 patches produce a
16-by-16 token map. It is projected and fused at the U-Net's 16-by-16 decoder
stage instead of being reduced immediately to one global vector. A pooled DINO
feature additionally informs depth, rotation, confidence, and quality.

With the default settings the model contains about 25.25 million parameters,
of which about 3.19 million are trainable in frozen mode; all DINO backbone
parameters remain frozen.

The RGB/mask CNN and frozen DINO encoding are computed once per image and reused
for every CAD candidate and iterative recovery pass. The full DINO state is
saved in each checkpoint, so inference does not download pretrained weights.

## Train frozen DINO patch fusion on the identical split

No dataset regeneration is required:

```bash
export DATA="$PWD/gigaPose_datasets/results/rgb_self_recovery_assetto_window_data"
export MODELS="$PWD/gigaPose_datasets/results/rgb_self_recovery_backbone_comparison"

python3 -m tracking.rgb_self_recovery.sliding_window.dino_matching_unet.train \
  --dino-mode frozen \
  --dino-model dinov2_vits14 \
  --dino-input-size 224 \
  --data "$DATA/train" \
  --validation-data "$DATA/validation" \
  --output-dir "$MODELS/dino_matching_unet" \
  --epochs 200 \
  --batch-size 4 \
  --anti-flip-training \
  --heatmap-weight 0.10 \
  --heatmap-sigma-px 4 \
  --patience 10 \
  --min-delta 1e-4 \
  --logger wandb \
  --wandb-project rgb-self-recovery-backbones \
  --run-name dino-matching-unet \
  --device cuda
```

Start with batch size 4 because this model retains both DINO patch tokens and
U-Net feature maps. DINO is frozen, but the RGB/mask encoder, CAD encoder,
multi-scale matching blocks, DINO projections, decoder, and output heads train.

The trainer writes `best_pose.ckpt`, `best_loss.ckpt`, `best_rotation.ckpt`,
`best_center.ckpt`, `best_depth.ckpt`, the compatible `best.ckpt` loss alias,
and `last.ckpt`. Begin inference with `best_pose.ckpt`.

## Optional later experiment: tune DINO's final block

The code also supports `--dino-mode last_block`, but first complete the frozen
run for a clean comparison. Last-block mode trains only DINO's final transformer
block and norm at `--dino-learning-rate` while training the rest at
`--learning-rate`. It currently starts a separate training run; it does not
implicitly load the frozen checkpoint.

## Run five-frame inference

Use exactly the same held-out sequence and settings as every other backbone:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.dino_matching_unet.run \
  --predictions "$GIGAPOSE_PREDICTIONS" \
  --dataset-dir "$EVAL_DATA" \
  --split test \
  --checkpoint "$MODELS/dino_matching_unet/best_pose.ckpt" \
  --mesh "$EVAL_DATA/models/obj_000001.ply" \
  --output-dir "$EVAL_RUNS/dino_matching_unet" \
  --window-size 5 \
  --save-overlays \
  --device cuda
```

Copy all additional flags from the CNN/DINO/UNet inference command, especially
translation units, masks, CAD alignment, and association settings.

## Compare all five models

After producing `tracked_predictions.csv` for each model, add this run to the
existing evaluator:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.evaluate_backbones \
  --dataset-dir "$EVAL_DATA" \
  --split test \
  --model cnn="$EVAL_RUNS/cnn" \
  --model dino_frozen="$EVAL_RUNS/dino_frozen" \
  --model dino_last_block="$EVAL_RUNS/dino_last_block" \
  --model matching_unet="$EVAL_RUNS/matching_unet" \
  --model dino_matching_unet="$EVAL_RUNS/dino_matching_unet" \
  --baseline cnn \
  --mesh "$EVAL_DATA/models/obj_000001.ply" \
  --distance-bins-m 0,20,40,60,80,100,120 \
  --max-overlays 120 \
  --output-dir "$EVAL_RUNS/all_five_backbones" \
  --overwrite
```

Choose the model from held-out GigaPose translation/rotation metrics and
overlays, not from training loss alone.
