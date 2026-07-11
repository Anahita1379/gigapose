# OT/soft-correspondence AE fine-tuning

This is an isolated experiment for training DINO/AE features with masked
optimal transport. It does not modify the existing GigaPose training,
pose-aware IST training, or inference code.

## What is trained?

The trainable part is the AE/DINO feature extractor. IST is loaded from the
checkpoint but frozen and unused by the OT objective.

The intended first setting is:

```bash
--ae-train-mode last-blocks \
--ae-train-last-n-blocks 1
```

You can also train selected DINO blocks by offset from the end:

```bash
--ae-train-mode block-offsets \
--ae-train-block-offsets 2
```

Here `2` means the penultimate DINO block. `2,3` means the two blocks before
the final block.

The OT entry point intentionally does not expose `--ae-train-mode norm`.
AENet consumes DINO's `x_prenorm` patch tokens, so the final DINO norm is not
on the feature path used by this objective. Transformer block modes do have a
valid gradient path.

## Objective

For every source/template crop and target/real crop, AE produces normalized
patch features:

```math
F^{src}\in\mathbb{R}^{S\times C},
\qquad
F^{tar}\in\mathbb{R}^{T\times C}.
```

The feature similarity logits are:

```math
Z_{ij}
=
\frac{
\langle f_i^{src}, f_j^{tar}\rangle
}{\tau}.
```

Invalid source/target patches from the masks are removed. A masked balanced
Sinkhorn layer produces the soft transport matrix:

```math
P
=
\operatorname{Sinkhorn}(Z).
```

The total optimized loss is:

```math
L_{\rm OT}
=
\lambda_c L_{\rm corr}
+
\lambda_p L_{\rm soft\text{-}patch}
+
\lambda_a L_{\rm soft\text{-}affine}
+
\lambda_e L_{\rm entropy}.
```

### Ground-truth correspondence loss

For each known patch correspondence \((i,j^*)\), we normalize each source row:

```math
Q_{ij}
=
\frac{P_{ij}}{\sum_k P_{ik}+\epsilon}.
```

Then, the correspondence loss is averaged within each car before averaging
cars. This prevents a large, unobstructed car with many valid GT patch pairs
from dominating a smaller or partially visible car:

```math
L_{\rm corr}
=
-\frac{1}{B}\sum_b\frac{1}{N_b}
\sum_{(i,j^*)\in\mathcal P_b}\log(Q_{i j^*}+\epsilon).
```

### Soft patch reprojection loss

The expected target patch coordinate for each source patch is:

```math
\hat x_i^{tar}
=
\sum_j Q_{ij}x_j^{tar}.
```

For ground-truth correspondences:

```math
L_{\rm soft\text{-}patch}
=
\frac{1}{B}\sum_b\frac{1}{N_b}
\sum_{(i,j^*)\in\mathcal P_b}
\rho_\beta
\left(
\frac{
\|\hat x_i^{tar}-x_{j^*}^{tar}\|_2
}{
224\sqrt{2}
}
\right).
```

Patch coordinates use DINO token centers,
`(patch_index + 0.5) * patch_size`, in both geometric objectives.

### Soft global affine center loss

The code also fits a differentiable weighted affine map from all valid source
patch coordinates to their OT-expected target coordinates. It then maps the
source object center to the target crop and compares it with the ground-truth
target object center:

```math
L_{\rm soft\text{-}affine}
=
\rho_\beta
\left(
\frac{
\|A(\hat o^{src})-o^{tar}\|_2
}{
224\sqrt{2}
}
\right).
```

This is the "soft global affine" branch. It is not RANSAC; it is differentiable
and meant to guide the feature matcher during training.

### Entropy

`L_entropy` is the transport entropy:

```math
L_{\rm entropy}
=
-
\sum_{ij}
P_{ij}\log(P_{ij}+\epsilon).
```

Positive `--entropy-weight` encourages sharper transport. Keep it at `0.0`
initially.

## Example run

```bash
python -m fine_tuning.ot_training.train \
  --dataset-name assettocorsa_new_dataset \
  --train-split train_pbr_web_gsam_clean \
  --validation-split val_pbr_web_gsam_clean \
  --checkpoint gigaPose_datasets/pretrained/gigaPose_v1.ckpt \
  --ae-lr 1e-6 \
  --ae-train-mode last-blocks \
  --ae-train-last-n-blocks 1 \
  --batch-size 16 \
  --num-workers 4 \
  --max-steps 25000 \
  --validation-interval 250 \
  --checkpoint-interval 1000 \
  --feature-temperature 0.07 \
  --sinkhorn-iterations 30 \
  --correspondence-weight 1.0 \
  --soft-patch-reprojection-weight 0.25 \
  --soft-affine-center-weight 0.25 \
  --entropy-weight 0.0 \
  --hard-match-confidence 0.05 \
  --best-ot-checkpoints 3 \
  --run-name assettocorsa_ot_ae_lastblock \
  --logger wandb \
  --print-loss-every 50 \
  --devices 0
```

Penultimate block variant:

```bash
--ae-train-mode block-offsets \
--ae-train-block-offsets 2
```

## What to watch

Training logs:

- `train/loss_ot_total`
- `train/loss_ot_correspondence`
- `train/loss_ot_soft_patch_reprojection`
- `train/loss_ot_soft_affine_center`
- `train/monitor_ot_gt_probability`
- `train/monitor_ot_gt_top1_accuracy`
- `train/monitor_ot_gt_confident_mutual_accuracy`
- `train/monitor_ot_mean_confident_match_score`
- `train/monitor_ot_mutual_matches_per_instance`
- `train/monitor_ot_confident_mutual_matches_per_instance`
- `train/monitor_ot_valid_gt_pairs`

Validation logs use the same names with `val/`.

Validation images are written under:

```text
gigaPose_datasets/results/<run-name>/validation_images/
```

You should see:

- `val_ot_gt_*.png`: ground-truth patch correspondences;
- `val_ot_pred_*.png`: hard correspondences induced by the OT matrix.

Predicted images include only mutual nearest matches whose row-conditional OT
probability is at least `--hard-match-confidence`. This removes the previous
clutter from drawing one forced match for every source patch. It does not prune
the soft transport used for training.

Periodic checkpoints and `last.ckpt` are written normally. In addition, the
best `--best-ot-checkpoints` files are written as `best-ot-step*.ckpt`, ranked
by `val/monitor_ot_gt_top1_accuracy`.

## Current scope

This first version trains AE/DINO features and visualizes OT correspondences.
It does not yet replace GigaPose inference or RANSAC. Once the OT matches look
good, the next safe step is to export the hard or top-k OT correspondences into
the existing pose-recovery path.
