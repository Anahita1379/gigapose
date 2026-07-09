# Pose-aware GigaPose fine-tuning

This is an isolated experimental training path. It does not modify
`fine_tuning.train`, `fine_tuning.train_val`, `src.models.gigaPose`, or the
existing Assetto Corsa loader.

## Objectives

For IST-only training, the default optimized objective is:

```math
L_{\rm IST,base} =
\lambda_z L_{\log\text{-depth}} +
\lambda_\theta L_{\text{inplane}} +
\lambda_p L_{\text{reprojection}}.
```

- `log-depth`: Smooth-L1 of
  `log(predicted_relative_scale) - log(gt_relative_scale)`. Since target depth
  is inversely proportional to relative scale, this is the relative log-depth
  error up to sign.
- `inplane`: `1 - cos(predicted_angle - gt_angle)`. This is bounded and avoids
  the unstable derivative of `acos` close to a perfect prediction.
- `reprojection`: each valid patch prediction defines a differentiable
  scale/rotation/translation mapping. The mapped template object center is
  compared with the ground-truth target object center, normalized by the
  224x224 crop diagonal, using Smooth-L1.

More explicitly, for each valid patch correspondence \(i\), IST predicts a
relative scale \(\hat s_i\) and an in-plane rotation represented as a normalized
2-vector \(\hat u_i=[\hat c_i,\hat s_i]\). The ground truth values are
\(s_i^*\) and \(\theta_i^*\). With a Smooth-L1/Huber residual
\(\rho_\beta(\cdot)\):

```math
L_{\log\text{-depth}}
=
\frac{1}{N}\sum_i
\rho_{\beta_z}
\left(
\log(\hat s_i+\epsilon)-\log(s_i^*+\epsilon)
\right).
```

Target depth is inversely proportional to relative scale, so this is a relative
log-depth loss up to sign.

```math
L_{\text{inplane}}
=
\frac{1}{N}\sum_i
\left[
1 -
\hat u_i^\top
\begin{bmatrix}
\cos \theta_i^*\\
\sin \theta_i^*
\end{bmatrix}
\right].
```

For reprojection, each patch pair gives a 2D similarity transform from template
crop coordinates to target crop coordinates. If \(p_i^{src}\) and
\(p_i^{tar}\) are valid source and target patch centers, then:

```math
\hat A_i(x)
=
\hat s_i R(\hat u_i)x
+
\left(
p_i^{tar} - \hat s_i R(\hat u_i)p_i^{src}
\right).
```

Let \(o^{src}\) and \(o^{tar}\) be the object-center projections in the source
and target 224x224 crop coordinates. The optimized center reprojection term is:

```math
L_{\text{reprojection}}
=
\frac{1}{N}\sum_i
\rho_{\beta_p}
\left(
\frac{
\left\|
\hat A_i(o^{src}) - o^{tar}
\right\|_2
}{
224\sqrt{2}
}
\right).
```

## Optional direct pose losses

By default, the approximate translation/depth/full-rotation errors described
below are monitoring-only. If you pass:

```bash
--optimize-pose-monitor-errors
```

then the training objective becomes:

```math
L_{\rm IST}
=
L_{\rm IST,base}
+
\lambda_t L_{\rm direct\text{-}translation}
+
\lambda_d L_{\rm direct\text{-}depth}
+
\lambda_R L_{\rm direct\text{-}rotation}
+
\lambda_{p2} L_{\rm direct\text{-}reprojection}.
```

Patch predictions are first aggregated per instance:

```math
\bar{\ell}
=
\frac{1}{|\mathcal P|}
\sum_{i\in\mathcal P}\log \hat s_i,\qquad
\bar{s}=\exp(\bar{\ell}),
```

```math
\bar u
=
\operatorname{normalize}
\left(
\frac{1}{|\mathcal P|}
\sum_{i\in\mathcal P}\hat u_i
\right),
```

```math
\hat o^{tar}
=
\frac{1}{|\mathcal P|}
\sum_{i\in\mathcal P}
\hat A_i(o^{src}).
```

The approximate target depth is reconstructed from the source depth, crop
scales, and focal lengths:

```math
\hat z^{tar}
=
z^{src}
\frac{
\alpha^{tar}
}{
\alpha^{src}
}
\frac{
f_x^{tar}
}{
f_x^{src}
}
\frac{1}{\bar{s}}.
```

Here \(\alpha\) is the crop scaling term from the crop transform matrix. The
predicted target crop center is mapped back through the inverse crop transform
and camera intrinsics to form a camera ray \(r(\hat o^{tar})\), giving:

```math
\hat t^{tar}
=
r(\hat o^{tar})\hat z^{tar}.
```

The approximate rotation is:

```math
\hat R^{tar}
=
R_{\rm inplane}(\bar u) R^{src}.
```

The optional direct losses are normalized robust losses:

```math
L_{\rm direct\text{-}translation}
=
\rho_{\beta_t}
\left(
\frac{\|\hat t^{tar}-t^{tar*}\|_2}{S_t}
\right),
```

```math
L_{\rm direct\text{-}depth}
=
\rho_{\beta_d}
\left(
\frac{|\hat z^{tar}-z^{tar*}|}{S_d}
\right),
```

```math
L_{\rm direct\text{-}rotation}
=
\rho_{\beta_R}
\left(
\frac{
d_{SO(3)}(\hat R^{tar}, R^{tar*})
}{\pi}
\right),
```

where:

```math
d_{SO(3)}(R_1,R_2)
=
\operatorname{atan2}
\left(
\frac{1}{2}
\left\|
\begin{bmatrix}
R_{32}-R_{23}\\
R_{13}-R_{31}\\
R_{21}-R_{12}
\end{bmatrix}
\right\|_2,
\frac{
\operatorname{tr}(R)-1
}{2}
\right).
```

Here \(R=R_1^\top R_2\). This is the same SO(3) geodesic angle as the
traditional `acos` formula, but the `atan2(sin(theta), cos(theta))` form is
more numerically stable near perfect alignment and near 180 degrees.

\(S_t\) and \(S_d\) are normalization scales in the same units as
`cam_t_m2c`. For Assetto datasets, the defaults are 1000 mm. If the data is in
metres, the defaults are 1.0 m.

`L_direct-reprojection` is the same normalized center reprojection term as
`L_reprojection`. Its default weight is 0.0 because `L_reprojection` is already
part of the base objective; the extra knob exists only if you want to emphasize
that term when the direct pose block is enabled.

For `--nets-to-train all`, the objective additionally contains:

```math
L_{\rm all} = L_{\rm IST} +
\lambda_{\rm NCE}L_{\rm InfoNCE} +
\lambda_{\rm soft}L_{\rm soft-template}.
```

The soft-template loss mask-pools AE feature maps, calculates an in-batch
template probability

```math
p_k = \operatorname{softmax}(a_k/\tau),
```

and constructs a pose-aware target distribution

```math
q_k = \operatorname{softmax}(-d_{SO(3)}(R_k,R^*)/\tau_{\rm pose}).
```

It minimizes:

```math
L_{\rm soft-template}=-\sum_k q_k\log p_k.
```

Only candidates with the same object label participate. This is deliberately
an in-batch objective: it does not run hard retrieval or RANSAC during every
training step.

## Monitoring-only pose reconstruction

The code robustly averages patch log-scales, unit rotation vectors, and
predicted projected centers as described above. It then reconstructs an
approximate pose using the calibrated crop transforms and intrinsics. The
following human-readable metrics are always logged:

- `monitor_translation_error_mm`
- `monitor_depth_abs_error_mm`
- `monitor_rotation_error_deg`
- `monitor_reprojection_error_px`

With the default `--no-optimize-pose-monitor-errors`, these are **not included
in the loss**. They are dashboard metrics only. With
`--optimize-pose-monitor-errors`, normalized differentiable versions of these
errors are added through `loss_direct_translation`, `loss_direct_depth`,
`loss_direct_rotation`, and optionally `loss_direct_reprojection`.

Actual inference continues to use normal GigaPose retrieval and RANSAC. The
direct pose losses optimize the differentiable IST approximation, not the final
hard RANSAC output.

## IST-only example

```bash
python -m fine_tuning.pose_aware_training.train \
  --dataset-name assettocorsa_new_dataset \
  --train-split train_pbr_web_gsam_clean \
  --validation-split val_pbr_web_gsam_clean \
  --checkpoint gigaPose_datasets/pretrained/gigaPose_v1.ckpt \
  --nets-to-train ist \
  --ist-lr 1e-5 \
  --batch-size 16 \
  --num-workers 4 \
  --max-steps 25000 \
  --validation-interval 250 \
  --checkpoint-interval 1000 \
  --run-name assettocorsa_pose_aware_ist \
  --logger wandb \
  --devices 0
```

The InfoNCE and soft-template weights are ignored in IST-only mode because the
AE network is frozen.

## Joint AE + IST example

Joint training is more memory-intensive. Starting with the final DINOv2 block
is safer than immediately unfreezing the entire backbone:

```bash
python -m fine_tuning.pose_aware_training.train \
  --dataset-name assettocorsa_new_dataset \
  --train-split train_pbr_web_gsam_clean \
  --validation-split val_pbr_web_gsam_clean \
  --checkpoint gigaPose_datasets/pretrained/gigaPose_v1.ckpt \
  --nets-to-train all \
  --ist-lr 1e-5 \
  --ae-lr 1e-6 \
  --ae-train-mode last-blocks \
  --ae-train-last-n-blocks 1 \
  --batch-size 8 \
  --num-workers 4 \
  --max-steps 25000 \
  --validation-interval 250 \
  --checkpoint-interval 1000 \
  --run-name assettocorsa_pose_aware_all \
  --logger wandb \
  --devices 0
```

Larger batches provide more in-batch template candidates. If memory prevents
that, gradient accumulation does not automatically enlarge the soft-template
candidate set; a cross-batch queue would be a separate extension.

## Default weights

| Term | Default |
|---|---:|
| log-depth | 1.0 |
| in-plane | 1.0 |
| reprojection | 0.1 |
| optimize direct pose monitor errors | false |
| direct translation | 0.05, used only when enabled |
| direct depth | 0.05, used only when enabled |
| direct rotation | 0.05, used only when enabled |
| direct reprojection extra | 0.0, used only when enabled |
| InfoNCE (`all`) | 1.0 |
| soft template (`all`) | 0.25 |

These are conservative starting values. Compare both the validation losses and
the monitoring-only millimetre/degree metrics before changing them.

To enable the direct pose terms:

```bash
python -m fine_tuning.pose_aware_training.train \
  ... \
  --optimize-pose-monitor-errors \
  --direct-translation-weight 0.05 \
  --direct-depth-weight 0.05 \
  --direct-rotation-weight 0.05
```

If the direct losses make training noisy, turn them back off and keep using the
base objective while watching the monitor metrics.

## Where results are written

For run `MY_RUN`:

```text
gigaPose_datasets/results/MY_RUN/
├── checkpoints/
├── pose_metrics.csv
├── validation_images/
└── heavy_validation/       # only when enabled
```

`pose_metrics.csv` is long-form and can be plotted with:

```bash
python -m fine_tuning.pose_aware_training.plot_metrics \
  gigaPose_datasets/results/MY_RUN/pose_metrics.csv
```

This writes `pose_metric_history.png` beside the CSV.
