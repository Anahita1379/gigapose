# Pose-aware GigaPose fine-tuning

This is an isolated experimental training path. It does not modify
`fine_tuning.train`, `fine_tuning.train_val`, `src.models.gigaPose`, or the
existing Assetto Corsa loader.

## Objectives

For IST-only training, the default optimized objective is:

```math
L_{\rm IST,base} =
\lambda_z L_{\text{balanced-patch-scale}} +
\lambda_s L_{\text{instance-scale}} +
\lambda_c L_{\text{scale-consistency}} +
\lambda_\theta L_{\text{inplane}} +
\lambda_p L_{\text{reprojection}} +
\lambda_f L_{\text{anti-flip}}.
```

- `balanced-patch-scale`: Smooth-L1 of
  `log(predicted_relative_scale) - log(gt_relative_scale)`. Since target depth
  is inversely proportional to relative scale, this is the relative log-depth
  error up to sign. Patch losses are averaged inside each car first, so cars
  with more valid correspondences cannot dominate the batch.
- `instance-scale`: robustly supervises the geometric-mean patch scale for each
  car, which is the aggregate used by differentiable pose reconstruction.
- `scale-consistency`: discourages patch log-scales for one car from spreading
  around their per-car mean.
- `inplane`: `1 - cos(predicted_angle - gt_angle)`. This is bounded and avoids
  the unstable derivative of `acos` close to a perfect prediction.
- `reprojection`: patch transforms are first aggregated into one predicted
  target center per car. That center is compared with the ground-truth target
  center, normalized by the 224x224 crop diagonal, using Smooth-L1. Its default
  weight is 0 because the previous Assetto run showed that the old per-patch
  reprojection objective conflicted with scale.
- `anti-flip`: optional margin penalty that discourages mirrored in-plane patch
  predictions. Its default weight is 0.0, so flip diagnostics are logged without
  changing optimization unless `--anti-flip-weight` is positive.

More explicitly, for each valid patch correspondence \(i\), IST predicts a
relative scale \(\hat s_i\) and an in-plane rotation represented as a normalized
2-vector \(\hat u_i=[\hat c_i,\hat s_i]\). The ground truth values are
\(s_i^*\) and \(\theta_i^*\). With a Smooth-L1/Huber residual
\(\rho_\beta(\cdot)\):

```math
L_{\text{balanced-patch-scale}}
=
\frac{1}{B}\sum_b\frac{1}{N_b}\sum_{i\in\mathcal P_b}
\rho_{\beta_z}
\left(
\log(\hat s_i+\epsilon)-\log(s_i^*+\epsilon)
\right).
```

Target depth is inversely proportional to relative scale, so this is a relative
log-depth loss up to sign. In code, non-positive predicted scales are clamped
to a tiny positive value before `log` and geometric reconstruction. This matches
the original log-scale loss behavior and avoids dropping the whole patch pair
when only the predicted scale is invalid.

For car (b), define the aggregate log-scale and geometric-mean scale as:

```math
\bar\ell_b=\frac{1}{N_b}\sum_{i\in\mathcal P_b}\log\hat s_{bi},
\qquad
\bar s_b=\exp(\bar\ell_b).
```

The additional scale objectives are:

```math
L_{\text{instance-scale}}
=
\frac{1}{B}\sum_b
\rho_{\beta_z}\left(\bar\ell_b-\log s_b^*\right),
```

```math
L_{\text{scale-consistency}}
=
\frac{1}{B}\sum_b\frac{1}{N_b}\sum_{i\in\mathcal P_b}
\rho_{\beta_z}\left(\log\hat s_{bi}-\bar\ell_b\right).
```

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

Negative angles are handled naturally by the \([\cos\theta,\sin\theta]\)
representation. A flipped prediction is different: it is a reflection, not a
proper rotation. To detect that failure mode, the code compares each prediction
against the correct target and a mirrored target:

```math
u_i^*
=
\begin{bmatrix}
\cos\theta_i^*\\
\sin\theta_i^*
\end{bmatrix},
\qquad
u_{i,\mathrm{flip}}^*
=
\begin{bmatrix}
\cos\theta_i^*\\
-\sin\theta_i^*
\end{bmatrix}.
```

```math
d_i^{correct}=1-\hat u_i^\top u_i^*,
\qquad
d_i^{flip}=1-\hat u_i^\top u_{i,\mathrm{flip}}^*.
```

The logged flip diagnostics are:

```math
\mathrm{flip\_margin}
=
\frac{1}{N}\sum_i
\left(d_i^{flip}-d_i^{correct}\right),
```

and the fraction of patch pairs where the mirrored target is closer:

```math
\mathrm{flip\_closer\_fraction}
=
\frac{1}{N}\sum_i
\mathbb{1}\left[d_i^{flip}<d_i^{correct}\right].
```

Positive `flip_margin` is good: the correct target is closer than the mirrored
one. If `--anti-flip-weight` is positive, the optimized penalty is:

```math
L_{\text{anti-flip}}
=
\frac{1}{N}\sum_i
\max
\left(
0,\,
m+d_i^{correct}-d_i^{flip}
\right),
```

where \(m\) is `--anti-flip-margin`. This does not allow IST to output
reflections; it simply penalizes patch-pair predictions that look more like the
mirrored target than the correct one.

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

Let \(o_b^{src}\) and \(o_b^{tar}\) be the object-center projections for car
\(b\). The patch predictions are aggregated before applying the loss:

```math
\hat o_b^{tar}
=
\frac{1}{N_b}\sum_{i\in\mathcal P_b}\hat A_{bi}(o_b^{src}).
```

The center reprojection term is then:

```math
L_{\text{reprojection}}
=
\frac{1}{B}\sum_b
\rho_{\beta_p}
\left(
\frac{
\left\|
\hat o_b^{tar} - o_b^{tar}
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
`L_reprojection`. Its default weight is 0.0. The extra knob exists only if you
want to enable that term as part of the direct-pose block independently of the
base `--reprojection-weight`.

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
- `monitor_scale_signed_log_bias` (positive means systematic over-scaling)
- `monitor_scale_abs_log_error` (used to rank best-scale checkpoints)
- `monitor_scale_median_pred_gt_ratio` (ideal value is 1)
- `monitor_scale_within_5pct`
- `monitor_scale_within_10pct`
- `monitor_scale_within_20pct`
- `monitor_scale_log_std` (within-car patch disagreement; lower is better)
- `monitor_flip_closer_fraction`
- `monitor_flip_margin`

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
  --devices 0 \
  --log-depth-weight 1.0 \
  --instance-log-scale-weight 0.5 \
  --scale-consistency-weight 0.05 \
  --inplane-weight 0.5 \
  --reprojection-weight 0.0
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
| balanced patch log-scale | 1.0 |
| instance log-scale | 0.5 |
| scale consistency | 0.05 |
| in-plane | 1.0 |
| reprojection | 0.0 |
| anti-flip | 0.0 |
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

To add the mirrored-prediction penalty, start gently:

```bash
--anti-flip-weight 0.01 \
--anti-flip-margin 0.25
```

Before optimizing it, check `val/monitor_flip_closer_fraction`. If it is already
near zero, you probably do not need this term.

## Where results are written

For run `MY_RUN`:

```text
gigaPose_datasets/results/MY_RUN/
├── checkpoints/
├── pose_metrics.csv
├── validation_images/
└── heavy_validation/       # only when enabled
```

In addition to periodic checkpoints, training keeps the best three
`best-scale-step*.ckpt` files ranked by
`val/monitor_scale_abs_log_error`. Change the number with
`--best-scale-checkpoints`, or pass `--best-scale-checkpoints 0` to disable
scale-ranked checkpointing.

The lightweight IST overlay prints `pred`, `gt`, and `ratio=pred/gt` for
relative crop scale. Red is the warped template prediction and green is the
target mask. It remains a 2D IST diagnostic rather than a full CAD-pose render.

`pose_metrics.csv` is long-form and can be plotted with:

```bash
python -m fine_tuning.pose_aware_training.plot_metrics \
  gigaPose_datasets/results/MY_RUN/pose_metrics.csv
```

This writes `pose_metric_history.png` beside the CSV.
