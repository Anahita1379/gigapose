# Residual-pose GigaPose fine-tuning

This package is isolated from the existing training implementations.  It does
not modify `src/models/gigaPose.py`, `fine_tuning.train`, or
`fine_tuning.pose_aware_training`.

It adds two corrections after the ordinary IST pose estimate:

1. a mandatory translation residual head predicting crop-center offsets and a
   log-depth residual;
2. an optional bounded three-axis SO(3) rotation residual head.

The same residual heads are applied by the package's end-to-end inference
entry point.  Do not use the repository's ordinary `test.py` with one of these
checkpoints: that script instantiates the original model and does not know how
to apply the new heads.

## Geometry

Let the ordinary GigaPose/IST reconstruction provide a baseline pose

\[
T_0 = [R_0\mid t_0], \qquad t_0=(x_0,y_0,z_0)^T.
\]

The baseline translation is projected through the target camera and crop:

\[
p_0 = \pi(MK t_0)=(u_0,v_0)^T.
\]

The translation head predicts three raw values.  They are bounded and mapped
to geometric residuals:

\[
\Delta u = u_{\max}\tanh(a_u),\qquad
\Delta v = u_{\max}\tanh(a_v),
\]

\[
\Delta\log z = \zeta_{\max}\tanh(a_z).
\]

The refined center and depth are

\[
\hat p=p_0+(\Delta u,\Delta v)^T,
\qquad
\hat z=z_0\exp(\Delta\log z).
\]

Translation is reconstructed through the inverse crop and camera matrices:

\[
\boxed{
\hat t = \hat z\,
\operatorname{normalize}_z
\left(K^{-1}M^{-1}[\hat u,\hat v,1]^T\right)
}
\]

This parameterization cannot produce an arbitrary translation inconsistent
with the camera projection.  The default bounds are 56 crop pixels and
`abs(delta log z) <= 0.5`, corresponding to a maximum depth multiplier of
approximately `exp(0.5) = 1.65` in either direction.

When `--rotation-residual` is enabled, the second head predicts an axis-angle
vector.  Its magnitude is bounded to `--max-rotation-deg` and applied in the
camera frame:

\[
\hat R = \exp([\Delta\omega]_\times)R_0.
\]

Axis-angle is used instead of Euler angles to avoid wrapping and gimbal-lock
singularities.  Rotation is a proper SO(3) transform: no reflection is
representable.

## Residual objective

The optimized translation-head objective is

\[
L_{\rm residual,t}=
\lambda_c L_{\rm center}
+\lambda_z L_{\log z}
+\lambda_t L_{\rm metric\ translation}
+\lambda_{reg} L_{\rm residual\ magnitude}.
\]

The terms are

\[
L_{\rm center}=\rho
\left(\frac{\|\hat p-p^*\|_2}{\sqrt{H^2+W^2}}\right),
\]

\[
L_{\log z}=\rho(\log\hat z-\log z^*),
\]

\[
L_{\rm metric\ translation}=\rho
\left(\frac{\|\hat t-t^*\|_2}{1000\ {\rm mm}}\right).
\]

The robust function `rho` is Smooth-L1.  Center and log depth are the primary
losses.  The metric translation term has a smaller default weight because it
couples lateral and depth errors and can otherwise dominate distant cars.

With optional rotation,

\[
L_{\rm residual}=L_{\rm residual,t}
+\lambda_R\rho
\left(\frac{d_{SO(3)}(\hat R,R^*)}{\pi}\right).
\]

The ordinary instance-balanced pose-aware IST losses remain active when IST is
trainable.  `--no-train-ist` freezes IST while still computing those values for
baseline comparison; gradients then update only the residual head.

## Recommended first run: translation head only

Start from the best existing center-refinement checkpoint and freeze IST.  This
isolates whether the explicit head improves translation without moving a good
scale/in-plane solution:

```bash
python -m fine_tuning.residual_pose_training.train \
  --dataset-name assettocorsa_new_dataset \
  --train-split train_pbr_web_gsam_clean \
  --validation-split val_pbr_web_gsam_clean \
  --checkpoint gigaPose_datasets/results/assettocorsa_pose_aware_instance_scale_center_refinement/checkpoints/best-scale-step002250.ckpt \
  --nets-to-train ist \
  --no-train-ist \
  --residual-lr 1e-4 \
  --batch-size 16 \
  --num-workers 4 \
  --max-steps 5000 \
  --validation-interval 250 \
  --checkpoint-interval 1000 \
  --run-name assettocorsa_translation_residual \
  --logger wandb \
  --print-loss-every 50 \
  --devices 0 \
  --residual-center-weight 1.0 \
  --residual-log-depth-weight 1.0 \
  --residual-translation-weight 0.05 \
  --residual-regularization-weight 0.001 \
  --no-rotation-residual \
  --heavy-validation \
  --heavy-validation-interval 1000 \
  --heavy-validation-images 4
```

The residual head is zero-initialized.  Validation step zero must therefore
match the source checkpoint up to normal validation variation.

## Optional joint IST + translation-head refinement

After the residual-only run, initialize from its best residual checkpoint and
allow a small IST learning rate:

```bash
python -m fine_tuning.residual_pose_training.train \
  --dataset-name assettocorsa_new_dataset \
  --train-split train_pbr_web_gsam_clean \
  --validation-split val_pbr_web_gsam_clean \
  --checkpoint gigaPose_datasets/results/assettocorsa_translation_residual/checkpoints/best-residual-stepXXXXXX.ckpt \
  --nets-to-train ist \
  --train-ist \
  --ist-lr 2e-6 \
  --residual-lr 2e-5 \
  --batch-size 16 \
  --num-workers 4 \
  --max-steps 3000 \
  --validation-interval 250 \
  --checkpoint-interval 1000 \
  --run-name assettocorsa_translation_residual_joint \
  --logger wandb \
  --print-loss-every 50 \
  --devices 0 \
  --no-rotation-residual
```

## Enable continuous rotation correction

For a separate experiment, add:

```bash
  --rotation-residual \
  --residual-rotation-weight 0.25 \
  --max-rotation-deg 20
```

Use a different run name.  Rotation changes the question being tested, so the
translation-only run should remain the initial comparison.

## Logged metrics and checkpoints

The run writes `pose_metrics.csv`.  Important validation fields are:

- `val/monitor_baseline_translation_error_mm`
- `val/monitor_refined_translation_error_mm`
- `val/monitor_translation_improvement_mm`
- `val/monitor_baseline_depth_error_mm`
- `val/monitor_refined_depth_error_mm`
- `val/monitor_depth_improvement_mm`
- `val/monitor_baseline_center_error_px`
- `val/monitor_refined_center_error_px`
- `val/monitor_center_improvement_px`
- `val/monitor_baseline_rotation_error_deg`
- `val/monitor_refined_rotation_error_deg`
- `val/monitor_center_offset_px`
- `val/monitor_abs_log_depth_residual`
- `val/monitor_rotation_residual_deg`

Positive `*_improvement_*` means the residual head improved the baseline.

The top checkpoints ranked by refined validation translation error are saved
as `best-residual-stepXXXXXX.ckpt`.  Periodic and `last.ckpt` files are also
written by the normal checkpoint callback.

When heavy validation is enabled, its CAD overlays use the refined pose because
the model's end-to-end `eval_retrieval` method applies the residual head.

## End-to-end inference

Use this package's inference entry point.  The head size, bounds, translation
unit, and rotation toggle must match training:

```bash
python -m fine_tuning.residual_pose_training.infer \
  --dataset-name assettocorsa_benchmark \
  --checkpoint gigaPose_datasets/results/assettocorsa_translation_residual/checkpoints/best-residual-stepXXXXXX.ckpt \
  --run-name assettocorsa_translation_residual_benchmark \
  --batch-size 16 \
  --num-workers 4 \
  --devices 0 \
  --apply-residual \
  --no-rotation-residual \
  --max-center-offset-px 56 \
  --max-log-depth-residual 0.5
```

For a rotation-enabled checkpoint, replace `--no-rotation-residual` with
`--rotation-residual` and use the same `--max-rotation-deg` as training.

For a controlled baseline using the identical checkpoint, detections,
retrieval features, and matching thresholds, run the command again with a new
run name and:

```bash
--no-apply-residual
```

Comparing these two CSVs isolates the effect of the residual head.

Prediction CSVs are written under

```text
gigaPose_datasets/results/<run-name>/predictions/
```

Evaluate those CSVs against the exact same benchmark and detections used for
the baseline.  Lightweight training validation uses prepared GT patch pairs;
only end-to-end inference measures the combined effects of retrieval,
correspondence matching, RANSAC, IST, and the residual head.
