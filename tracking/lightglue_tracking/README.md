# LightGlue-assisted RGB pose tracking

This folder is an isolated extension of the existing tracking system. It does
not modify:

- `tracking/run_tracking.py`;
- `tracking/rgb_self_recovery/`;
- the GigaPose training code;
- existing tracking configuration files.

The new runner uses the trained RGB self-recovery model as its main candidate
refiner and adds frozen ALIKED + LightGlue for temporal matching, multi-car
association, and optional metric PnP recovery.

## Why this extends RGB self-recovery

LightGlue does not predict a 6D pose by itself. The existing RGB self-recovery
pipeline already has:

- multiple GigaPose and temporal pose hypotheses;
- CAD rendering;
- RGB/render candidate refinement;
- silhouette scoring;
- temporal state and orientation gates.

This is therefore the correct place to add correspondence evidence. The
ordinary tracker remains available as the faster deterministic baseline.

## Components

### 1. Frozen ALIKED + LightGlue

ALIKED detects sparse keypoints and descriptors inside each detected car mask.
LightGlue matches them between a stored real-image keyframe and the current
real image. All parameters are frozen.

The local checkout is expected at `LightGlue/`. On the first run, its official
ALIKED and LightGlue weights are downloaded into the Torch checkpoint cache.

### 2. RANSAC similarity propagation

Given matched pixels $\mathbf{x}_{k}^{q}$ in a keyframe and
$\mathbf{x}_{k}^{t}$ in the current frame, the tracker estimates

$$
A^*
=
\underset{A}{\operatorname{argmin}}
\sum_k
\rho\left(
\left\|
\mathbf{x}_{k}^{t}
-
A\widetilde{\mathbf{x}}_{k}^{q}
\right\|_2
\right).
$$

`cv2.estimateAffinePartial2D` restricts $A$ to translation, uniform scale, and
in-plane rotation. A proposal is accepted only when the match count, RANSAC
inlier count, inlier ratio, scale, and median reprojection error pass their
configured gates.

By default this proposal runs only for `uncertain` or `lost` tracks:

```text
--lightglue-flow-policy uncertain_lost
```

Lucas–Kanade remains a fallback whenever LightGlue does not produce a valid
similarity transform.

### Rotation-rate versus no-rotation selection

For every existing track, the candidate bank contains competing temporal
branches:

- robust constant translation and rotation velocity;
- the same propagated translation with the last accepted rotation held fixed;
- the LightGlue-observed in-plane similarity rotation when available;
- the optional full 3D LightGlue-PnP rotation;
- fresh GigaPose rotation hypotheses on global-check frames.

The hold-rotation branch is saved with source
`constant_translation_zero_rotation`. This lets the RGB/render scorer prefer
no angular motion instead of accumulating a noisy angular velocity.
`lightglue_rotation_deg` records the RANSAC similarity angle for inspection.

### 3. LightGlue-aware Hungarian association

For ambiguous track/detection pairs, the Hungarian cost additionally contains
the verified LightGlue quality $q_{ij}$:

$$
C_{ij}
=
\frac{
w_{\mathrm{IoU}}(1-\mathrm{IoU}_{ij})
+
w_c d_{ij}
+
w_{\mathrm{LG}}(1-q_{ij})
+
\text{existing identity terms}
}{
w_{\mathrm{IoU}}+w_c+w_{\mathrm{LG}}+\cdots
}.
$$

The default `ambiguous` policy invokes LightGlue when:

- more than one track can plausibly match a detection;
- more than one detection can plausibly match a track; or
- the track is already uncertain/lost.

This avoids paying the full matching cost on ordinary unambiguous frames.

### 4. Optional temporal render-backed PnP

Directly matching real RGB to an untextured CAD normal/depth render is not
reliable. Instead, the optional PnP path uses an accepted real frame as its
visual template:

1. Extract ALIKED descriptors from the accepted real car crop.
2. Render CAD depth at its accepted pose.
3. Back-project every template keypoint with valid rendered depth.
4. Transform those camera points into CAD/object coordinates.
5. LightGlue matches the real template descriptors to the current real image.
6. PnP-RANSAC estimates the current full metric pose.

For a template object point $\mathbf{X}_k$ and current image match
$\mathbf{u}_k$, PnP solves

$$
\mathbf{u}_k
\sim
\pi\left(
K\left(R\mathbf{X}_k+\mathbf{t}\right)
\right).
$$

No observed depth is used. Rendered CAD depth is used only to attach fixed 3D
object coordinates to the earlier real-image descriptors.

This path is disabled by default. Enable it only after comparing the basic
LightGlue tracker:

```text
--lightglue-pnp-recovery
--lightglue-pnp-policy uncertain_lost
```

## Recommended smoke run

Replace the prediction and checkpoint paths with the ones currently present
on the machine:

```bash
python -m tracking.lightglue_tracking.run \
  --predictions gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v1v4_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v1v4_front_gsam_v4-test_large_real_20260518v1v4_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_front_gsam_v4 \
  --split test \
  --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
  --association-config tracking/configs/improved.json \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/real_20260518v1v4_front_rotation_gated_lightglue_tracking_smoke \
  --device cuda \
  --max-frames 100 \
  --lightglue \
  --lightglue-flow-policy uncertain_lost \
  --lightglue-association \
  --lightglue-association-policy ambiguous \
  --lightglue-lk-fallback \
  --orientation-gates \
  --no-allow-flip-hypotheses \
  --save-overlays \
  --overlay-every 10 \
  --overwrite
```

The smoke run deliberately leaves PnP disabled. Check
`candidate_diagnostics.csv` for:

- `lightglue_matches`;
- `lightglue_inliers`;
- `lightglue_inlier_ratio`;
- `lightglue_reprojection_px`;
- `lightglue_quality`;
- `lightglue_rotation_deg`.

## Full run with temporal PnP

After the smoke run shows reliable matches:

```bash
python -m tracking.lightglue_tracking.run \
  --predictions gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v1v4_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v1v4_front_gsam_v4-test_large_real_20260518v1v4_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv  \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_front_gsam_v4 \
  --split test \
  --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
  --association-config tracking/configs/improved.json \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/light_glue_runs/real_20260518v1v4_front_rotation_gated_lightglue_tracking \
  --device cuda \
  --lightglue \
  --lightglue-flow-policy uncertain_lost \
  --lightglue-association-policy ambiguous \
  --lightglue-pnp-recovery \
  --lightglue-pnp-policy uncertain_lost \
  --orientation-gates \
  --no-allow-flip-hypotheses \
  --save-overlays \
  --overlay-every 10 \
  --overwrite
```

PnP diagnostics add:

- `lightglue_pnp_inliers`;
- `lightglue_pnp_inlier_ratio`;
- `lightglue_pnp_reprojection_px`;
- `lightglue_pnp_quality`.

## Evaluation against GT

Run the original RGB tracker and the new LightGlue tracker on the same
synthetic benchmark. Then compare both:

```bash
python -m tracking.lightglue_tracking.evaluate \
  --model rgb=gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_front_rotation_gated/tracked_predictions.csv \
  --model lightglue=gigaPose_datasets/results/lightglue_tracking/tracked_predictions.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_front_gsam_v4 \
  --split test \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/light_glue_runs/lightglue_tracking_comparison
```

The evaluator saves:

- `per_instance_errors.csv`;
- `model_summary.csv`;
- `summary.json`.

For each model it reports:

- translation mean, median, and p90 in metres;
- rotation mean, median, and p90 in degrees;
- missed GT instances and false positives;
- lost-state rate;
- approximate identity-switch count;
- elapsed runtime and frames per second when `run_report.json` is present.

The identity metric assumes that the ordering of objects in Assetto
`gt.json` stays stable within each scene. It should be interpreted together
with pose errors and overlays.

Real-world recordings without GT cannot provide translation or rotation
accuracy. They can still be compared using
`tracking.compare_predictions_without_gt`, but that measures disagreement and
temporal consistency rather than accuracy.

## Runtime controls

The main controls are:

| Option | Effect |
|---|---|
| `--lightglue-max-keypoints` | More keypoints improve coverage but cost memory/time. |
| `--lightglue-resize` | ALIKED extraction resolution. |
| `--lightglue-association-policy ambiguous` | Match only ambiguous or weak associations. |
| `--lightglue-flow-policy uncertain_lost` | Do not run LightGlue flow on normal tracks. |
| `--lightglue-keyframe-interval 10` | Refresh normal-track visual memory every ten accepted frames. |
| `--lightglue-pnp-policy uncertain_lost` | Reserve full PnP for recovery. |
| `--no-lightglue-lk-fallback` | Disable the existing LK fallback for ablation only. |

Start with 1024 keypoints and resize 640. If runtime is too high, try:

```text
--lightglue-max-keypoints 512
--lightglue-resize 512
```

Do not increase the LightGlue association weight until the diagnostics show
that correct car pairs consistently have higher verified quality than wrong
pairs.

## Output compatibility

`tracked_predictions.csv` uses the same format as the existing trackers and
can be passed to the existing label-selection and prediction-comparison tools.
`run_report.json` records every LightGlue option, invocation counts, extractor
time, matcher time, state counts, lost rate, and overall throughput.
