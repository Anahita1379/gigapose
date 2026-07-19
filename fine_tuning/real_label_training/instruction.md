# Using extrinsics-corrected EPnP labels and GigaPose predictions

## Purpose

This document explains how to use the real-world samples selected by
`fine_tuning.select_real_label_candidates_with_extrinsics` in a new training
pipeline.

The downstream model does not have to be GigaPose. The same records can support:

- direct 6D-pose regression;
- keypoint, correspondence, and PnP models;
- template retrieval and contrastive representation learning;
- pose-refinement or residual networks;
- confidence and hypothesis-ranking networks;
- differentiable rendering and CAD-alignment objectives;
- detector and instance-segmentation training;
- temporal tracking and map-frame trajectory training;
- teacher-student or pseudo-label domain adaptation.

The most important rules are:

1. Treat the selected rows as **extrinsics-corrected pseudo-labels**, not
   automatically exact ground truth.
2. Use `T_epnp_obj` as the default camera-frame target.
3. Use the centered CAD convention with that target.
4. Do not apply the optimized extrinsic correction a second time.
5. Do not use an empirically aligned GigaPose prediction as ground truth.
6. Keep camera/session identity when combining files.
7. Split sequential sessions before constructing train/validation subsets.

## Audit of the current selected data

As of 2026-07-18, the folders named
`label_candidates_with_optimized_extrinsics_fixed` contain:

- 11 selection outputs;
- 2,659 selected rows;
- camera types including front, rear, and stereo-left;
- the complete 32-column pose and provenance contract;
- a right-side GigaPose-to-EPnP comparison alignment;
- camera-specific optimized-extrinsics paths.

The saved map-frame and camera-frame calculations are internally consistent:

- `T_epnp_obj` and `T_epnp_camera_obj_optimized` are exactly identical in all
  2,659 selected rows;
- maximum matrix-entry error in
  `T_epnp_map_obj ≈ T_map_cam_optimized @ T_epnp_obj` is approximately
  `1.1e-6`;
- median map/camera translation disagreement: approximately
  `0.000039 mm`;
- median map/camera rotation disagreement: approximately
  `0.000035 degrees`.

All 2,659 corrected targets also passed the basic structural checks:

- no invalid rotation determinants;
- no non-positive camera depths;
- no invalid homogeneous last rows.

This confirms that the transform composition used by the selector is
algebraically consistent.

It does **not** prove that every selected pose is physically exact. Across the
current selected rows, the GigaPose-versus-corrected-EPnP agreement is:

- translation mean: approximately `1393 mm`;
- translation median: approximately `1433 mm`;
- translation p90: approximately `2246 mm`;
- rotation mean: approximately `7.51 degrees`;
- rotation median: approximately `6.00 degrees`;
- rotation p90: approximately `14.60 degrees`.

The selection cutoff was intentionally loose:

- translation: at most `2500 mm`;
- total rotation: at most `30 degrees`;
- roll: at most `10 degrees`;
- pitch: at most `10 degrees`;
- yaw: at most `20 degrees`.

These errors measure agreement between the selected GigaPose prediction and
the corrected EPnP pose. They are not direct measurements of EPnP label error.
A large disagreement can be caused by GigaPose, EPnP, or both. Therefore,
agreement is a useful quality proxy but not a guarantee.

The current `*_fixed` selection folders do not contain completed visual-overlay
indexes. Before using every row for precision pose supervision, render and
inspect a stratified sample from every camera/session.

## What the optimized-extrinsics selector computes

For each EPnP record, define:

$$
M = T_{\mathrm{map,obj}},
\qquad
E = T_{\mathrm{cam,obj}}^{\mathrm{original}}.
$$

The original map-from-camera transform implied by the label is:

$$
C = M E^{-1}.
$$

The optimized-extrinsics JSON contains a left correction $D$. The selector
computes:

$$
C_{\mathrm{opt}} = D C.
$$

The corrected camera-frame target is:

$$
T_{\mathrm{cam,obj}}^{\mathrm{opt}}
=
C_{\mathrm{opt}}^{-1}M.
$$

That corrected target is written to both:

- `T_epnp_obj`;
- `T_epnp_camera_obj_optimized`.

Therefore:

$$
\boxed{
T_{\mathrm{target}}
=
T_{\mathrm{epnp\_obj}}
=
T_{\mathrm{epnp\_camera\_obj\_optimized}}
}
$$

within numerical precision.

The optimized extrinsic has already affected the saved target. A camera-frame
training loader should not read the extrinsics JSON and correct
`T_epnp_obj` again.

## Visualizer centering versus extrinsic correction

The corrected extrinsics and CAD centering solve different problems.

The extrinsic correction changes the relationship between camera and map
coordinates:

$$
T_{\mathrm{map,cam}}^{\mathrm{opt}}
=
D T_{\mathrm{map,cam}}^{\mathrm{initial}}.
$$

CAD centering changes the object-coordinate origin used for rendering:

$$
p_{\mathrm{centered}} = p_{\mathrm{raw}} - c.
$$

For the current CAD:

$$
c =
[6.89,\ 0.003,\ 498.77]\ \mathrm{mm}.
$$

The fixed visualizer centers the CAD vertices and draws:

- corrected EPnP: `T_epnp_obj`;
- aligned GigaPose: `T_gigapose_aligned_epnp_obj`.

It does not rewrite any pose, selected CSV, or optimized-extrinsics JSON.
Changing the visualizer therefore has no effect on selection or training
labels.

## Coordinate and unit contract

### Pose direction

Camera-frame labels use:

$$
T_{\mathrm{cam,obj}},
$$

meaning a point in object coordinates is transformed into OpenCV camera
coordinates:

$$
p_{\mathrm{cam}}
=
R_{\mathrm{cam,obj}}p_{\mathrm{obj}}
+t_{\mathrm{cam,obj}}.
$$

### Camera convention

Use the OpenCV convention:

- $+X$: image right;
- $+Y$: image down;
- $+Z$: forward from the camera.

### Translation units

All 4x4 matrices written by the selectors use translation in **millimetres**.

For BOP/MegaPose-style records:

- `cam_R_m2c`: row-major 3x3 rotation;
- `cam_t_m2c`: translation in millimetres.

Convert to metres only if the new model explicitly expects metres:

$$
t_{\mathrm{m}} = 0.001\,t_{\mathrm{mm}}.
$$

Do not infer the unit independently for each sample.

### Rotation

Every rotation should satisfy:

$$
R^\top R \approx I,
\qquad
\det(R)\approx 1.
$$

Do not convert a proper rotation into a reflection when changing CAD axes.

### Intrinsics

Camera intrinsics $K$ use pixel units. If an image is cropped, resized, or
padded, apply the identical geometric operation to:

- RGB;
- depth;
- segmentation and instance masks;
- bbox/keypoints;
- validity mask;
- camera intrinsics.

For a crop beginning at $(x_0,y_0)$:

$$
c_x' = c_x-x_0,\qquad c_y'=c_y-y_0.
$$

For a resize by $(s_x,s_y)$:

$$
f_x'=s_xf_x,\quad c_x'=s_xc_x,
\qquad
f_y'=s_yf_y,\quad c_y'=s_yc_y.
$$

For symmetric padding by `(left, top)`:

$$
c_x'=c_x+\mathrm{left},
\qquad
c_y'=c_y+\mathrm{top}.
$$

## CAD convention

The recommended convention for new pose training is:

1. load the original CAD;
2. convert its vertices to millimetres if necessary;
3. subtract the fixed CAD center;
4. use `T_epnp_obj` without an additional origin adjustment.

That is:

$$
p_{\mathrm{centered}}
=
p_{\mathrm{raw}}-c.
$$

If a new architecture absolutely requires the raw uncentered CAD, convert the
pose instead:

$$
T_{\mathrm{cam,raw}}
=
T_{\mathrm{cam,centered}}
\operatorname{Trans}(-c).
$$

Choose exactly one:

- centered CAD with `T_epnp_obj`; or
- raw CAD with the converted raw-CAD pose.

Do not center the CAD and also modify the pose. That would apply the origin
change twice.

## Meaning of every important CSV column

### Identity and provenance

- `match_key`: normalized image/EPnP matching key.
- `scene_id`, `im_id`: prepared-dataset frame identifier.
- `obj_id`: object class identifier.
- `instance_id`: GigaPose prediction-instance identifier.
- `prediction_row_index`: source row in the prediction CSV.
- `epnp_label_path`: original EPnP label JSON.
- `epnp_record_index`: selected record inside that JSON.
- `optimized_extrinsics_path`: camera correction used by this row.

`instance_id` is not guaranteed to equal the integer value inside a GSAM mask.
When building an object crop, use the original inference/detection mapping or
match the projected corrected pose to the nearest detection center. Do not
blindly interpret `instance_id` as a mask-pixel ID.

### Quality and selection diagnostics

- `score`: raw GigaPose prediction score.
- `translation_error_mm`: agreement between aligned GigaPose and corrected
  EPnP camera-frame poses.
- `rotation_error_deg`: total SO(3) agreement error.
- `roll_error_deg`, `pitch_error_deg`, `yaw_error_deg`: component diagnostics.
- `raw_*`: raw GigaPose versus original EPnP comparison.
- `original_*`: aligned GigaPose versus original EPnP comparison.
- `optimized_camera_*`: aligned GigaPose versus corrected EPnP comparison.
- `map_camera_*_disagreement`: consistency check between equivalent map- and
  camera-frame calculations.

These are selection diagnostics. They are not direct measurements against
independent motion-capture ground truth.

### Pose columns

- `T_epnp_obj`: recommended corrected camera-frame training target.
- `T_epnp_camera_obj_optimized`: explicit alias of the same corrected target.
- `T_epnp_camera_obj_original`: camera-frame EPnP pose before correction.
- `T_epnp_map_obj`: EPnP map-frame object pose.
- `T_map_cam_initial`: original map-from-camera transform.
- `T_map_cam_optimized`: corrected map-from-camera transform.
- `T_gigapose_cam_obj`: raw GigaPose prediction.
- `T_gigapose_aligned_epnp_obj`: prediction after empirical right alignment.
- `T_gigapose_map_obj_optimized`: aligned prediction mapped using corrected
  extrinsics.

## Which pose should be used?

### Default camera-frame supervised target

Use:

```text
T_epnp_obj
```

This is the correct default for a model that receives one camera image and
predicts object pose relative to that camera.

### Original uncorrected EPnP experiment

Use:

```text
T_epnp_camera_obj_original
```

This should only be used for a controlled ablation comparing original and
optimized extrinsics.

### Map/world-frame training

Use:

```text
T_epnp_map_obj
```

and retain:

```text
T_map_cam_optimized
```

The relationship should be:

$$
T_{\mathrm{map,obj}}
\approx
T_{\mathrm{map,cam}}^{\mathrm{opt}}
T_{\mathrm{cam,obj}}^{\mathrm{opt}}.
$$

### GigaPose prediction as a model input

Use:

```text
T_gigapose_cam_obj
```

if the new network is supposed to correct the raw deployed GigaPose output.

Use:

```text
T_gigapose_aligned_epnp_obj
```

only if the new network is explicitly defined after the empirical right-side
alignment.

Never silently mix raw and aligned initial poses in one training set.

### Do not use as the supervised target

Do not use either GigaPose column as ground truth for ordinary supervised pose
training. They are model predictions and were involved in sample selection.
They can be used as:

- initialization;
- teacher prediction;
- ranking hypothesis;
- hard-negative candidate;
- consistency target in semi-supervised learning;
- an input to a correction/refinement network.

## Building a safe unified manifest

Do not identify a sample using only `(scene_id, im_id, instance_id)` after
combining sessions. Scene and image IDs restart in separate prepared datasets.

Use a compound identity such as:

```text
(
  source_dataset_id,
  scene_id,
  im_id,
  epnp_label_path,
  epnp_record_index
)
```

A unified manifest should preserve:

- source dataset/session ID;
- camera ID;
- image path or WebDataset key;
- mask/detection association;
- camera intrinsics;
- corrected target pose;
- original target pose;
- raw and aligned GigaPose predictions;
- agreement metrics;
- optimized-extrinsics path;
- CAD version and CAD center;
- train/validation/test split.

Never use front-camera extrinsics for rear or stereo-left images. The current
optimized files are camera-specific. Reusing one across recording dates is
reasonable only when the physical camera mounting and calibration are
unchanged and a new overlay check confirms it.

## Quality tiers

The current selection threshold is useful for collecting candidates but is
loose for metric-pose supervision. Build explicit quality tiers.

The following are recommended starting points, not claims of ground-truth
accuracy.

### Tier A: strict

- GigaPose score at least `0.10`;
- agreement translation at most `500 mm`;
- total rotation at most `5 degrees`;
- roll/pitch/yaw each at most `3/3/5 degrees`;
- correct CAD overlay confirmed for a representative subset;
- correct instance/mask association.

There are currently 117 rows satisfying these numerical proxy conditions.

Recommended uses:

- high-weight direct pose supervision;
- metric translation/depth supervision;
- validation after manual visual confirmation;
- high-quality correspondence/keypoint targets.

### Tier B: moderate

- GigaPose score at least `0.10`;
- agreement translation at most `1000 mm`;
- total rotation at most `10 degrees`;
- roll/pitch/yaw each at most `5/5/10 degrees`.

There are currently 595 rows satisfying these numerical proxy conditions.

Recommended uses:

- medium-weight pose supervision;
- refinement/recovery training;
- template and representation learning;
- confidence/ranking training.

### Tier C: broad selected set

This is the complete 2,659-row selection under the existing 2.5 m and angular
limits.

Recommended uses:

- low-weight pseudo-label training;
- hard-example mining;
- hypothesis ranking;
- teacher-student consistency;
- initialization/recovery training;
- detector or segmenter training when the 2D annotation is reliable.

Do not give every Tier C row the same weight as manually verified metric GT.

## Optional continuous sample weighting

Agreement can be converted into a soft weight:

$$
w_t=\exp(-E_t/\tau_t),
\qquad
w_R=\exp(-E_R/\tau_R),
$$

$$
w_{\mathrm{sample}}
=
w_t w_R w_{\mathrm{score}} w_{\mathrm{visual}}.
$$

For example:

- $E_t$: `translation_error_mm`;
- $E_R$: `rotation_error_deg`;
- $\tau_t$: 500–1000 mm;
- $\tau_R$: 5–10 degrees;
- $w_{\mathrm{score}}$: a calibrated mapping of the GigaPose score;
- $w_{\mathrm{visual}}$: 1 for manually approved, smaller otherwise.

This weight is a pseudo-label confidence proxy. It must not be interpreted as
the probability that the EPnP pose is correct.

## Training type 1: direct 6D pose prediction

Inputs can include:

- object crop or full RGB image;
- instance mask;
- camera intrinsics;
- optional depth.

Target:

$$
T^*=T_{\mathrm{epnp\_obj}}.
$$

Recommended losses:

- robust log-depth loss;
- image-center reprojection loss;
- camera-frame translation loss;
- SO(3) geodesic rotation loss using `atan2`;
- ADD or ADD-S point-cloud loss;
- optional silhouette/depth rendering loss.

Use sample weights or quality tiers. Keep an independently verified test set.

## Training type 2: detector or instance segmenter

Pose extrinsics are not required if the model predicts only:

- class;
- bbox;
- instance mask.

Use the GSAM/detection bbox and mask after confirming the selected pose belongs
to that same car. Do not derive a mask ID directly from the GigaPose
`instance_id` without checking the mapping.

The extrinsics-aware selection can still help reject obviously mismatched cars,
but pose agreement should not replace 2D annotation quality checks.

## Training type 3: keypoints, dense correspondences, or PnP

Use the centered CAD to define object-coordinate points $P_{\mathrm{obj}}$.
Generate image targets with:

$$
\tilde p
=
K
\left(
R^*P_{\mathrm{obj}}+t^*
\right),
$$

followed by perspective division.

Only supervise:

- projected points with positive depth;
- points inside the image;
- visible points according to depth/rendering;
- pixels inside the valid and object masks.

For a dense coordinate model, every supervised object-coordinate target must
use the same centered CAD origin.

## Training type 4: template retrieval or contrastive learning

Use the corrected EPnP rotation to select or render the positive template.

Positive pairs:

- observed crop;
- centered-CAD template at `T_epnp_obj`;
- same object and compatible viewpoint.

Negatives can include:

- other viewpoints;
- explicit 180-degree front/rear alternatives;
- other cars;
- high-scoring but rejected GigaPose hypotheses.

Do not use the selected GigaPose template as the sole definition of the
positive. That would teach the new model to repeat the selector model's error.

## Training type 5: pose residual or iterative refinement

Choose one initial-pose convention.

### Raw-GigaPose correction

$$
T_0=T_{\mathrm{gigapose\_cam\_obj}},
\qquad
T^*=T_{\mathrm{epnp\_obj}}.
$$

A left camera-frame residual is:

$$
\Delta T_{\mathrm{left}}
=
T^*T_0^{-1},
\qquad
T_{\mathrm{pred}}
=
\Delta T_{\mathrm{left}}T_0.
$$

This is appropriate when deployed inference receives raw GigaPose outputs.

### Aligned-GigaPose correction

$$
T_0=T_{\mathrm{gigapose\_aligned\_epnp\_obj}},
\qquad
T^*=T_{\mathrm{epnp\_obj}}.
$$

Use this only when deployment also applies the same empirical right transform.

The right transform may contain prediction bias, not only a physical
object-frame conversion. Never apply it to the CAD or EPnP target merely
because it improves numerical agreement.

Train from both small and deliberately large perturbations so the refiner can
recover instead of assuming a nearly correct initialization.

## Training type 6: confidence or hypothesis ranking

For each image/instance, construct candidates from:

- raw GigaPose top-K;
- aligned top-K if that is the deployed convention;
- EPnP/PnP alternatives;
- 180-degree flips;
- depth/translation perturbations;
- temporally propagated poses.

Calculate candidate error against:

$$
T^*=T_{\mathrm{epnp\_obj}}.
$$

Confidence targets may use thresholds on:

- translation;
- rotation;
- reprojection;
- ADD/ADD-S;
- silhouette and depth consistency.

A ranking loss should prefer the candidate with the lowest target-pose error,
not merely the pose closest to the previous frame.

## Training type 7: tracking or temporal models

Preserve chronological frame order and session identity.

Camera-frame target per frame:

$$
T_{\mathrm{cam,obj},t}^*
=
T_{\mathrm{epnp\_obj},t}.
$$

Map-frame target:

$$
T_{\mathrm{map,obj},t}^*
=
T_{\mathrm{epnp\_map\_obj},t}.
$$

Camera/map consistency:

$$
T_{\mathrm{map,obj},t}^*
\approx
T_{\mathrm{map,cam},t}^{\mathrm{opt}}
T_{\mathrm{cam,obj},t}^*.
$$

Do not impose a static-camera motion model when the ego camera moves. Use the
per-frame map-camera transform.

Useful temporal objectives include:

- pose velocity/acceleration consistency;
- optical-flow correspondence;
- temporal silhouette consistency;
- re-identification across cars;
- confidence-gated global relocalization.

## Training type 8: differentiable rendering or analysis-by-synthesis

Render the centered CAD using `T_epnp_obj` and the frame intrinsics.

Possible objectives:

- silhouette IoU or boundary distance;
- robust metric-depth agreement;
- edge alignment;
- photometric loss when CAD texture is trustworthy;
- image-feature/template consistency.

Mask padded pixels, invalid image borders, missing depth, and occluded
object regions. A rendering mismatch can indicate a bad pseudo-label, incorrect
CAD origin, wrong intrinsics, or imperfect mask; do not automatically assume
the network is responsible.

## Training type 9: teacher-student/domain adaptation

For the intended use, the teacher is an **offline geometric ensemble**, not one
network. It combines:

- the extrinsics-corrected EPnP pose;
- GigaPose top-K pose/template hypotheses and scores;
- CAD silhouette, edge, bbox, and optional depth agreement;
- optional PnP/correspondence/FoundationPose predictions;
- optional temporal/tracking consistency;
- source-specific confidence estimates.

The student is a much lighter model trained from the fused teacher output. At
deployment, only the student is required; the expensive teacher ensemble,
optimized extrinsics, GigaPose, and CAD hypothesis search can remain offline.

### 9.1 Put every teacher source in one convention

Every teacher candidate must become a centered-object OpenCV
camera-from-object pose in millimetres:

$$
T_k \in SE(3),
\qquad
T_k = T_{\mathrm{cam,obj},k}^{\mathrm{centered}}.
$$

Use:

- corrected EPnP candidate:
  `T_epnp_obj`;
- GigaPose comparison candidate:
  `T_gigapose_aligned_epnp_obj`;
- raw GigaPose for provenance and ablations:
  `T_gigapose_cam_obj`;
- other estimators: explicitly convert their camera axes, units, CAD origin,
  and pose direction before adding them.

`T_epnp_obj` already contains the optimized-extrinsics correction. Do not
correct it again. The empirical right transform used to obtain
`T_gigapose_aligned_epnp_obj` is allowed as a GigaPose-candidate conversion,
but it must not be applied to the EPnP target or CAD.

The optimized extrinsics do not need to be an input to a camera-frame student.
They are used offline to construct the corrected EPnP teacher candidate. A
map-frame student may additionally receive `T_map_cam_optimized`.

### 9.2 Do not average all predictions blindly

Teacher sources can be multimodal. A front/rear-flipped GigaPose candidate can
be far from EPnP while still having a high raw score. Directly averaging
translations and rotations across incompatible modes can create a pose that no
teacher predicted and that does not align with the image.

First cluster or gate candidates using a pose distance:

$$
d(T_i,T_j)
=
\frac{\lVert t_i-t_j\rVert_2}{\tau_t}
+
\frac{d_{SO(3)}(R_i,R_j)}{\tau_R}.
$$

A robust teacher medoid is:

$$
j^*
=
\underset{j}{\arg\min}
\sum_k w_k\,d(T_j,T_k).
$$

Keep the consensus cluster around $T_{j^*}$. Reject incompatible
front/rear, depth, or instance-association modes before continuous fusion.

Within the winning cluster:

- fuse translation with a weighted robust mean/Huber estimator;
- fuse rotation with an SO(3) Karcher/geodesic mean;
- or use the medoid pose directly when only a few sources agree.

This produces:

$$
T_{\mathrm{teacher}}
=
\operatorname{RobustFuse}
\left(
\{T_k,w_k\}_{k\in\mathcal C^*}
\right).
$$

### 9.3 Score teacher candidates independently

Do not use raw confidence numbers from different models as though they had the
same calibration. For each candidate, calculate a source-calibrated weight
from:

- model confidence;
- CAD silhouette IoU;
- projected bbox/center agreement;
- edge alignment;
- metric-depth agreement when depth is valid;
- agreement with other independent sources;
- temporal consistency;
- visibility/occlusion;
- whether the candidate belongs to the correct detected car.

For example:

$$
w_k
=
w_{\mathrm{source},k}
w_{\mathrm{image},k}
w_{\mathrm{consensus},k}
w_{\mathrm{visibility},k}.
$$

The GigaPose score is one feature in
$w_{\mathrm{source},k}$, not the complete teacher confidence.

Corrected EPnP is the main geometric anchor, but it should still lose weight
when its CAD projection, mask, depth, or temporal behavior is poor. Likewise,
a GigaPose candidate may receive high weight when its image evidence is much
better and independent sources support it.

### 9.4 Save a teacher distribution, not only one pose

For each training instance, the offline teacher cache should contain:

- `T_teacher`: fused/selected pose;
- `teacher_confidence`;
- translation and rotation dispersion of the winning cluster;
- all source candidates and source confidences;
- winning source/cluster membership;
- corrected EPnP and raw/aligned GigaPose poses;
- silhouette/depth/reprojection evidence;
- front/rear ambiguity flag;
- source dataset, camera, session, image, bbox, mask, and intrinsics;
- CAD version and centered-CAD offset.

Retaining the candidate distribution allows a future student to learn
uncertainty and ambiguity instead of treating every pseudo-label as exact.

### 9.5 Teacher confidence and rejection

Define teacher uncertainty from both candidate dispersion and image evidence.
A conceptual confidence is:

$$
q_{\mathrm{teacher}}
=
\exp
\left(
-\frac{\sigma_t}{\tau_t}
-\frac{\sigma_R}{\tau_R}
\right)
q_{\mathrm{image}}
q_{\mathrm{visibility}}.
$$

Use hard rejection when:

- no pose has acceptable projection/mask alignment;
- every source disagrees;
- front/rear modes remain equally plausible;
- corrected depth is non-positive or implausible;
- the pose projects to the wrong detected car;
- camera/CAD/unit conventions cannot be verified.

A rejected sample is better than a high-confidence wrong teacher target.

### 9.6 Lightweight student outputs

A practical small student can predict:

- 2D object center;
- log depth;
- rotation representation;
- optional full camera-frame translation;
- pose confidence/uncertainty;
- optional instance mask or bbox.

The student may use a small CNN or compact ViT backbone and a direct pose head.
It does not need GigaPose templates, EPnP, optimized extrinsics, or CAD
rendering during deployment.

### 9.7 Student loss

For a fused pose target:

$$
\mathcal L_{\mathrm{pose}}
=
\lambda_t\mathcal L_t
+
\lambda_z\mathcal L_{\log z}
+
\lambda_R\mathcal L_{SO(3)}
+
\lambda_p\mathcal L_{\mathrm{reproj}}
+
\lambda_{\mathrm{ADD}}\mathcal L_{\mathrm{ADD/S}}.
$$

Weight pseudo-label supervision by teacher confidence:

$$
\mathcal L_{\mathrm{weighted\ pose}}
=
q_{\mathrm{teacher}}\,
\mathcal L_{\mathrm{pose}}.
$$

If the teacher preserves a distribution over top-K hypotheses, distill it:

$$
\mathcal L_{\mathrm{KD}}
=
\operatorname{KL}
\left(
p_{\mathrm{teacher}}(T\mid I)
\;\Vert\;
p_{\mathrm{student}}(T\mid I)
\right).
$$

Train the student confidence against teacher validity/uncertainty:

$$
\mathcal L_{\mathrm{confidence}}
=
\operatorname{BCE}
\left(
\hat q_{\mathrm{student}},
q_{\mathrm{teacher}}
\right).
$$

The complete objective can be:

$$
\mathcal L_{\mathrm{student}}
=
q_{\mathrm{teacher}}\mathcal L_{\mathrm{pose}}
+
\lambda_{\mathrm{KD}}\mathcal L_{\mathrm{KD}}
+
\lambda_c\mathcal L_{\mathrm{confidence}}
+
\lambda_m\mathcal L_{\mathrm{mask}}
.
$$

Do not force a single-pose metric loss on rejected or genuinely ambiguous
examples. They can still be used for representation consistency, detection,
segmentation, or soft top-K distillation.

### 9.8 Recommended teacher-student curriculum

1. Build the teacher cache offline from corrected EPnP, GigaPose top-K, CAD
   evidence, and the other available estimators.
2. Manually verify a stratified subset and calibrate source confidences.
3. Train first on manually approved/Tier A high-confidence teacher samples.
4. Add Tier B using teacher-confidence weighting and soft top-K distillation.
5. Use Tier C primarily for detection, representation consistency, ranking,
   and low-weight pose supervision.
6. Measure calibration: higher teacher confidence should correspond to lower
   error on an independently verified set.
7. Optionally let the trained student contribute a new teacher candidate, but
   do not allow it to reinforce its own low-confidence errors.
8. Keep a test set that was not selected or fused using the GigaPose model
   being compared.

### 9.9 Prevent teacher leakage

The current samples were selected partly because GigaPose and corrected EPnP
agreed. Consequently:

- the training distribution is biased toward cases GigaPose already handles;
- evaluating on the same selected rows is not independent;
- a student may appear to match the teacher without improving physical pose
  accuracy.

Split by complete recording session before teacher fusion where possible, and
evaluate the lightweight student on manually verified or otherwise independent
sequences. Avoid selecting and evaluating with the same GigaPose model on the
same frames and reporting that result as independent generalization.

## Train/validation/test splitting

These images are sequential. Random per-frame splitting will leak nearly
identical neighboring frames.

Split by:

- recording session;
- lap or long contiguous time range;
- camera where cross-camera generalization is being tested;
- weather/lighting condition where possible.

Do not let adjacent frames from one sequence appear in both training and
validation.

Because GigaPose predictions were used to select the samples, a validation set
built from those same selections measures performance on GigaPose-agreeing
examples. Maintain a separate benchmark that was not selected using the model
being evaluated.

## Required validation before a long training run

### Matrix checks

For every target:

- shape is 4x4;
- final row is `[0, 0, 0, 1]`;
- all values are finite;
- rotation determinant is approximately 1;
- translation $z>0$ in the camera frame.

### Alias check

Confirm:

$$
T_{\mathrm{epnp\_obj}}
\approx
T_{\mathrm{epnp\_camera\_obj\_optimized}}.
$$

### Map-camera check

Confirm:

$$
T_{\mathrm{epnp\_map\_obj}}
\approx
T_{\mathrm{map\_cam\_optimized}}
T_{\mathrm{epnp\_obj}}.
$$

### Projection check

Project:

- centered CAD origin;
- centered CAD bbox corners;
- a sampled CAD point cloud.

The projection should agree with the selected car's image/mask.

### Visual check

For every camera/session:

- inspect strict, median, and near-threshold samples;
- inspect near/far objects;
- inspect front/rear appearances;
- inspect occlusions;
- confirm the detection bbox belongs to the same car;
- compare corrected EPnP and aligned GigaPose separately.

### Distribution check

Report by camera/session:

- sample count;
- translation/depth distribution;
- yaw/pitch/roll distribution;
- bbox area;
- object distance;
- GigaPose score;
- agreement-error quantiles;
- accepted/rejected ratio.

## Exporting to common formats

### BOP/MegaPose-style pose training

For each selected object:

- `obj_id` from the selected row;
- `cam_R_m2c` from `T_epnp_obj[:3, :3]`;
- `cam_t_m2c` from `T_epnp_obj[:3, 3]` in millimetres;
- camera `cam_K`;
- bbox and instance mask associated with the same car;
- centered object CAD.

### Generic pose CSV

At minimum preserve:

- unique source/session ID;
- image path or key;
- object/instance identity;
- flattened target rotation;
- target translation and explicit unit;
- intrinsics;
- mask/bbox reference;
- sample quality tier/weight;
- provenance paths.

### COCO detection/segmentation

Export only:

- image;
- category;
- bbox;
- segmentation;
- unique annotation ID.

Keep pose/provenance in extra fields if desired, but COCO consumers will
usually ignore them.

## Camera-specific extrinsics

Use the optimized extrinsics matching the camera:

- front images: front optimized extrinsics;
- rear images: rear optimized extrinsics;
- stereo-left images: stereo-left optimized extrinsics.

Do not share corrections across cameras.

For a different physical recording rig or changed camera mount:

1. begin with the existing calibration only as a prior;
2. render corrected labels on the new session;
3. verify systematic image-center and pose errors;
4. recalibrate if the mounting changed.

## Recommended practical workflow

1. Collect every `selected_samples.csv` with its source dataset and camera.
2. Add a unique session/dataset identifier before combining.
3. Parse the 4x4 matrices as millimetre transforms.
4. Validate all matrix and map-camera identities.
5. Render corrected EPnP using the centered CAD.
6. Resolve the correct detection/mask for each selected pose.
7. Assign Tier A/B/C or a continuous sample weight.
8. Split by complete sequence/session.
9. Build the target format required by the new architecture.
10. Run a small overfit/smoke experiment on manually verified Tier A samples.
11. Confirm that translation, rotation, projection, and overlay metrics improve.
12. Expand to Tier B and then selectively to Tier C.
13. Evaluate on an untouched independently verified test set.

## Final recommendation

The new extrinsics-aware selected labels are appropriate for new training when
used as **corrected, quality-controlled pseudo-labels**.

For camera-frame pose supervision:

$$
\boxed{T^*=T_{\mathrm{epnp\_obj}}}
$$

with:

- translation in millimetres;
- the centered CAD;
- no second extrinsic correction;
- no empirical right transform applied to the target or CAD;
- camera/session-aware sample identity;
- sequence-level data splitting;
- quality weighting and visual verification.

The raw and aligned GigaPose poses should be preserved as model outputs,
initializations, teachers, or competing hypotheses—not substituted for the
corrected EPnP target.
