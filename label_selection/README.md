# Absolute GigaPose–EPnP label selection

This folder is a new, isolated label-selection pipeline. It does **not** modify
or call the older empirical GigaPose-to-EPnP alignment workflow.

The purpose is:

1. load GigaPose predictions;
2. load EPnP ground-truth labels;
3. compare them in the same, known object convention;
4. visualize native GigaPose, known-center-converted GigaPose, and EPnP;
5. select one-to-one prediction/label pairs using absolute pose error;
6. combine selected rows separately for each camera;
7. optimize one fixed camera–LiDAR calibration for that camera;
8. reconstruct corrected EPnP camera targets with that calibration and reselect
   final pairs for any downstream training system.

## The coordinate convention

All translations written by this package are in millimetres. The notation
$T_{A\leftarrow B}$ means “the rigid transform that maps coordinates from
frame $B$ into frame $A$.”

GigaPose predicts a camera pose for the native/raw CAD frame:

$$
G_i^{r}
=
T_{\mathrm{camera}\leftarrow\mathrm{object\_raw},\,i}^{\mathrm{GigaPose}}.
$$

EPnP provides the centered camera pose:

$$
E_i^{c}
=
T_{\mathrm{camera}\leftarrow\mathrm{object\_centered},\,i}^{\mathrm{EPnP}}.
$$

The known raw-CAD point used as the centered origin is

$$
c_r =
\begin{bmatrix}
-241.1941141 &
\phantom{-}0.9010172 &
\phantom{-}332.9219520
\end{bmatrix}^{\!\top}
\mathrm{mm}.
$$

Define

$$
C
=
T_{\mathrm{object\_raw}\leftarrow\mathrm{object\_centered}}
=
\begin{bmatrix}
I & c_r\\
\mathbf{0}^{\top} & 1
\end{bmatrix}.
$$

The GigaPose pose in the centered convention is therefore

$$
G_i^{c}=G_i^{r}C.
$$

Equivalently,

$$
R_i^{c}=R_i^{r},\qquad
t_i^{c}=t_i^{r}+R_i^{r}c_r.
$$

This is a known CAD-origin conversion, not a fitted alignment.

Absolute errors are

$$
e_{t,i}
=
\left\lVert t_i^{c}-t_{E,i}^{c}\right\rVert_2
$$

and

$$
e_{R,i}
=
\operatorname{angle}\!\left(
\left(R_{E,i}^{c}\right)^\top R_i^{c}
\right).
$$

No constant transform is estimated from the predictions and labels before
these errors are measured. Consequently there are intentionally no
`--frame-transform-side` or transform-refinement arguments in this package.

## Output columns that matter

`selected_samples.csv` contains explicit, unambiguous transforms:

- `T_gigapose_native_raw_cad`: original GigaPose CSV pose;
- `T_object_raw_object_centered`: known CAD-center conversion;
- `T_gigapose_known_center_aligned`: original GigaPose pose after only the
  known CAD-center conversion;
- `T_epnp_camera_object_centered_original`: EPnP camera pose read directly from
  `T_camera_object_centered`;
- `T_epnp_map_object_raw`: EPnP map pose read directly from
  `T_map_object_raw`;
- `absolute_translation_error_mm` and `absolute_rotation_error_deg`.

The word “aligned” in `T_gigapose_known_center_aligned` means aligned to the
known centered CAD convention. It never means empirically fitted to EPnP.

After calibration, the target is also stored as
`T_epnp_camera_object_centered_corrected`. A downstream training pipeline
should use:

- image/mask and object identity from the selected row;
- `T_epnp_camera_object_centered_corrected` as the calibrated camera-frame
  target;
- `T_epnp_map_object_raw` if it needs the original map-frame target;
- `T_gigapose_known_center_aligned` only as a teacher prediction or diagnostic,
  not as ground truth.

## Step 1: absolute selection for 20260718 front

```bash
python -m label_selection.select_absolute \
  --gigapose-predictions gigaPose_datasets/results/large_real_20260718_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260718_front_gsam_v4-test_large_real_20260718_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260718_front_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-07-18/front/EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-camera-pose-key T_camera_object_centered \
  --epnp-map-pose-unit m \
  --epnp-camera-pose-unit m \
  --raw-object-center-m -0.2411941141 0.0009010172 0.3329219520 \
  --prediction-translation-unit mm \
  --min-score 0.05 \
  --max-translation-error-mm 4000 \
  --max-rotation-error-deg 30 \
  --max-roll-error-deg 10 \
  --max-pitch-error-deg 10 \
  --max-yaw-error-deg 15 \
  --output-dir gigaPose_datasets/results/label_selection/20260718/front/absolute \
  --overwrite
```

This writes:

- `all_candidate_pairs.csv`: every feasible prediction/label pairing per image;
- `assigned_pairs.csv`: minimum-cost one-to-one Hungarian assignment;
- `selected_samples.csv`: assigned rows passing all absolute thresholds;
- `selection_report.json`: counts, equations, and error summaries.

The assignment cost is only used to decide which car corresponds to which
label when an image has multiple cars:

$$
\mathcal{C}_{ij}
=
\frac{e_{t,ij}}{1000\,\mathrm{mm}}
+
\frac{e_{R,ij}}{10^\circ}.
$$

Selection thresholds are applied after the one-to-one assignment.

## Step 2: visualize raw, known-center, and EPnP poses

```bash
python -m label_selection.visualize \
  --candidate-csv gigaPose_datasets/results/label_selection/20260718/front/absolute/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260718_front_gsam_v4 \
  --output-dir gigaPose_datasets/results/label_selection/20260718/front/absolute/visual_overlays \
  --projection-model metadata \
  --draw-mask-bbox \
  --max-images 100 \
  --overwrite
```

Colors:

- blue: raw CAD rendered with the native GigaPose pose;
- magenta: centered CAD rendered with the known-center-converted GigaPose pose;
- green: EPnP centered target;
- yellow: detector/GSAM box.

Blue and magenta must coincide. The visualizer calculates their maximum pixel
difference and raises an error if the known center conversion changes the
physical projection. Green is not fitted to either of them.

## Step 3: run the same absolute selection for 20260718 rear

```bash
python -m label_selection.select_absolute \
  --gigapose-predictions gigaPose_datasets/results/large_real_20260718_rear_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260718_rear_gsam_v4-test_large_real_20260718_rear_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260718_rear_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-07-18/rear/EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-camera-pose-key T_camera_object_centered \
  --epnp-map-pose-unit m \
  --epnp-camera-pose-unit m \
  --raw-object-center-m -0.2411941141 0.0009010172 0.3329219520 \
  --prediction-translation-unit mm \
  --min-score 0.05 \
  --max-translation-error-mm 4000 \
  --max-rotation-error-deg 30 \
  --max-roll-error-deg 10 \
  --max-pitch-error-deg 10 \
  --max-yaw-error-deg 15 \
  --output-dir gigaPose_datasets/results/label_selection/20260718/rear/absolute \
  --overwrite
```

Visualize rear by changing the candidate, dataset, and output paths in the Step
2 command from `front` to `rear`.

## Step 4: combine selected rows per camera

Never combine front, rear, and stereo-left in the same calibration CSV.

For the single 20260718 front selection:

```bash
python -m label_selection.combine \
  --camera-id front \
  --input gigaPose_datasets/results/label_selection/20260718/front/absolute/selected_samples.csv \
  --output gigaPose_datasets/results/label_selection/combined_front_absolute.csv \
  --overwrite
```

For rear:

```bash
python -m label_selection.combine \
  --camera-id rear \
  --input gigaPose_datasets/results/label_selection/20260718/rear/absolute/selected_samples.csv \
  --output gigaPose_datasets/results/label_selection/combined_rear_absolute.csv \
  --overwrite
```

To combine more dates for the same camera, repeat `--input`:

```bash
python -m label_selection.combine \
  --camera-id front \
  --input /date_1/front/absolute/selected_samples.csv \
  --input /date_2/front/absolute/selected_samples.csv \
  --input /date_3/front/absolute/selected_samples.csv \
  --output gigaPose_datasets/results/label_selection/combined_front_absolute.csv \
  --overwrite
```

The combiner rejects mixed comparison modes and mixed cameras. Duplicate EPnP
records retain the row with the lowest absolute pairing cost.

## Step 5: optimize one fixed camera–LiDAR transform

For each selected sample define

$$
G_i
=
T_{\mathrm{camera}\leftarrow\mathrm{object\_centered},\,i}^{\mathrm{GigaPose}},
\quad
M_i
=
T_{\mathrm{map}\leftarrow\mathrm{object\_raw},\,i}^{\mathrm{EPnP}},
\quad
L_i
=
T_{\mathrm{map}\leftarrow\mathrm{LiDAR},\,i}^{\mathrm{metadata}}.
$$

The centered EPnP target in LiDAR coordinates is

$$
Q_i=L_i^{-1}M_iC.
$$

The optimizer estimates one fixed
$X=T_{\mathrm{LiDAR}\leftarrow\mathrm{camera}}$ for the entire camera:

$$
P_i(X)=XG_i.
$$

It robustly minimizes translation and $\operatorname{SO}(3)$ rotation
residuals between $P_i(X)$ and $Q_i$, with a prior around the robust average
of the metadata `t_lidar_camera_prior` values.

Front:

```bash
python -m label_selection.optimize_extrinsics \
  --selected-samples gigaPose_datasets/results/label_selection/combined_front_absolute.csv \
  --camera-id front \
  --translation-residual-components xyz \
  --translation-sigma-mm 1000 \
  --rotation-sigma-deg 10 \
  --translation-prior-weight 1000 \
  --rotation-prior-weight 20 \
  --robust-loss soft_l1 \
  --output-dir gigaPose_datasets/results/label_selection/calibration_front \
  --overwrite
```

Rear uses `--camera-id rear`,
`combined_rear_absolute.csv`, and `calibration_rear`.

The optimizer saves both directions:

- `T_lidar_camera_optimized`;
- `T_camera_lidar_optimized = inv(T_lidar_camera_optimized)`;
- metre-valued copies with the `_m` suffix;
- `errors_before.csv`, `errors_after.csv`, and
  `per_sample_calibration.csv`;
- `recommended_for_reselection` in `optimized_extrinsics.json`.

`T_map_lidar` is read per selected sample. The optimized camera–LiDAR
extrinsic itself is one shared transform for the complete camera dataset.

For an unbiased accuracy report, calibrate on one set of sessions and apply
the saved calibration to different held-out sessions. The optimizer's
`recommended_for_reselection` flag describes only its input rows; it is not a
held-out performance guarantee.

## Step 6: reselect with the optimized extrinsics

The corrected EPnP camera target is

$$
E_{i,\mathrm{corrected}}^{c}
=
X_{\mathrm{opt}}^{-1}L_i^{-1}M_iC.
$$

Front:

```bash
python -m label_selection.reselect_with_extrinsics \
  --gigapose-predictions gigaPose_datasets/results/large_real_20260718_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260718_front_gsam_v4-test_large_real_20260718_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260718_front_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-07-18/front/EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-camera-pose-key T_camera_object_centered \
  --epnp-map-pose-unit m \
  --epnp-camera-pose-unit m \
  --optimized-extrinsics gigaPose_datasets/results/label_selection/calibration_front/optimized_extrinsics.json \
  --camera-id front \
  --prediction-translation-unit mm \
  --min-score 0.05 \
  --max-translation-error-mm 3000 \
  --max-rotation-error-deg 30 \
  --max-roll-error-deg 10 \
  --max-pitch-error-deg 10 \
  --max-yaw-error-deg 15 \
  --output-dir gigaPose_datasets/results/label_selection/20260718/front/reselected \
  --overwrite
```

Visualize the calibrated rows with the same visualizer:

```bash
python -m label_selection.visualize \
  --candidate-csv gigaPose_datasets/results/label_selection/20260718/front/reselected/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260718_front_gsam_v4 \
  --output-dir gigaPose_datasets/results/label_selection/20260718/front/reselected/visual_overlays \
  --projection-model metadata \
  --draw-mask-bbox \
  --max-images 100 \
  --overwrite
```

For rear, replace the front prediction/dataset/EPnP paths, use
`calibration_rear/optimized_extrinsics.json`, set `--camera-id rear`, and write
to the rear result directory.

## Which CSV should be used for new training?

Use the final `reselected/selected_samples.csv`. It contains only one-to-one
pairs passing the requested calibrated absolute-error bounds. Preserve the
following provenance fields in any converted dataset:

- `comparison_mode`;
- `optimized_extrinsics_path`;
- `epnp_label_path` and `epnp_record_index`;
- `sample_metadata_path`;
- `T_epnp_camera_object_centered_original`;
- `T_epnp_camera_object_centered_corrected`;
- `T_epnp_map_object_raw`;
- all absolute error fields.

Do not call a fitted GigaPose-to-label transform “ground truth,” and do not
estimate a new alignment on the same rows used to report model accuracy.
