# Real-world branch A: without camera anchoring

This branch reproduces the ordinary benchmark-style selector, translation-only
GRU–Kalman cascade, and SciPy sequence graph. It is useful as a baseline. Its
outputs never overlap the camera-anchored branch.

First define the shared variables from
`README_REAL_WORLD_TRANSLATION_CASCADE.md`, then confirm:

```bash
test -f "$CANDIDATES/candidates.npz"
test -f "$GP_CSV"
export BRANCH_RUN="$WITHOUT_ANCHOR_RUN"
mkdir -p "$BRANCH_RUN"
```

## 1. Ordinary GRU selector and fixed-lag orientation

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.select \
  --data "$CANDIDATES" \
  --checkpoint "$SELECTOR_CKPT" \
  --minimum-recovery-probability 0 \
  --minimum-recovery-margin-over-gigapose 0 \
  --temporal-orientation \
  --orientation-soft-start-deg 45 \
  --orientation-hard-limit-deg 90 \
  --flip-min-angle-deg 135 \
  --rotation-penalty-weight 1 \
  --flip-confirmation-frames 2 \
  --stable-orientation-frames 4 \
  --stable-flip-confirmation-frames 5 \
  --flip-min-probability 0.15 \
  --flip-min-margin 0.03 \
  --flip-consistency-deg 45 \
  --maximum-angular-prediction-deg 30 \
  --orientation-velocity-window 5 \
  --orientation-fixed-lag \
  --output-dir "$BRANCH_RUN/selector" \
  --device cuda \
  --overwrite

python3 -m json.tool "$BRANCH_RUN/selector/run_report.json"
```

## 2. Translation-only GRU–Kalman

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.export_measurements \
  --dataset-dir "$DATASET" \
  --split test \
  --predictions "$BRANCH_RUN/selector/tracked_predictions.csv" \
  --prediction-translation-unit mm \
  --sequence-max-frame-gap 1 \
  --sequence-max-time-gap-s 0.5 \
  --output-dir "$BRANCH_RUN/measurements" \
  --overwrite

python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.filter \
  --data "$BRANCH_RUN/measurements" \
  --checkpoint "$TRANSLATION_KF_CKPT" \
  --max-correction-translation-m 100 \
  --max-correction-rotation-deg 180 \
  --output-dir "$BRANCH_RUN/translation_cascade" \
  --device cuda \
  --overwrite
```

The translation-only checkpoint preserves the selector rotation.

## 3. SciPy sequence graph

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.sequence_factor_graph.run \
  --data "$CANDIDATES" \
  --mode prior \
  --prior-predictions "$BRANCH_RUN/translation_cascade/tracked_predictions.csv" \
  --prior-translation-unit mm \
  --output-dir "$BRANCH_RUN/scipy_graph" \
  --outer-iterations 5 \
  --maximum-nfev 100 \
  --overwrite

python3 -m json.tool "$BRANCH_RUN/scipy_graph/run_report.json"

export OPTIMIZED_POSES="$BRANCH_RUN/scipy_graph/tracked_predictions.csv"
```

## 4. Overlays

```bash
mkdir -p "$BRANCH_RUN/overlays"

for item in \
  "gigapose=$GP_CSV" \
  "selector=$BRANCH_RUN/selector/tracked_predictions.csv" \
  "translation_cascade=$BRANCH_RUN/translation_cascade/tracked_predictions.csv" \
  "scipy_graph=$OPTIMIZED_POSES"
do
  name="${item%%=*}"
  predictions="${item#*=}"
  python3 -m fine_tuning.overlay_gigapose_predictions \
    --predictions "$predictions" \
    --dataset-dir "$DATASET" \
    --split test \
    --mesh "$MESH" \
    --translation-scale 0.001 \
    --max-images 200 \
    --output-dir "$BRANCH_RUN/overlays/$name"
done
```

## 5. Iteration-0 EPnP candidate selection

EPnP is introduced here, after graph prediction.

```bash
export EPNP_SELECTION="$BRANCH_RUN/epnp_selection_iteration_0"

python3 -m fine_tuning.select_real_label_candidates \
  --gigapose-predictions "$OPTIMIZED_POSES" \
  --dataset-dir "$DATASET" \
  --epnp-root "$EPNP_ROOT" \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --frame-transform-side right \
  --frame-transform-refine-iterations 5 \
  --frame-transform-inlier-translation-mm 5000 \
  --frame-transform-inlier-rotation-deg 60 \
  --epnp-translation-unit m \
  --min-score 0.05 \
  --max-translation-error-mm 3000 \
  --max-rotation-error-deg 30 \
  --max-roll-error-deg 5 \
  --max-pitch-error-deg 5 \
  --max-yaw-error-deg 15 \
  --output-dir "$EPNP_SELECTION"

python3 -m json.tool "$EPNP_SELECTION/selection_report.json"
```

Generate selection overlays:

```bash
python3 -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv "$EPNP_SELECTION/selected_samples.csv" \
  --dataset-dir "$DATASET" \
  --split test \
  --mesh "$MESH" \
  --output-dir "$EPNP_SELECTION/overlays" \
  --max-images 100 \
  --sort-by translation_error \
  --draw-mask-bbox \
  --bbox-match-mode nearest_projected_center \
  --projection-model metadata \
  --object-center-mode epnp_raw \
  --frame-transform-side right
```

Pool this branch's `selected_samples.csv` files by physical camera as described
in the main README. Do not mix them with camera-anchored selections.

## 6. Iteration-1 selection after pooled calibration

After the pooled camera calibration exists:

```bash
export CAMERA_CAL_ROOT="$PWD/gigaPose_datasets/results/real_world_translation_cascade_scipy/calibration_${CAMERA}/without_anchor"

python3 -m fine_tuning.select_real_label_candidates_with_centered_camera_lidar_extrinsics \
  --gigapose-predictions "$OPTIMIZED_POSES" \
  --dataset-dir "$DATASET" \
  --epnp-root "$EPNP_ROOT" \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --optimized-extrinsics "$CAMERA_CAL_ROOT/extrinsic_iteration_1/optimized_extrinsics.json" \
  --frame-transform-json "$EPNP_SELECTION/frame_transform_gigapose_to_epnp.json" \
  --frame-transform-side right \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-camera-pose-key T_camera_object_centered \
  --epnp-map-pose-unit m \
  --epnp-camera-pose-unit m \
  --min-score 0.05 \
  --max-translation-error-mm 3000 \
  --max-rotation-error-deg 30 \
  --max-roll-error-deg 5 \
  --max-pitch-error-deg 5 \
  --max-yaw-error-deg 15 \
  --output-dir "$BRANCH_RUN/epnp_selection_iteration_1"
```
