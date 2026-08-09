# Real-world branch B: with camera anchoring

This is the recommended branch for the current rear-camera run. It replaces
only the selector stage with camera-facing orientation initialization and
fixed-lag handling. Candidate generation, translation-only GRU–Kalman, and the
SciPy graph remain the same model families as the baseline.

The selector is configured so that:

- EPnP rotation cost is zero;
- EPnP translation cost is zero;
- map-heading cost is zero;
- camera-facing cost is enabled;
- `front` and `stereo_left` expect the opponent to face away;
- `rear` expects the opponent to face toward the camera.

First define the shared variables from
`README_REAL_WORLD_TRANSLATION_CASCADE.md`, then confirm:

```bash
test -f "$CANDIDATES/candidates.npz"
test -f "$GP_CSV"
export BRANCH_RUN="$CAMERA_ANCHOR_RUN"
mkdir -p "$BRANCH_RUN"
```

## 1. Camera-anchored selector — your next command

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.orientation_bootstrap.select_anchored \
  --data "$CANDIDATES" \
  --checkpoint "$SELECTOR_CKPT" \
  --dataset-dir "$DATASET" \
  --mesh "$MESH" \
  --epnp-root "$EPNP_ROOT" \
  --gigapose-predictions "$GP_CSV" \
  --epnp-rotation-weight 0 \
  --epnp-translation-weight 0 \
  --map-heading-weight 0 \
  --camera-facing-weight 50 \
  --camera-facing-rules front:away,stereo_left:away,rear:toward \
  --window-length 5 \
  --flip-axis z \
  --object-forward-axis x \
  --output-dir "$BRANCH_RUN/selector" \
  --device cuda \
  --overwrite

python3 -m json.tool "$BRANCH_RUN/selector/run_report.json"
```

Confirm the report contains:

```text
epnp_pose_cost_enabled: false
map_heading_cost_enabled: false
camera_facing_cost_enabled: true
```

EPnP may still appear in diagnostic counts because it is evaluated after pose
selection. It does not influence the selected rotation or translation here.

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

python3 -m json.tool "$BRANCH_RUN/translation_cascade/run_report.json"
```

The translation-only checkpoint preserves the selector's anchored rotation.

## 2a. Required pre-SciPy orientation audit

Before running the graph, compare both intermediate CSVs with quality-approved
EPnP anchors. This evaluator is read-only and does not replace or modify any
pose:

```bash
export PRE_SCIPY_AUDIT="$BRANCH_RUN/pre_scipy_epnp_audit"

for item in \
  "selector=$BRANCH_RUN/selector/tracked_predictions.csv" \
  "translation_cascade=$BRANCH_RUN/translation_cascade/tracked_predictions.csv"
do
  name="${item%%=*}"
  predictions="${item#*=}"
  python3 -m tracking.rgb_self_recovery.sliding_window.orientation_bootstrap.evaluate_anchors \
    --predictions "$predictions" \
    --dataset-dir "$DATASET" \
    --mesh "$MESH" \
    --epnp-root "$EPNP_ROOT" \
    --translation-unit mm \
    --output-dir "$PRE_SCIPY_AUDIT/$name" \
    --overwrite
done

python3 -m json.tool "$PRE_SCIPY_AUDIT/selector/report.json"
python3 -m json.tool "$PRE_SCIPY_AUDIT/translation_cascade/report.json"
```

Inspect `quality_approved.front_rear_flip_like_count` and
`quality_approved.front_rear_flip_like_fraction`. A frame is called
flip-like when its EPnP rotation difference is at least 135 degrees. Ideally
both counts are zero. The selector and translation-cascade rotation summaries
should be identical because the Kalman checkpoint preserves measurement
rotation. EPnP agreement is diagnostic rather than ground truth; inspect any
flagged frames visually before rejecting the pipeline.

Generate ordinary CAD overlays for the two intermediate outputs:

```bash
for item in \
  "selector=$BRANCH_RUN/selector/tracked_predictions.csv" \
  "translation_cascade=$BRANCH_RUN/translation_cascade/tracked_predictions.csv"
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
    --output-dir "$PRE_SCIPY_AUDIT/overlays/$name"
done
```

## 3. SciPy sequence graph

The 10-degree rotation prior discourages the graph from switching back to an
opposite-polarity RGB/CAD candidate.

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.sequence_factor_graph.run \
  --data "$CANDIDATES" \
  --mode prior \
  --prior-predictions "$BRANCH_RUN/translation_cascade/tracked_predictions.csv" \
  --prior-translation-unit mm \
  --prior-rotation-sigma-deg 10 \
  --output-dir "$BRANCH_RUN/scipy_graph" \
  --outer-iterations 5 \
  --maximum-nfev 100 \
  --overwrite

python3 -m json.tool "$BRANCH_RUN/scipy_graph/run_report.json"

export OPTIMIZED_POSES="$BRANCH_RUN/scipy_graph/tracked_predictions.csv"
```

`success=false` can mean a long segment reached the evaluation cap. Inspect
`solution_accepted`, objective before/after, and
`solution_accepted_fraction`; accepted solutions are still written.

## 4. EPnP diagnostics after prediction

These diagnostics do not alter any poses:

```bash
for item in \
  "selector=$BRANCH_RUN/selector/tracked_predictions.csv" \
  "translation_cascade=$BRANCH_RUN/translation_cascade/tracked_predictions.csv" \
  "scipy_graph=$OPTIMIZED_POSES"
do
  name="${item%%=*}"
  predictions="${item#*=}"
  python3 -m tracking.rgb_self_recovery.sliding_window.orientation_bootstrap.evaluate_anchors \
    --predictions "$predictions" \
    --dataset-dir "$DATASET" \
    --mesh "$MESH" \
    --epnp-root "$EPNP_ROOT" \
    --output-dir "$BRANCH_RUN/anchor_evaluation/$name" \
    --overwrite
done
```

## 5. Overlays

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

for the camera 
```bash
cd /home/anahita/gigapose

export CAMERA="stereo_left"

export DATASET="$PWD/gigaPose_datasets/datasets/real_20260518v1v4_${CAMERA}_gsam_v4"
export EPNP_ROOT="$PWD/gigaPose_datasets/datasets/2026-05-18-v1-v4/${CAMERA}/EPnPv2_gt_mesh_z_hybrid_labels"

export RUN="$PWD/gigaPose_datasets/results/real_world_translation_cascade_scipy/20260518v1v4_${CAMERA}"
export BRANCH_RUN="$RUN/with_camera_anchor"

export OPTIMIZED_POSES="$BRANCH_RUN/scipy_graph/tracked_predictions.csv"
export EPNP_SELECTION="$BRANCH_RUN/epnp_selection_iteration_0"
```

Check the input before continuing:
```bash
test -f "$OPTIMIZED_POSES" || {
  echo "Missing optimized poses: $OPTIMIZED_POSES"
  exit 1
}

echo "Using predictions: $OPTIMIZED_POSES"
```






## 6. Iteration-0 EPnP candidate selection

Only now is EPnP used to select calibration correspondences.

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
  --max-translation-error-mm 5000 \
  --max-rotation-error-deg 30 \
  --max-roll-error-deg 7.5 \
  --max-pitch-error-deg 7.5 \
  --max-yaw-error-deg 20 \
  --output-dir "$EPNP_SELECTION"

python3 -m json.tool "$EPNP_SELECTION/selection_report.json"
```

The selector's legacy argument and CSV columns still say `gigapose`, but the
actual input here is the camera-anchored SciPy graph CSV.

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
in the main README. Do not mix them with unanchored selections.

## 7. Iteration-1 selection after pooled calibration

After the pooled rear-camera calibration exists:

```bash
export CAMERA_CAL_ROOT="$PWD/gigaPose_datasets/results/real_world_translation_cascade_scipy/calibration_${CAMERA}/with_camera_anchor"

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

Do not automatically optimize a second time from iteration-1 selections.
Inspect overlays and held-out behavior first.
