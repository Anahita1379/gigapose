# Translation-cascade + SciPy on real-world ARCL recordings

This guide runs the benchmarked offline pipeline on a real-world camera
sequence and then uses the graph-refined poses together with
`EPnPv2_gt_mesh_z_hybrid_labels` to optimize one fixed camera/LiDAR
calibration.

The stages are:

```text
GigaPose multi-hypothesis CSV + image/mask/CAD evidence
                         |
                         v
              RGB/CAD candidate exporter
                         |
                         v
        causal GRU selector + fixed-lag orientation
                         |
                         v
             translation-only GRU-Kalman
                         |
                         v
       full-sequence SciPy factor graph (soft prior)
                         |
                         v
       graph-refined camera/object poses (all usable frames)
                         |
                         v
 select close EPnP hybrid correspondences in every recording
                         |
                         v
 pool accepted correspondences from the same physical camera
                         |
                         v
        optimize one fixed camera/LiDAR extrinsic (pooled frames)
                         |
                         v
 rerun selection in every recording with the optimized extrinsic
```

The SciPy graph does not use EPnP or ground truth. EPnP is introduced only
after `tracked_predictions.csv` has been generated.

## Pose convention and outputs

Throughout this code, `T_A_B` maps points from frame B into frame A. Metadata
contains `t_lidar_camera_prior`, which is therefore camera to LiDAR:

```text
p_lidar = T_lidar_camera * p_camera
```

The calibration optimizer writes both directions:

- `T_lidar_camera_optimized`: camera to LiDAR;
- `T_camera_lidar_optimized`: LiDAR to camera, the inverse requested by code
  that uses `Tcamera_lidar` naming.

There are two different optimized-pose products:

- `<RUN>/scipy_graph/tracked_predictions.csv` contains the optimized
  camera-frame object pose for every usable sequence frame;
- `<RUN>/extrinsic/optimized_extrinsics.json` contains the optimized fixed
  camera/LiDAR transform. It does not replace the camera-frame pose CSV.

Do not overwrite the EPnP label JSONs or the metadata prior. Treat the new
extrinsic JSON as a versioned calibration candidate until it passes held-out
and cross-recording checks.

## What is available for `2026-05-18-v1-v4`

The raw recording currently has `front`, `rear`, and `stereo_left`. The
prepared WebDataset folders are not currently present, but the raw images,
Grounded-SAM-v4 masks, metadata, and hybrid EPnP labels are present. Old
GigaPose CSVs also exist, but this guide deliberately does not use them. At the
time this guide was written:

| Camera | Images | Grounded-SAM metadata | Hybrid EPnP labels |
|---|---:|---:|---:|
| front | 398 | 353 | 280 |
| rear | 3,126 | 3,126 | 3,535 |
| stereo_left | 286 | 286 | 277 |

Only the intersection of prepared frames, valid masks, GigaPose predictions,
and EPnP labels can reach calibration. Different counts are expected.

## 1. Choose one recording and one camera

Run from the repository root in the `gigapose` environment:

```bash
cd /home/anahita/gigapose
conda activate gigapose

export RECORDING_ID="20260518v1v4"
export RAW_RECORDING="$PWD/gigaPose_datasets/datasets/2026-05-18-v1-v4"
export CAMERA="front"

export DATASET_NAME="real_${RECORDING_ID}_${CAMERA}_gsam_v4"
export DATASET="$PWD/gigaPose_datasets/datasets/$DATASET_NAME"
export RAW_CAMERA="$RAW_RECORDING/$CAMERA"
export EPNP_ROOT="$RAW_CAMERA/EPnPv2_gt_mesh_z_hybrid_labels"

export GIGAPOSE_CKPT="$PWD/gigaPose_datasets/results/assettocorsa_ot2block_pose_aware_ist_translation_residual/checkpoints/best-residual-step010000.ckpt"
export GIGAPOSE_RUN="gigapose_real_${RECORDING_ID}_${CAMERA}_translation_residual_rerun"
export GIGAPOSE_RESULT="$PWD/gigaPose_datasets/results/$GIGAPOSE_RUN"
export GP_CSV="$GIGAPOSE_RESULT/predictions/large-pbrreal-rgb-mmodel_${DATASET_NAME}-test_${GIGAPOSE_RUN}MultiHypothesis.csv"

export CANDIDATE_CKPT="$PWD/gigaPose_datasets/results/rgb_self_recovery_backbone_comparison/dino_matching_unet/best_pose.ckpt"
export SELECTOR_CKPT="$PWD/gigaPose_datasets/results/rgb_self_recovery_gru_dino_matching/models/dino_matching_gru/best.ckpt"
export TRANSLATION_KF_CKPT="$PWD/gigaPose_datasets/results/rgb_self_recovery_fixed_lag_kalman/model_translation_only/best.ckpt"

export RUN="$PWD/gigaPose_datasets/results/real_world_translation_cascade_scipy/${RECORDING_ID}_${CAMERA}"
mkdir -p "$RUN"
```

For another available camera, change only `CAMERA` to `rear` or
`stereo_left`; the May 18 naming pattern above resolves the corresponding
GigaPose CSV automatically.

For another recording folder, set `RAW_RECORDING` and `RECORDING_ID`. The new
run and CSV names are then derived from those variables. Do not list a
recording that is not mounted or copied locally.

## 2. Recreate the prepared inference dataset if it is missing

The cascade cannot consume the raw camera directory directly because its
loader needs `frame_map.json`, masks/detections, intrinsics, and WebDataset
shards. Recreate only the missing prepared folder:

```bash
if [ ! -f "$DATASET/frame_map.json" ]; then
  python3 -m Assetto_data_prep.prepare_grounded_sam_inference \
    --source-root "$RAW_CAMERA" \
    --grounded-sam-dir Grounded_Sam_v4 \
    --cad-path "$PWD/gigaPose_datasets/datasets/racecar/models/obj_000001.ply" \
    --dataset-name "$DATASET_NAME" \
    --overwrite
fi

export MESH="$DATASET/models/obj_000001.ply"
```

This preparation step does not render the GigaPose CAD template bank. Every
new dataset name needs a corresponding directory under
`gigaPose_datasets/datasets/templates/`, even when it uses the same physical
CAD as another dataset. Render it before inference:

```bash
export TEMPLATE_DIR="$PWD/gigaPose_datasets/datasets/templates/$DATASET_NAME"

if [ ! -f "$TEMPLATE_DIR/000001/000000.png" ]; then
  python3 -m src.scripts.render_custom_templates \
    custom_dataset_name="$DATASET_NAME" \
    machine.num_workers=1
fi
```

For one object this should create 162 RGB/RGBA template images and 162 depth
images, or 324 PNG files total, plus `object_poses/000001.npy`. Check the
prepared data, templates, and model checkpoints before continuing:

```bash
test -f "$DATASET/frame_map.json"
test -f "$DATASET/test/key_to_shard.json"
test -f "$MESH"
test -f "$TEMPLATE_DIR/000001/000000.png"
test -f "$TEMPLATE_DIR/object_poses/000001.npy"
test "$(find "$TEMPLATE_DIR/000001" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 324
test -f "$GIGAPOSE_CKPT"
test -f "$CANDIDATE_CKPT"
test -f "$SELECTOR_CKPT"
test -f "$TRANSLATION_KF_CKPT"
test -d "$EPNP_ROOT"
```

Run the dedicated real-world dataset validator before GigaPose:

```bash
python3 -m Assetto_data_prep.validate_grounded_sam_inference \
  --dataset-dir "$DATASET" \
  --split test \
  --minimum-mask-pixels 25 \
  --maximum-decoded-frames 512 \
  --require-templates \
  --output "$DATASET/validation_report.json"

python3 -m json.tool "$DATASET/validation_report.json"
```

This validator is intentionally different from
`gru_selector.validate_assetto_dataset`: the Assetto validator requires
synthetic ground truth and is not valid for this real-world inference set. The
real-world validator checks every frame/target/detection/shard key, every
source image/mask/metadata path, single-car instance counts, sequence ordering,
and template completeness. It decodes an evenly distributed sample of up to
512 RGB/mask/intrinsics records. Pass `--maximum-decoded-frames 0` to decode
and validate every frame, which is slower for the rear camera.

Do not continue unless the report ends with `"valid": true`.

## 3. Rerun GigaPose with the benchmark checkpoint

Use the same residual GigaPose checkpoint that produced the regenerated-pose
benchmark inputs:

```text
assettocorsa_ot2block_pose_aware_ist_translation_residual/
  checkpoints/best-residual-step010000.ckpt
```

This is the Assetto-trained OT-2-block, pose-aware IST translation-residual
model. It is not the generic pretrained `gigaPose_v1.ckpt`. Using it keeps the
real-world experiment compatible with the input distribution used to train
and evaluate the downstream RGB/CAD selector and translation cascade.

Run GigaPose under a new result name:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.run_gigapose \
  --dataset-name "$DATASET_NAME" \
  --datasets-root "$PWD/gigaPose_datasets/datasets" \
  --checkpoint "$GIGAPOSE_CKPT" \
  --run-name "$GIGAPOSE_RUN" \
  --batch-size 16 \
  --num-workers 2 \
  --devices 0

python3 -m json.tool "$GIGAPOSE_RESULT/gru_gigapose_manifest.json"
test -f "$GP_CSV"
```

The manifest records the exact checkpoint, command, result directory, and
resolved prediction CSV. From this point onward, every `--predictions` or
GigaPose-reference argument in this guide uses this newly generated `$GP_CSV`.

If GPU memory is insufficient, reduce `--batch-size`; this does not change the
checkpoint or inference model. A failed inference run can be restarted with
the same run name: the residual inference entry point deletes partial `.npz`
prediction shards before testing again.

## 4. Export deployment candidates without ground truth

This exporter is located in the GTSAM package because it was first added for
deployment there, but its output is the generic `candidates.npz` consumed by
the SciPy graph. It neither runs GTSAM nor reads ground truth.

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.sequence_factor_graph_gtsam.export_runtime_candidates \
  --dataset-dir "$DATASET" \
  --split test \
  --predictions "$GP_CSV" \
  --prediction-translation-unit mm \
  --checkpoint "$CANDIDATE_CKPT" \
  --mesh "$MESH" \
  --saved-candidates 16 \
  --top-k-gigapose 5 \
  --max-candidates 48 \
  --refinement-iterations 2 \
  --sequence-max-frame-gap 1 \
  --sequence-max-time-gap-s 0.5 \
  --output-dir "$RUN/candidates" \
  --device cuda \
  --overwrite

python3 -m json.tool "$RUN/candidates/manifest.json"
```

Inspect `frame_count`, `segment_count`, `skipped_count`, and every skipped
reason. Real frames are not required to be contiguous; missing predictions
create independent segments and the graph never smooths across them.

## 5. Run the GRU selector and fixed-lag orientation

These are the settings used for the regenerated-pose benchmark translation
cascade:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.select \
  --data "$RUN/candidates" \
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
  --output-dir "$RUN/selector_fixed_lag" \
  --device cuda \
  --overwrite

python3 -m json.tool "$RUN/selector_fixed_lag/run_report.json"


export ANCHORED_SELECTOR="$RUN/selector_epnp_map_anchored"

python3 -m tracking.rgb_self_recovery.sliding_window.orientation_bootstrap.select_anchored \
  --data "$RUN/candidates" \
  --checkpoint "$SELECTOR_CKPT" \
  --dataset-dir "$DATASET" \
  --mesh "$MESH" \
  --epnp-root "$EPNP_ROOT" \
  --gigapose-predictions "$GP_CSV" \
  --window-length 5 \
  --flip-axis z \
  --object-forward-axis x \
  --output-dir "$ANCHORED_SELECTOR" \
  --device cuda \
  --overwrite


```

## 6. Run the translation-only GRU-Kalman stage

First convert the selector CSV into the sequence measurement format:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.export_measurements \
  --dataset-dir "$DATASET" \
  --split test \
  --predictions "$RUN/selector_fixed_lag/tracked_predictions.csv" \
  --prediction-translation-unit mm \
  --sequence-max-frame-gap 1 \
  --sequence-max-time-gap-s 0.5 \
  --output-dir "$RUN/selector_measurements" \
  --overwrite

python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.filter \
  --data "$RUN/selector_measurements" \
  --checkpoint "$TRANSLATION_KF_CKPT" \
  --max-correction-translation-m 100 \
  --max-correction-rotation-deg 180 \
  --output-dir "$RUN/translation_cascade" \
  --device cuda \
  --overwrite

python3 -m json.tool "$RUN/translation_cascade/run_report.json"

---------------------------------------
---------------------------------------

export ANCHORED_MEASUREMENTS="$RUN/selector_epnp_map_anchored_measurements"
export ANCHORED_CASCADE="$RUN/translation_cascade_epnp_map_anchored"

python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.export_measurements \
  --dataset-dir "$DATASET" \
  --split test \
  --predictions "$ANCHORED_SELECTOR/tracked_predictions.csv" \
  --prediction-translation-unit mm \
  --sequence-max-frame-gap 1 \
  --sequence-max-time-gap-s 0.5 \
  --output-dir "$ANCHORED_MEASUREMENTS" \
  --overwrite

python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.filter \
  --data "$ANCHORED_MEASUREMENTS" \
  --checkpoint "$TRANSLATION_KF_CKPT" \
  --max-correction-translation-m 100 \
  --max-correction-rotation-deg 180 \
  --output-dir "$ANCHORED_CASCADE" \
  --device cuda \
  --overwrite


```

The checkpoint has `preserve_measurement_rotation=true`: it filters only
translation and copies the selector/fixed-lag rotation.

## 7. Optimize each complete segment with the SciPy graph

The translation cascade is the soft trajectory prior; all saved RGB/CAD
candidates remain available to the graph. Five outer assignment/optimization
rounds match the stronger benchmark experiment:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.sequence_factor_graph.run \
  --data "$RUN/candidates" \
  --mode prior \
  --prior-predictions "$RUN/translation_cascade/tracked_predictions.csv" \
  --prior-translation-unit mm \
  --output-dir "$RUN/scipy_graph" \
  --outer-iterations 5 \
  --maximum-nfev 100 \
  --overwrite

python3 -m json.tool "$RUN/scipy_graph/run_report.json"

--------------------------------------------
-------------------------------------------

export ANCHORED_GRAPH="$RUN/scipy_graph_epnp_map_anchored"

python3 -m tracking.rgb_self_recovery.sliding_window.sequence_factor_graph.run \
  --data "$RUN/candidates" \
  --mode prior \
  --prior-predictions "$ANCHORED_CASCADE/tracked_predictions.csv" \
  --prior-translation-unit mm \
  --prior-rotation-sigma-deg 10 \
  --output-dir "$ANCHORED_GRAPH" \
  --outer-iterations 5 \
  --maximum-nfev 100 \
  --overwrite


```

`success=False` for a long segment commonly means SciPy reached the evaluation
cap, not that its output was discarded. Check `solution_accepted`, objective
before/after, and `solution_accepted_fraction` in the report. The optimized
camera/object poses are now:



<!-- # potential orientation correction: 
```bash
cd /home/anahita/gigapose

export RUN="$PWD/gigaPose_datasets/results/real_world_translation_cascade_scipy/20260518v1v4_front"
export DATASET="$PWD/gigaPose_datasets/datasets/real_20260518v1v4_front_gsam_v4"
export CANDIDATES="$RUN/candidates"
export EPNP_ROOT="$PWD/gigaPose_datasets/datasets/2026-05-18-v1-v4/front/EPnPv2_gt_mesh_z_hybrid_labels"

export GP_CSV="$PWD/gigaPose_datasets/results/gigapose_real_20260518v1v4_front_translation_residual_rerun/predictions/large-pbrreal-rgb-mmodel_real_20260518v1v4_front_gsam_v4-test_gigapose_real_20260518v1v4_front_translation_residual_rerunMultiHypothesis.csv"

export OPTIMIZED_POSES="$RUN/scipy_graph/tracked_predictions.csv"
export ORIENTATION_RUN="$RUN/scipy_graph_orientation_bootstrapped"

python3 -m tracking.rgb_self_recovery.sliding_window.orientation_bootstrap.run \
  --predictions "$OPTIMIZED_POSES" \
  --candidates "$CANDIDATES" \
  --dataset-dir "$DATASET" \
  --epnp-root "$EPNP_ROOT" \
  --gigapose-predictions "$GP_CSV" \
  --window-length 5 \
  --flip-axis z \
  --object-forward-axis x \
  --output-dir "$ORIENTATION_RUN" \
  --overwrite
``` -->





```bash
export OPTIMIZED_POSES="$RUN/scipy_graph/tracked_predictions.csv"
```







## 8. Compare trajectory changes without ground truth

This reports changes and temporal behavior relative to GigaPose; it must not
be interpreted as real-world pose accuracy:

```bash
python3 -m tracking.compare_predictions_without_gt \
  --model gigapose="$GP_CSV" \
  --model selector="$RUN/selector_fixed_lag/tracked_predictions.csv" \
  --model translation_cascade="$RUN/translation_cascade/tracked_predictions.csv" \
  --model translation_cascade_scipy="$OPTIMIZED_POSES" \
  --reference gigapose \
  --dataset-dir "$DATASET" \
  --split test \
  --translation-unit mm \
  --camera "$CAMERA" \
  --output-dir "$RUN/no_gt_comparison"




  export MESH="$DATASET/models/obj_000001.ply"

mkdir -p "$RUN/no_gt_overlays"

for item in \
  "gigapose=$GP_CSV" \
  "selector=$RUN/selector_fixed_lag/tracked_predictions.csv" \
  "translation_cascade=$RUN/translation_cascade/tracked_predictions.csv" \
  "translation_cascade_scipy=$OPTIMIZED_POSES"
do
  name="${item%%=*}"
  predictions="${item#*=}"

  python3 -m fine_tuning.overlay_gigapose_predictions \
    --predictions "$predictions" \
    --dataset-dir "$DATASET" \
    --split test \
    --mesh "$MESH" \
    --translation-scale 0.001 \
    --max-images 100 \
    --output-dir "$RUN/no_gt_overlays/$name"
done
```


python3 -m tracking.rgb_self_recovery.sliding_window.orientation_bootstrap.evaluate_anchors \
  --predictions "$ANCHORED_GRAPH/tracked_predictions.csv" \
  --dataset-dir "$DATASET" \
  --mesh "$MESH" \
  --epnp-root "$EPNP_ROOT" \
  --output-dir "$RUN/anchor_evaluation_graph" \
  --overwrite


Accuracy evidence comes from the independent EPnP/map support in the next
stages, not from this comparison.

## 9. Select graph poses close to the hybrid EPnP labels

Use the graph output, not the old GigaPose or old tracker CSV:

```bash
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
  --output-dir "$RUN/epnp_selection"

python3 -m json.tool "$RUN/epnp_selection/selection_report.json"
```

The key output is `$RUN/epnp_selection/selected_samples.csv`. The right-side
frame transform is an object/CAD convention alignment:

```text
T_camera_object_aligned = T_camera_object_graph * X_object_frame
```

It is not the camera/LiDAR extrinsic.

Generate selection overlays before calibration:

```bash
python3 -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv "$RUN/epnp_selection/selected_samples.csv" \
  --dataset-dir "$DATASET" \
  --split test \
  --mesh "$MESH" \
  --output-dir "$RUN/epnp_selection/overlays" \
  --max-images 100 \
  --sort-by translation_error \
  --draw-mask-bbox \
  --bbox-match-mode nearest_projected_center \
  --projection-model metadata \
  --object-center-mode epnp_raw \
  --frame-transform-side right
```

Visually reject systematic object-frame, front/rear, mask, or timestamp
mismatches before calibrating.

## 10. Pool accepted candidates across recordings of the same camera

Do not optimize the calibration from the current recording immediately when
other recordings from the same physical camera are available. First run
Sections 1--9 independently for every available `(recording, camera)` pair.
This preserves sequence boundaries and produces one pre-calibration
`selected_samples.csv` per recording.

Then pool only rows belonging to the same unchanged physical camera/rig. For
example, combine all front-camera selections into one front calibration set:

```bash
export CAMERA_CAL_ROOT="$PWD/gigaPose_datasets/results/real_world_translation_cascade_scipy/calibration_front"
mkdir -p "$CAMERA_CAL_ROOT"

python3 -m fine_tuning.combine_selected_samples \
  --input /path/to/recording_A_front/epnp_selection_camera_anchored/selected_samples.csv \
  --input /path/to/recording_B_front/epnp_selection_camera_anchored/selected_samples.csv \
  --input /path/to/recording_C_front/epnp_selection_camera_anchored/selected_samples.csv \
  --output "$CAMERA_CAL_ROOT/combined_selected_samples_iteration_0.csv" \
  --dedupe-by match_key_epnp \
  --keep lowest-error
```

Include only paths that actually exist. If only one recording is currently
available, its selection can be used for a diagnostic calibration, but it is
not yet the intended pooled result. Never pool front, rear, and stereo-left
rows into this one-transform optimizer: each camera has its own
`T_lidar_camera`. Also do not pool recordings taken after the camera mount or
LiDAR calibration changed.

For a defensible result, reserve at least one complete recording, or a
spatially distributed independent subset, for validation instead of fitting
and evaluating on every accepted row.

## 11. Optimize the fixed camera/LiDAR extrinsic once from the pooled set

The centered optimizer is the appropriate implementation here. It explicitly
converts `T_map_object_raw` into the centered object convention used by the
GigaPose/RGB-CAD poses and starts from metadata `t_lidar_camera_prior`.

Because these are `gt_mesh_z_hybrid` labels, use `epnp_corrected`. It replaces
only the map Z of `T_map_lidar` with the label's
`gt_xy_mesh_z_label.ego_z_compensation.corrected_z_m`; XY and rotation stay
from metadata.

```bash
python3 -m fine_tuning.optimize_camera_lidar_extrinsics_centered \
  --selected-samples "$CAMERA_CAL_ROOT/combined_selected_samples_iteration_0.csv" \
  --use-sample-metadata \
  --epnp-label-dir-name EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-map-pose-unit m \
  --raw-object-center-m -0.2411941141 0.0009010172 0.3329219520 \
  --gigapose-pose-source aligned \
  --timestamp-alignment raw \
  --target-lidar-z-mode epnp_corrected \
  --translation-residual-components xyz \
  --translation-sigma-mm 1000 \
  --rotation-sigma-deg 10 \
  --image-center-weight 0 \
  --projection-model metadata \
  --translation-prior-weight 1000 \
  --rotation-prior-weight 20 \
  --robust-loss soft_l1 \
  --max-nfev 500 \
  --max-reusable-correction-mm 2000 \
  --output-dir "$CAMERA_CAL_ROOT/extrinsic_iteration_1"

python3 -m json.tool "$CAMERA_CAL_ROOT/extrinsic_iteration_1/optimization_report.json"
python3 -m json.tool "$CAMERA_CAL_ROOT/extrinsic_iteration_1/optimized_extrinsics.json"
```

Do not deploy it unless all of these hold:

- `optimizer_success` is true;
- `calibration_valid_for_reuse` is true;
- translation and rotation errors improve, including upper-tail behavior;
- correction size is physically plausible;
- overlays do not show a systematic wrong-frame solution;
- performance also improves on EPnP-supported frames not used to fit it.

Plot the before/after residual distributions:

```bash
python3 -m fine_tuning.plot_extrinsic_optimization \
  --optimization-dir "$CAMERA_CAL_ROOT/extrinsic_iteration_1" \
  --selected-samples "$CAMERA_CAL_ROOT/combined_selected_samples_iteration_0.csv" \
  --output-dir "$CAMERA_CAL_ROOT/extrinsic_iteration_1/plots"
```

The desired LiDAR-to-camera matrix is stored at:

```text
$CAMERA_CAL_ROOT/extrinsic_iteration_1/optimized_extrinsics.json
  -> T_camera_lidar_optimized
```

Here, `iteration_0` means candidate selection using the metadata/prior
extrinsic. `iteration_1` means the one camera calibration obtained from the
pooled iteration-0 correspondences.

## 12. Rerun selection for every recording using the optimized calibration

This paired selector replays the exact centered-object and corrected-Z
conventions recorded by the optimizer. It rejects a calibration marked unsafe:

```bash
python3 -m fine_tuning.select_real_label_candidates_with_centered_camera_lidar_extrinsics \
  --gigapose-predictions "$OPTIMIZED_POSES" \
  --dataset-dir "$DATASET" \
  --epnp-root "$EPNP_ROOT" \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --optimized-extrinsics "$CAMERA_CAL_ROOT/extrinsic_iteration_1/optimized_extrinsics.json" \
  --frame-transform-json "$RUN/epnp_selection/frame_transform_gigapose_to_epnp.json" \
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
  --output-dir "$RUN/epnp_selection_iteration_1"
```

Repeat this command separately for every recording used in the pool, changing
`RUN`, `DATASET`, `EPNP_ROOT`, and `OPTIMIZED_POSES` to that recording while
reusing the same camera-specific optimized-extrinsics JSON. Keep each
recording's own iteration-0 `frame_transform_gigapose_to_epnp.json`.

This closes the intended loop:

```text
per-recording tracked poses
  -> iteration-0 candidate selection
  -> same-camera pooled calibration
  -> iteration-1 candidate selection with optimized extrinsic
```

Do not optimize a second time automatically from the enlarged iteration-1
selection. That would turn the process into an uncontrolled self-selecting
loop. First inspect overlays and evaluate the optimized calibration on the
held-out recording/subset. A second calibration iteration should be a
separately reported experiment.

## Multiple recording folders: summary

Run Sections 1–9 separately for every available `(recording, camera)` pair.
This preserves true sequence boundaries and avoids smoothing across recording
gaps. Then combine only selected CSVs belonging to the same unchanged physical
camera/rig:

```bash
python3 -m fine_tuning.combine_selected_samples \
  --input /path/to/recording_A_front/epnp_selection/selected_samples.csv \
  --input /path/to/recording_B_front/epnp_selection/selected_samples.csv \
  --output /path/to/combined_front_selected_samples.csv \
  --dedupe-by match_key_epnp \
  --keep lowest-error
```

Pass the combined CSV to Section 11. Omit recordings that are not locally
available; do not create placeholder paths. After optimization, run Section
12 once per recording with the shared camera-specific calibration.

For a defensible calibration result, reserve at least one complete recording
or a spatially distributed subset of EPnP-supported frames for validation.
Selecting frames close to EPnP and evaluating on those same frames is
in-sample and will overstate generalization.

## Important interpretation

This pipeline refines GigaPose-derived camera/object predictions. The learned
models were trained on Assetto Corsa, so their real-world domain transfer is an
experiment, not an assumed improvement. The SciPy graph can make a trajectory
smoother while making absolute pose worse. EPnP overlays, calibration
residuals, held-out samples, correction magnitude, and cross-recording
stability must all agree before the extrinsic is treated as physical.

The EPnP labels themselves contain both `T_camera_object_centered` and
`T_map_object_raw`. If either was produced using the same extrinsic being
optimized, that part of the calibration evidence is circular. The strongest
validation uses independently supported map/LiDAR or manually checked frames.





<!-- -------------------------------- -->
<!--  ------------------------------- -->
<!-- -------------------------------- -->
You only need to rerun from the selector onward. GigaPose and candidate export can remain unchanged.

1. Define paths
```bash

cd /home/anahita/gigapose
conda activate gigapose

export CAMERA="front"

export RUN="$PWD/gigaPose_datasets/results/real_world_translation_cascade_scipy/20260518v1v4_front"
export DATASET="$PWD/gigaPose_datasets/datasets/real_20260518v1v4_front_gsam_v4"
export MESH="$DATASET/models/obj_000001.ply"
export CANDIDATES="$RUN/candidates"

export EPNP_ROOT="$PWD/gigaPose_datasets/datasets/2026-05-18-v1-v4/front/EPnPv2_gt_mesh_z_hybrid_labels"

export GP_CSV="$PWD/gigaPose_datasets/results/gigapose_real_20260518v1v4_front_translation_residual_rerun/predictions/large-pbrreal-rgb-mmodel_real_20260518v1v4_front_gsam_v4-test_gigapose_real_20260518v1v4_front_translation_residual_rerunMultiHypothesis.csv"

export SELECTOR_CKPT="$PWD/gigaPose_datasets/results/rgb_self_recovery_gru_dino_matching/models/dino_matching_gru/best.ckpt"

export TRANSLATION_KF_CKPT="$PWD/gigaPose_datasets/results/rgb_self_recovery_fixed_lag_kalman/model_translation_only/best.ckpt"

export ANCHORED_SELECTOR="$RUN/selector_camera_anchored"
export ANCHORED_MEASUREMENTS="$RUN/selector_camera_anchored_measurements"
export ANCHORED_CASCADE="$RUN/translation_cascade_camera_anchored"
export ANCHORED_GRAPH="$RUN/scipy_graph_camera_anchored"
```

2. Run the camera-anchored selector
EPnP and map weights are explicitly zero. EPnP is loaded only so the report can calculate diagnostics afterward.

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
  --output-dir "$ANCHORED_SELECTOR" \
  --device cuda \
  --overwrite
```

Inspect the report:
```bash
python3 -m json.tool "$ANCHORED_SELECTOR/run_report.json"
```

3. Export measurements for GRU–Kalman
```bash

python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.export_measurements \
  --dataset-dir "$DATASET" \
  --split test \
  --predictions "$ANCHORED_SELECTOR/tracked_predictions.csv" \
  --prediction-translation-unit mm \
  --sequence-max-frame-gap 1 \
  --sequence-max-time-gap-s 0.5 \
  --output-dir "$ANCHORED_MEASUREMENTS" \
  --overwrite

```
4. Run the translation-only cascade
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.filter \
  --data "$ANCHORED_MEASUREMENTS" \
  --checkpoint "$TRANSLATION_KF_CKPT" \
  --max-correction-translation-m 100 \
  --max-correction-rotation-deg 180 \
  --output-dir "$ANCHORED_CASCADE" \
  --device cuda \
  --overwrite
```

Check that rotation was preserved:
```bash
python3 -m json.tool "$ANCHORED_CASCADE/run_report.json"
```

5. Run the SciPy graph
The stronger 10° rotation prior helps prevent the graph from selecting the old flipped candidates.
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.sequence_factor_graph.run \
  --data "$CANDIDATES" \
  --mode prior \
  --prior-predictions "$ANCHORED_CASCADE/tracked_predictions.csv" \
  --prior-translation-unit mm \
  --prior-rotation-sigma-deg 10 \
  --output-dir "$ANCHORED_GRAPH" \
  --outer-iterations 5 \
  --maximum-nfev 100 \
  --overwrite
```
Inspect:
```bash
python3 -m json.tool "$ANCHORED_GRAPH/run_report.json"
```

Your final optimized poses are:
```bash
$ANCHORED_GRAPH/tracked_predictions.csv
```
6. Evaluate selector, cascade, and graph
This uses EPnP only after predictions have been generated.

```bash
for item in \
  "selector=$ANCHORED_SELECTOR/tracked_predictions.csv" \
  "cascade=$ANCHORED_CASCADE/tracked_predictions.csv" \
  "graph=$ANCHORED_GRAPH/tracked_predictions.csv"
do
  name="${item%%=*}"
  predictions="${item#*=}"

  python3 -m tracking.rgb_self_recovery.sliding_window.orientation_bootstrap.evaluate_anchors \
    --predictions "$predictions" \
    --dataset-dir "$DATASET" \
    --mesh "$MESH" \
    --epnp-root "$EPNP_ROOT" \
    --output-dir "$RUN/anchor_evaluation_$name" \
    --overwrite
done
```

View the summaries:
```bash
python3 -m json.tool "$RUN/anchor_evaluation_selector/report.json"
python3 -m json.tool "$RUN/anchor_evaluation_cascade/report.json"
python3 -m json.tool "$RUN/anchor_evaluation_graph/report.json"
```

The graph report is the final one. Check:
quality_approved.translation_error_m
quality_approved.rotation_error_deg
quality_approved.front_rear_flip_like_count
Ideally, front_rear_flip_like_count should be zero.


7. Generate overlays
Generate overlays for all four stages:
```bash
mkdir -p "$RUN/camera_anchored_overlays"

for item in \
  "gigapose=$GP_CSV" \
  "selector=$ANCHORED_SELECTOR/tracked_predictions.csv" \
  "cascade=$ANCHORED_CASCADE/tracked_predictions.csv" \
  "graph=$ANCHORED_GRAPH/tracked_predictions.csv"
do
  name="${item%%=*}"
  predictions="${item#*=}"

  python3 -m fine_tuning.overlay_gigapose_predictions \
    --predictions "$predictions" \
    --dataset-dir "$DATASET" \
    --split test \
    --mesh "$MESH" \
    --translation-scale 0.001 \
    --max-images 214 \
    --output-dir "$RUN/camera_anchored_overlays/$name"
done
```

Set the new output paths
```bash

export ANCHORED_SELECTOR="$RUN/selector_camera_anchored"
export ANCHORED_CASCADE="$RUN/translation_cascade_camera_anchored"
export ANCHORED_GRAPH="$RUN/scipy_graph_camera_anchored"

export OPTIMIZED_POSES="$ANCHORED_GRAPH/tracked_predictions.csv"

export COMPARISON_DIR="$RUN/no_gt_comparison_camera_anchored"
export OVERLAY_ROOT="$RUN/camera_anchored_overlays"
export EPNP_SELECTION="$RUN/epnp_selection_camera_anchored"
```
Check that all results exist:
```bash
test -f "$GP_CSV"
test -f "$ANCHORED_SELECTOR/tracked_predictions.csv"
test -f "$ANCHORED_CASCADE/tracked_predictions.csv"
test -f "$OPTIMIZED_POSES"
```

2. Compare trajectory changes without ground truth
```bash
python3 -m tracking.compare_predictions_without_gt \
  --model gigapose="$GP_CSV" \
  --model selector_camera_anchored="$ANCHORED_SELECTOR/tracked_predictions.csv" \
  --model translation_cascade_camera_anchored="$ANCHORED_CASCADE/tracked_predictions.csv" \
  --model scipy_graph_camera_anchored="$OPTIMIZED_POSES" \
  --reference gigapose \
  --dataset-dir "$DATASET" \
  --split test \
  --translation-unit mm \
  --camera "$CAMERA" \
  --output-dir "$COMPARISON_DIR"
```

3. Generate overlays for all stages
```bash
export MESH="$DATASET/models/obj_000001.ply"

mkdir -p "$OVERLAY_ROOT"

for item in \
  "gigapose=$GP_CSV" \
  "selector_camera_anchored=$ANCHORED_SELECTOR/tracked_predictions.csv" \
  "translation_cascade_camera_anchored=$ANCHORED_CASCADE/tracked_predictions.csv" \
  "scipy_graph_camera_anchored=$OPTIMIZED_POSES"
do
  name="${item%%=*}"
  predictions="${item#*=}"

  python3 -m fine_tuning.overlay_gigapose_predictions \
    --predictions "$predictions" \
    --dataset-dir "$DATASET" \
    --split test \
    --mesh "$MESH" \
    --translation-scale 0.001 \
    --max-images 214 \
    --output-dir "$OVERLAY_ROOT/$name"
done
```
4. Evaluate against EPnP anchors
Evaluate all three stages:
```bash
for item in \
  "selector=$ANCHORED_SELECTOR/tracked_predictions.csv" \
  "cascade=$ANCHORED_CASCADE/tracked_predictions.csv" \
  "graph=$OPTIMIZED_POSES"
do
  name="${item%%=*}"
  predictions="${item#*=}"

  python3 -m tracking.rgb_self_recovery.sliding_window.orientation_bootstrap.evaluate_anchors \
    --predictions "$predictions" \
    --dataset-dir "$DATASET" \
    --mesh "$MESH" \
    --epnp-root "$EPNP_ROOT" \
    --output-dir "$RUN/anchor_evaluation_camera_anchored_$name" \
    --overwrite
done
```

View the final graph report:
```bash
python3 -m json.tool \
  "$RUN/anchor_evaluation_camera_anchored_graph/report.json"
```

The report includes:
```bash
all.translation_error_m
all.rotation_error_deg
all.front_rear_flip_like_count

quality_approved.translation_error_m
quality_approved.rotation_error_deg
quality_approved.front_rear_flip_like_count
```

5. Select graph poses close to EPnP
```bash
# export EPNP_SELECTION="$RUN/epnp_selection_camera_anchored_moderate"


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
```

Then inspect:
```bash
python3 -m json.tool "$EPNP_SELECTION/selection_report.json"
```
And generate overlays:
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
