# Obsolete mixed draft — do not run

This file is retained only as historical notes. Its standard and anchored
commands are interleaved and its paths are not the current layout. Use:

- `README_REAL_WORLD_TRANSLATION_CASCADE.md` for shared setup;
- `README_REAL_WORLD_WITHOUT_ANCHOR.md` for the baseline branch;
- `README_REAL_WORLD_WITH_CAMERA_ANCHOR.md` for the recommended anchored branch.

---

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

## What is available for 
`2026-05-18-v1-v4`

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
export CAMERA="rear"

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


cd /home/anahita/gigapose

python3 -m Assetto_data_prep.prepare_grounded_sam_inference \
  --source-root "$RAW_CAMERA" \
  --grounded-sam-dir Grounded_Sam_v4 \
  --cad-path "$PWD/gigaPose_datasets/datasets/racecar/models/obj_000001.ply" \
  --dataset-name "$DATASET_NAME" \
  --single-instance-policy epnp_bbox \
  --reference-label-dir EPnPv2_gt_mesh_z_hybrid_labels \
  --overwrite



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

Now rerun GigaPose:

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



# ##################################################
# ##################################################
The intended process is:
Run tracking independently for every recording
        ↓
Select reliable EPnP/tracked-pose pairs in each recording
        ↓
Combine selected pairs from the same physical camera
        ↓
Optimize one fixed extrinsic for that camera
        ↓
Rerun candidate selection in every recording
using the optimized extrinsic


Important: combine all front recordings together, all rear recordings together, etc. Do not combine front, rear, and stereo-left because they have different camera–LiDAR extrinsics.

# Camera Extrinsic Optimization:
## 1. Combine iteration-0 selections

```bash
export CAL_ROOT="$PWD/gigaPose_datasets/results/real_world_translation_cascade_scipy/calibration_front"
mkdir -p "$CAL_ROOT"

python3 -m fine_tuning.combine_selected_samples \
  --input "$RUN_A/epnp_selection_camera_anchored/selected_samples.csv" \
  --input "$RUN_B/epnp_selection_camera_anchored/selected_samples.csv" \
  --input "$RUN_C/epnp_selection_camera_anchored/selected_samples.csv" \
  --output "$CAL_ROOT/combined_selected_samples_iteration_0.csv" \
  --dedupe-by match_key_epnp \
  --keep lowest-error

```
Only include recordings currently available and captured with the unchanged front-camera mount.

## 2. Optimize once using the combined samples
```bash
python3 -m fine_tuning.optimize_camera_lidar_extrinsics_centered \
  --selected-samples "$CAL_ROOT/combined_selected_samples_iteration_0.csv" \
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
  --translation-prior-weight 1000 \
  --rotation-prior-weight 20 \
  --robust-loss soft_l1 \
  --max-nfev 500 \
  --max-reusable-correction-mm 2000 \
  --output-dir "$CAL_ROOT/extrinsic_iteration_1"
```

Check:
```bash
python3 -m json.tool \
  "$CAL_ROOT/extrinsic_iteration_1/optimized_extrinsics.json"
```
Do not use --allow-unsafe-calibration if the result fails its safety checks.

## 3. Rerun selection for each recording
For each original recording, set its RUN, DATASET, EPNP_ROOT, and OPTIMIZED_POSES, then run:
```bash
python3 -m fine_tuning.select_real_label_candidates_with_centered_camera_lidar_extrinsics \
  --gigapose-predictions "$OPTIMIZED_POSES" \
  --dataset-dir "$DATASET" \
  --epnp-root "$EPNP_ROOT" \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --optimized-extrinsics "$CAL_ROOT/extrinsic_iteration_1/optimized_extrinsics.json" \
  --frame-transform-json "$RUN/epnp_selection_camera_anchored/frame_transform_gigapose_to_epnp.json" \
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
