# Translation-cascade + SciPy on real-world ARCL recordings

This is the entry point for the real-world pipeline. Shared work is performed
once through RGB/CAD candidate generation. After that, choose exactly one of
two independent branches:

| Branch | Purpose | Output root |
|---|---|---|
| [Without anchoring](README_REAL_WORLD_WITHOUT_ANCHOR.md) | Reproduce the benchmark-style baseline | `$RUN/without_anchor` |
| [With camera anchoring](README_REAL_WORLD_WITH_CAMERA_ANCHOR.md) | Use camera-facing orientation initialization and fixed-lag handling | `$RUN/with_camera_anchor` |

For the real rear-camera run, use **with camera anchoring**. The rear-camera
rule expects the opponent's front to face the camera. This addresses the
front/rear flips previously observed in the ordinary selector. It does not use
EPnP translation or rotation to choose the pose: those weights are explicitly
zero. EPnP is loaded only for post-run diagnostics and later calibration.

Do not run the commands from both branch documents into the same directory.
The output roots above make the two experiments independent and comparable.

## 1. Shared paths for the current rear-camera run

Run from the repository root:


<!-- REARS NEED TO BE REDONE
# 20260518v1v4  
# rear:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4/rear/EPnPv2_gt_mesh_z_hybrid_labels  Might need to rerun

20260505v2  
# rear:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v2/rear/EPnPv2_gt_mesh_z_hybrid_labels Might need to rerun

20260505v1 
# rear:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/rear/EPnPv2_gt_mesh_z_hybrid_labels

20260718 
# rear:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-07-18/rear/EPnPv2_gt_mesh_z_hybrid_labels
 -->

<!-- 
20260526  None
----------------------------------------------
20260518v1v4  front, rear, stereo_left => Done
# 20260518v1v4  
# front:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4/front/EPnPv2_gt_mesh_z_hybrid_labels
# stereo_left: /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels
-----------------------------------------------------
20260518v0  front, stereo_left => Done
# front:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v0/front/EPnPv2_gt_mesh_z_hybrid_labels 
# stereo_left:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v0/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels 
-----------------------------------------------------
20260518v2v4  front and stereo left: => almost Done (lable selection pending)
# front: /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v2-v4/front/EPnPv2_gt_mesh_z_hybrid_labels
# stereo_left:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v2-v4/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels
--------------------------------------------------------



20260505v1   front, rear, stereo_left
# front:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/front/EPnPv2_gt_mesh_z_hybrid_labels
# stereo_left:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels

------------------------------------------------------
# 20260718 front, rear
# front: /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-07-18/front/EPnPv2_gt_mesh_z_hybrid_labels

  


model_ckpt: gigaPose_datasets/results/new_dataset_ckeckpoints/assettocorsa_ot2block_pose_aware_ist_translation_residual/checkpoints/best-residual-step010000.ckpt

  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-07-18/rear \
  --cad-path gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
  --dataset-name real_20260718_rear_gsam_v4 \
  --grounded-sam-dir Grounded_Sam_v4 \

   -->
cd /home/anahita/gigapose

unset RECORDING_ID CAMERA RAW_RECORDING RAW_CAMERA EPNP_ROOT
unset DATASET_NAME DATASET MESH

export RECORDING_ID="20260518v0"
export CAMERA="stereo_left"

export RAW_RECORDING="/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v0"
export RAW_CAMERA="${RAW_RECORDING}/${CAMERA}"
export EPNP_ROOT="${RAW_CAMERA}/EPnPv2_gt_mesh_z_hybrid_labels"

export DATASET_NAME="real_${RECORDING_ID}_${CAMERA}_gsam_v4"
export DATASET="$PWD/gigaPose_datasets/datasets/$DATASET_NAME"
export MESH="$DATASET/models/obj_000001.ply"

```bash
cd /home/anahita/gigapose
conda activate gigapose




20260505v1   front, rear, stereo_left
# front:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/front/EPnPv2_gt_mesh_z_hybrid_labels
# stereo_left:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels

------------------------------------------------------
# 20260718 front, rear
# front: /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-07-18/front/EPnPv2_gt_mesh_z_hybrid_labels



# export RECORDING_ID="20260518v1v4"
# export RECORDING_ID="20260518v0"
# export RECORDING_ID="20260505v2"
# export RECORDING_ID="20260518v2v4"
export RECORDING_ID="20260505v1"

export CAMERA="front"
# export CAMERA="rear"
export CAMERA="stereo_left"


# export RAW_RECORDING="/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4"
# export RAW_RECORDING="/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v0"
# export RAW_RECORDING="/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v2"
# export RAW_RECORDING="/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v2-v4"
export RAW_RECORDING="/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1"
# export RAW_RECORDING="/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4"

# export RECORDING_ID="20260505v1"
export RECORDING_ID="20260718"

export CAMERA="front"
# export CAMERA="rear"
# export CAMERA="stereo_left"


# export RAW_RECORDING="/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4"
# export RAW_RECORDING="/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v0"
# export RAW_RECORDING="/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v2"
# export RAW_RECORDING="/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v2-v4"
# export RAW_RECORDING="/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1"
export RAW_RECORDING="/home/anahita/gigapose/gigaPose_datasets/datasets/2026-07-18"

export RAW_CAMERA="$RAW_RECORDING/$CAMERA"
export EPNP_ROOT="$RAW_CAMERA/EPnPv2_gt_mesh_z_hybrid_labels"

export DATASET_NAME="real_${RECORDING_ID}_${CAMERA}_gsam_v4"
export DATASET="$PWD/gigaPose_datasets/datasets/$DATASET_NAME"
export MESH="$DATASET/models/obj_000001.ply"

export GIGAPOSE_CKPT="$PWD/gigaPose_datasets/results/new_dataset_ckeckpoints/assettocorsa_ot2block_pose_aware_ist_translation_residual/checkpoints/best-residual-step010000.ckpt"
export GIGAPOSE_RUN="gigapose_real_${RECORDING_ID}_${CAMERA}_translation_residual_rerun"
export GIGAPOSE_RESULT="$PWD/gigaPose_datasets/results/$GIGAPOSE_RUN"
export GP_CSV="$GIGAPOSE_RESULT/predictions/large-pbrreal-rgb-mmodel_${DATASET_NAME}-test_${GIGAPOSE_RUN}MultiHypothesis.csv"

export CANDIDATE_CKPT="$PWD/gigaPose_datasets/results/rgb_self_recovery_backbone_comparison/dino_matching_unet/best_pose.ckpt"
export SELECTOR_CKPT="$PWD/gigaPose_datasets/results/rgb_self_recovery_gru_dino_matching/models/dino_matching_gru/best.ckpt"
export TRANSLATION_KF_CKPT="$PWD/gigaPose_datasets/results/rgb_self_recovery_fixed_lag_kalman/model_translation_only/best.ckpt"

export RUN="$PWD/gigaPose_datasets/results/real_world_translation_cascade_scipy/${RECORDING_ID}_${CAMERA}"
export CANDIDATES="$RUN/candidates"
export WITHOUT_ANCHOR_RUN="$RUN/without_anchor"
export CAMERA_ANCHOR_RUN="$RUN/with_camera_anchor"

mkdir -p "$RUN"
```

The resulting directory tree is intentionally simple:

```text
real_world_translation_cascade_scipy/
  20260518v1v4_rear/
    candidates/                 shared RGB/CAD candidates
    without_anchor/             ordinary selector branch
    with_camera_anchor/         camera-facing selector branch
```

## 2. Shared preparation and validation

The prepared dataset must contain exactly one target mask per frame. For the
rear recording, `epnp_bbox` uses only the hybrid label's detector bounding box
to select the intended Grounded-SAM car mask; it does not use any EPnP pose.
The quality-approved-unique policy keeps only frames having exactly one EPnP
label that passes label-weight, projected-center, and projected-box checks.
Frames with no approved label or more than one approved label are skipped, so
the second car cannot enter through the no-label fallback.

Changing to this filtered preparation changes the frame set. Delete or
overwrite no files manually; run the preparation command with `--overwrite`,
then rerun validation, GigaPose, and shared candidate generation. Do not reuse
a GigaPose CSV or `candidates.npz` produced from the earlier 3,123-frame
fallback dataset.

```bash
python3 -m Assetto_data_prep.prepare_grounded_sam_inference \
  --source-root "$RAW_CAMERA" \
  --grounded-sam-dir Grounded_Sam_v4 \
  --cad-path "$PWD/gigaPose_datasets/datasets/racecar/models/obj_000001.ply" \
  --dataset-name "$DATASET_NAME" \
  --single-instance-policy epnp_bbox \
  --reference-label-dir EPnPv2_gt_mesh_z_hybrid_labels \
  --reference-label-policy all \
  --reference-valid-target-policy auto \
  --missing-reference-policy skip \
  --overwrite
```

Render templates only if missing:

```bash
export TEMPLATE_DIR="$PWD/gigaPose_datasets/datasets/templates/$DATASET_NAME"

if [ ! -f "$TEMPLATE_DIR/000001/000000.png" ]; then
  python3 -m src.scripts.render_custom_templates \
    custom_dataset_name="$DATASET_NAME" \
    machine.num_workers=1
fi
```

Validate:

```bash
python3 -m Assetto_data_prep.validate_grounded_sam_inference \
  --dataset-dir "$DATASET" \
  --split test \
  --minimum-mask-pixels 25 \
  --maximum-decoded-frames 512 \
  --require-templates \
  --output "$DATASET/validation_report.json"
```

Continue only when the report contains `"valid": true`.

## 3. Shared GigaPose inference

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.run_gigapose \
  --dataset-name "$DATASET_NAME" \
  --datasets-root "$PWD/gigaPose_datasets/datasets" \
  --checkpoint "$GIGAPOSE_CKPT" \
  --run-name "$GIGAPOSE_RUN" \
  --batch-size 16 \
  --num-workers 2 \
  --devices 0

test -f "$GP_CSV" && echo "GigaPose CSV ready: $GP_CSV"
```

## 4. Shared RGB/CAD candidate generation

Run this once. Both branches consume the same candidate bundle.

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
  --output-dir "$CANDIDATES" \
  --device cuda \
  --overwrite

python3 -m json.tool "$CANDIDATES/manifest.json"
```

## 5. You are here after candidate generation

For the current rear-camera run, continue with:

```bash
sed -n '1,260p' \
  tracking/rgb_self_recovery/sliding_window/sequence_factor_graph/README_REAL_WORLD_WITH_CAMERA_ANCHOR.md
```

The first next command is the camera-anchored selector. Do not run the ordinary
`gru_selector.select` first unless you intentionally want the unanchored
baseline comparison.

## 6. Optional branch comparison

Run this only after both branches have completed:

```bash
python3 -m tracking.compare_predictions_without_gt \
  --model gigapose="$GP_CSV" \
  --model without_anchor="$WITHOUT_ANCHOR_RUN/scipy_graph/tracked_predictions.csv" \
  --model with_camera_anchor="$CAMERA_ANCHOR_RUN/scipy_graph/tracked_predictions.csv" \
  --reference gigapose \
  --dataset-dir "$DATASET" \
  --split test \
  --translation-unit mm \
  --camera "$CAMERA" \
  --output-dir "$RUN/branch_comparison_without_gt"
```

This comparison measures trajectory changes relative to GigaPose; it is not
ground-truth accuracy.

## 7. Calibration happens after per-recording selection

Whichever branch you choose produces:

```text
<branch>/epnp_selection_iteration_0/selected_samples.csv
```

Run the complete branch independently for every available recording. Then
combine selected CSVs only from the same branch and same unchanged physical
camera. For the recommended camera-anchored rear branch:

```bash
export CAMERA_CAL_ROOT="$PWD/gigaPose_datasets/results/real_world_translation_cascade_scipy/calibration_${CAMERA}/with_camera_anchor"
mkdir -p "$CAMERA_CAL_ROOT"

python3 -m fine_tuning.combine_selected_samples \
  --input /path/to/recording_A_rear/with_camera_anchor/epnp_selection_iteration_0/selected_samples.csv \
  --input /path/to/recording_B_rear/with_camera_anchor/epnp_selection_iteration_0/selected_samples.csv \
  --output "$CAMERA_CAL_ROOT/combined_selected_samples_iteration_0.csv" \
  --dedupe-by match_key_epnp \
  --keep lowest-error
```

Do not combine rear, front, and stereo-left rows. Each physical camera has a
different fixed `T_lidar_camera`.

Optimize one rear-camera extrinsic from the pooled set:

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
  --translation-prior-weight 1000 \
  --rotation-prior-weight 20 \
  --robust-loss soft_l1 \
  --max-nfev 500 \
  --max-reusable-correction-mm 2000 \
  --output-dir "$CAMERA_CAL_ROOT/extrinsic_iteration_1"
```

Only use a result whose report marks it safe and valid for reuse. Then rerun
selection for each recording using that shared camera-specific extrinsic. The
branch guides provide the exact iteration-1 command and use each recording's
own object-frame transform.

## Pose convention

`T_A_B` maps points from B into A. Therefore:

```text
p_lidar = T_lidar_camera * p_camera
```

The calibration JSON writes both:

- `T_lidar_camera_optimized`: camera to LiDAR;
- `T_camera_lidar_optimized`: LiDAR to camera.

The graph pose CSV and extrinsic JSON are different products. The graph CSV
contains per-frame camera/object poses; the extrinsic JSON contains one fixed
camera/LiDAR transform.
