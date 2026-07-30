








TRACKED=gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_rotation_gated/tracked_predictions.csv 

DATASET=gigaPose_datasets/datasets/real_20260505v1_front_gsam_v4

EPNP=/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/front/EPnPv2_gt_mesh_z_hybrid_labels

META=/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/front/metadata

TRACK_SURFACE=/media/hdd2/ARCL_multicar_bags/camera_dataset/Track_info/sim_track_info/putnam_park-no_chicanes_track_info/track_scene.ply

CAR_MESH="$DATASET/models/obj_000001.ply"


### V1-baseline original 100 frame run:
RUN=gigaPose_datasets/results/teacher_v1_20260505_front

PREDICTIONS=gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260505v1_front_gsam_v4-test_large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv

SELECTED=gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/selected_samples.csv

TRACKS=gigaPose_datasets/results/real_20260505v1_front_gsam_v4_tracking/tracked_predictions.csv

mkdir -p "$RUN"

# 1.Build the 100 observations

python3 -m teacher_pipeline.build_observations \
  --predictions "$PREDICTIONS" \
  --selected-samples "$SELECTED" \
  --tracks "$TRACKS" \
  --prediction-translation-unit mm \
  --epnp-map-pose-unit m \
  --epnp-camera-pose-unit m \
  --gigapose-pose-source auto \
  --target-lidar-z-mode epnp_corrected \
  --output "$RUN/observations.jsonl" \
  --strict

The report should say approximately:
prediction_rows: 2400
selected_rows: 100
observations_written: 100
resolved_gigapose_pose_source: aligned_csv
target_lidar_z_mode: epnp_corrected
corrected_lidar_z_count: 100
track_rows: 480
pose_associated_track_ids: 100

Inspect it with:
python3 -m json.tool "$RUN/observations.report.json"

# 2.Initialize iteration 0
python3 -m teacher_pipeline.trajectory \
  --observations "$RUN/observations.jsonl" \
  --output "$RUN/initial_trajectories_iteration_0.jsonl"

# 3. Refine iteration 0
python3 -m teacher_pipeline.refine_trajectories \
  --trajectories "$RUN/initial_trajectories_iteration_0.jsonl" \
  --epnp-anchor-weight 0.35 \
  --output "$RUN/refined_iteration_0.jsonl"

 # 4. Optimize the fixed extrinsic
 python3 -m teacher_pipeline.optimize_extrinsic \
  --trajectories "$RUN/refined_iteration_0.jsonl" \
  --output "$RUN/extrinsic_iteration_1.json"

Your successful original result had:
translation median: 2.292 → 1.599 m
rotation median:    2.415 → 2.315 degrees
extrinsic correction translation: 0.743 m
extrinsic correction rotation:    0.859 degrees

Inspect it: 
python3 -m json.tool "$RUN/extrinsic_iteration_1.json"

# 5. Reinitialize using the optimized extrinsic
python3 -m teacher_pipeline.trajectory \
  --observations "$RUN/observations.jsonl" \
  --extrinsics "$RUN/extrinsic_iteration_1.json" \
  --output "$RUN/initial_trajectories_iteration_1.jsonl"

# 6. Refine iteration 1
python3 -m teacher_pipeline.refine_trajectories \
  --trajectories "$RUN/initial_trajectories_iteration_1.jsonl" \
  --epnp-anchor-weight 0.35 \
  --output "$RUN/refined_iteration_1.jsonl"

# 7. Evaluate and make plots
python3 -m teacher_pipeline.evaluate \
  --trajectories "$RUN/refined_iteration_1.jsonl" \
  --extrinsics "$RUN/extrinsic_iteration_1.json" \
  --output-dir "$RUN/evaluation"

The original overlay command was:
python3 -m teacher_pipeline.evaluate \
  --trajectories "$RUN/refined_iteration_1.jsonl" \
  --extrinsics "$RUN/extrinsic_iteration_1.json" \
  --mesh gigaPose_datasets/datasets/real_20260505v1_front_gsam_v4/models/obj_000001.ply \
  --mesh-object-origin raw \
  --max-overlays 100 \
  --output-dir "$RUN/evaluation_with_overlays"


### V1-baseline:

RUN_v1=gigaPose_datasets/results/teacher_v1_20260505_front
V1_ALL_RUN="$RUN_v1/v1_all_frames"
export V1_RAW_RUN="$RUN_v1/v1_all_frames_raw_only"
mkdir -p "$V1_RAW_RUN"
mkdir -p "$V1_ALL_RUN"
RUN=gigaPose_datasets/results/teacher_v1_extended_20260505_front

Run the original V1 configuration:
python3 -m teacher_pipeline.v1.trajectory \
  --observations "$RUN/full_observations.jsonl" \
  --output "$V1_ALL_RUN/initial_iteration_0.jsonl"


python3 -m teacher_pipeline.v1.refine_trajectories \
  --trajectories "$V1_ALL_RUN/initial_iteration_0.jsonl" \
  --track-map "$RUN/track_map.npz" \
  --iterations 4 \
  --epnp-anchor-weight 0 \
  --output "$V1_RAW_RUN/refined_iteration_0.jsonl"

Then apply baseline smoothing:
python3 -m teacher_pipeline.v1.refine_trajectories \
  --trajectories "$V1_ALL_RUN/initial_iteration_0.jsonl" \
  --track-map "$RUN/track_map.npz" \
  --iterations 4 \
  --epnp-anchor-weight 0.35 \
  --output "$V1_ALL_RUN/refined_iteration_0.jsonl"

Verify equal coverage:
  wc -l \
  "$V1_ALL_RUN/refined_iteration_0.jsonl" \
  "$RUN/refined_physical_iteration_0_route_fixed.jsonl"

### V1-extended 
RUN=gigaPose_datasets/results/teacher_v1_extended_20260505_front

Regenerate the map:
python3 -m teacher_pipeline.v1_extended.extract_track_map_from_surface \
  --surface-ply "$TRACK_SURFACE" \
  --guide-observations "$RUN/full_observations.jsonl" \
  --axis-order x-negz-y \
  --resolution-m 0.5 \
  --output "$RUN/track_map.npz"


Check that correction was applied:
python3 -m json.tool "$RUN/track_map.report.json" | \
  grep -A12 '"guide_route_corrections"'

  Then inspect $RUN/track_map.png before running refinement again. The red line should now cross the central junction diagonally with the cyan route, without taking the small triangular detour.



Now verify the new track length and alignment:
python3 -m json.tool "$RUN/track_map.report.json" | \
  grep '"track_length_m"'

python3 -m teacher_pipeline.v1_extended.validate_track_alignment \
  --observations "$RUN/full_observations.jsonl" \
  --track-map "$RUN/track_map.npz"


If the red centerline also looks correct visually, run the route-fixed refinement:
```bash
python3 -m teacher_pipeline.v1_extended.refine_track_trajectories \
  --trajectories "$RUN/initial_physical_iteration_0.jsonl" \
  --track-map "$RUN/track_map.npz" \
  --pose-source raw \
  --output "$RUN/refined_physical_iteration_0_route_fixed.jsonl"

  ```
Then evaluate:
```bash
python3 -m teacher_pipeline.v1_extended.evaluate_physical \
  --trajectories "$RUN/refined_physical_iteration_0_route_fixed.jsonl" \
  --track-map "$RUN/track_map.npz" \
  --output-dir "$RUN/physical_evaluation_iteration_0_route_fixed"
  ```
Inspect:
```bash
python3 -m json.tool \
  "$RUN/physical_evaluation_iteration_0_route_fixed/physical_summary.json"
```


## Compare baseline V1 with V1 Extended
# 1. Compare baseline V1 with V1 Extendeds
export V1_TRAJECTORIES="gigaPose_datasets/results/teacher_v1_20260505_front/refined_iteration_0.jsonl"


python3 -m teacher_pipeline.v1_extended.compare_versions \
  --v1-trajectories "$V1_TRAJECTORIES" \
  --extended-trajectories "$RUN/refined_physical_iteration_0_route_fixed.jsonl" \
  --output-dir "$RUN/v1_vs_extended_route_fixed"


Inspect:
python3 -m json.tool \
  "$RUN/v1_vs_extended_route_fixed/summary.json"


# 2. Optionally generate CAD overlays

Add these arguments to the comparison command:
python3 -m teacher_pipeline.v1_extended.compare_versions \
  --v1-trajectories "$V1_TRAJECTORIES" \
  --extended-trajectories "$RUN/refined_physical_iteration_0_route_fixed.jsonl" \
  --output-dir "$RUN/v1_vs_extended_route_fixed" \
--mesh "$CAR_MESH" \
  --mesh-object-origin raw \
  --max-overlays 100

Overlay colors are:
Red: raw GigaPose
Yellow: baseline V1
Green: V1 Extended
Blue: EPnP reference


Now compare all common frames:
python3 -m teacher_pipeline.v1_extended.compare_versions \
  --v1-trajectories "$V1_ALL_RUN/refined_iteration_0.jsonl" \
  --extended-trajectories "$RUN/refined_physical_iteration_0_route_fixed.jsonl" \
  --output-dir "$RUN/v1_all_vs_extended_route_fixed" \
  --mesh "$CAR_MESH" \
  --mesh-object-origin raw \
  --max-overlays 100




  ## Hybrid teacher? 
  Original trusted V1: 100 selected frames.
  Route-fixed Extended: remaining 380 frames.

  I added the merge script. First confirm $V1_TRAJECTORIES points to the original 100-row V1 result:
  wc -l "$V1_TRAJECTORIES"

  It should print 100. 

  Build the hybrid:
  python3 -m teacher_pipeline.v1_extended.build_hybrid_teacher \
  --selected-v1-trajectories "$V1_TRAJECTORIES" \
  --extended-trajectories "$RUN/refined_physical_iteration_0_route_fixed.jsonl" \
  --output "$RUN/hybrid_teacher_final.jsonl"

  Inspect its report:
  python3 -m json.tool "$RUN/hybrid_teacher_final.report.json"

  Evaluate the final poses:
 python3 -m teacher_pipeline.v1_extended.evaluate_physical \
  --trajectories "$RUN/hybrid_teacher_final.jsonl" \
  --track-map "$RUN/track_map.npz" \
  --output-dir "$RUN/hybrid_teacher_evaluation" 

  Then export the training labels:
  python3 -m teacher_pipeline.v1_extended.export_student_labels \
  --trajectories "$RUN/hybrid_teacher_final.jsonl" \
  --extrinsics-version "epnp_hybrid_initial_fixed_v0" \
  --output-dir "$RUN/student_labels_hybrid_final"

  Use:
  $RUN/student_labels_hybrid_final/labels.jsonl




-------------------------------------------
# Rerun the originla v1 experiment for full 480 dataset:

1. Set paths

 ```bash
cd ~/gigapose

V1_RAW_RUN=gigaPose_datasets/results/teacher_v1_full_raw_20260505_front_rerun

RAW_GP=gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260505v1_front_gsam_v4-test_large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tran.csv

TRACK_IDS=gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_rotation_gated/tracked_predictions.csv


DATASET=gigaPose_datasets/datasets/real_20260505v1_front_gsam_v4

EPNP_ROOT=/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/front/EPnPv2_gt_mesh_z_hybrid_labels

METADATA_ROOT=/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/front/metadata

mkdir -p "$V1_RAW_RUN"

 ```
Confirm the prepared dataset path:
 ```bash
test -f "$DATASET/frame_map.json" && echo "Dataset found" || echo "DATASET path is wrong"
```
2. Build all-frame raw-GigaPose observations
 ```bash
python3 -m teacher_pipeline.v1_extended.build_full_observations \
  --predictions "$RAW_GP" \
  --track-ids-from "$TRACK_IDS" \
  --dataset-dir "$DATASET" \
  --epnp-root "$EPNP_ROOT" \
  --metadata-root "$METADATA_ROOT" \
  --prediction-translation-unit mm \
  --gigapose-object-origin raw \
  --output "$V1_RAW_RUN/full_raw_observations.jsonl" \
  --strict

 ```

R, t, and score come from the original GigaPose CSV.
Only track_id comes from tracked_predictions.csv.
EPnP poses are attached for evaluation.
EPnP poses will not influence refinement because we will use anchor weight zero.

Check the report:

```bash
python3 -m json.tool "$V1_RAW_RUN/full_raw_observations.report.json"
```
prediction_rows: 480
observations_written: 480
skipped_count: 0
track_id_source_counts:
track_ids_from.unique_frame: 480


3. Initialize raw V1 trajectories
Use the metadata’s original fixed camera–LiDAR extrinsic:
```bash
python3 -m teacher_pipeline.v1.trajectory \
  --observations "$V1_RAW_RUN/full_raw_observations.jsonl" \
  --output "$V1_RAW_RUN/initial_trajectories_iteration_0.jsonl"
```

4. Run V1 temporal refinement
The critical setting is --epnp-anchor-weight 0:

```bash
python3 -m teacher_pipeline.v1.refine_trajectories \
  --trajectories "$V1_RAW_RUN/initial_trajectories_iteration_0.jsonl" \
  --epnp-anchor-weight 0 \
  --output "$V1_RAW_RUN/refined_iteration_0.jsonl"
```
This tests whether V1’s temporal smoothing alone improves raw GigaPose.

5. Evaluate raw GigaPose versus V1
```bash
python3 -m teacher_pipeline.v1.evaluate \
  --trajectories "$V1_RAW_RUN/refined_iteration_0.jsonl" \
  --output-dir "$V1_RAW_RUN/evaluation_iteration_0"
```

Inspect:
```bash
python3 -m json.tool "$V1_RAW_RUN/evaluation_iteration_0/summary.json"
```