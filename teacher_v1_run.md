








TRACKED=gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_rotation_gated/tracked_predictions.csv 

DATASET=gigaPose_datasets/datasets/real_20260505v1_front_gsam_v4

EPNP=/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/front/EPnPv2_gt_mesh_z_hybrid_labels

META=/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/front/metadata

TRACK_SURFACE=/media/hdd2/ARCL_multicar_bags/camera_dataset/Track_info/sim_track_info/putnam_park-no_chicanes_track_info/track_scene.ply

CAR_MESH="$DATASET/models/obj_000001.ply"




### V1-baseline:

RUN_v1=gigaPose_datasets/results/teacher_v1_20260505_front
V1_ALL_RUN="$RUN_v1/v1_all_frames"
mkdir -p "$V1_ALL_RUN"
RUN=gigaPose_datasets/results/teacher_v1_extended_20260505_front

Run the original V1 configuration:
python3 -m teacher_pipeline.v1.trajectory \
  --observations "$RUN/full_observations.jsonl" \
  --output "$V1_ALL_RUN/initial_iteration_0.jsonl"


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