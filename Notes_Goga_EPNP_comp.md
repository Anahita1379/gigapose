Compare GigaPose predictions to the actual EPnPv2 label files:
<!-- 
for each camera seperately and for the predictions:
# 20260505v1 
front:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/front/EPnPv2_gt_mesh_z_hybrid_labels
rear:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/rear/EPnPv2_gt_mesh_z_hybrid_labels
stereo_left:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels


# 20260505v2  
rear:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v2/rear/EPnPv2_gt_mesh_z_hybrid_labels (No need to rerun)


# 20260518v0 
front:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v0/front/EPnPv2_gt_mesh_z_hybrid_labels (No need to rerun)
stereo_left:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v0/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels 


# 20260518v1v4 
front:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4/front/EPnPv2_gt_mesh_z_hybrid_labels
rear:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4/rear/EPnPv2_gt_mesh_z_hybrid_labels
stereo_left: /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels

# 20260526 
None

20260518v2v4
front: /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v2-v4/front/EPnPv2_gt_mesh_z_hybrid_labels
stereo_left:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v2-v4/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels
 -->
 
```bash 


20260518v2v4
front: /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v2-v4/front/EPnPv2_gt_mesh_z_hybrid_labels
stereo_left:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v2-v4/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels

 python -m fine_tuning.select_real_label_candidates \
  --gigapose-predictions gigaPose_datasets/results/real_world_data/large_real_20260518v2v4_front_gsam_v4_finetuned/predictions/large-pbrreal-rgb-mmodel_real_20260518v2v4_front_gsam_v4-test_real_20260518v2v4_front_gsam_v4_finetunedMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v2v4_front_gsam_v4 \
  --epnp-root  /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v2-v4/front/EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --frame-transform-side right \
  --frame-transform-refine-iterations 5 \
  --frame-transform-inlier-translation-mm 5000 \
  --frame-transform-inlier-rotation-deg 60 \
  --min-score 0.05 \
  --max-translation-error-mm 2000 \
  --max-rotation-error-deg 30 \
  --output-dir gigaPose_datasets/results/real_world_data/large_real_20260518v2v4_front_gsam_v4_finetuned/label_candidates_refined

  python -m fine_tuning.select_real_label_candidates \
  --gigapose-predictions gigaPose_datasets/results/real_world_data/large_real_20260518v2v4_rear_gsam_v4_finetuned/predictions/large-pbrreal-rgb-mmodel_real_20260518v2v4_rear_gsam_v4-test_real_20260518v2v4_rear_gsam_v4_finetunedMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v2v4_rear_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4/rear/EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --frame-transform-side right \
  --frame-transform-refine-iterations 5 \
  --frame-transform-inlier-translation-mm 5000 \
  --frame-transform-inlier-rotation-deg 60 \
  --min-score 0.05 \
  --max-translation-error-mm 2000 \
  --max-rotation-error-deg 30 \
  --output-dir gigaPose_datasets/results/real_world_data/large_real_20260518v2v4_rear_gsam_v4_finetuned/label_candidates_refined


  python -m fine_tuning.select_real_label_candidates \
  --gigapose-predictions gigaPose_datasets/results/real_world_data/large_real_20260518v2v4_stereo_left_gsam_v4_finetuned/predictions/large-pbrreal-rgb-mmodel_real_20260518v2v4_stereo_left_gsam_v4-test_real_20260518v2v4_stereo_left_gsam_v4_finetunedMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v2v4_stereo_left_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v2-v4/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --frame-transform-side right \
  --frame-transform-refine-iterations 5 \
  --frame-transform-inlier-translation-mm 5000 \
  --frame-transform-inlier-rotation-deg 60 \
  --min-score 0.05 \
  --max-translation-error-mm 2000 \
  --max-rotation-error-deg 30 \
  --output-dir gigaPose_datasets/results/real_world_data/large_real_20260518v2v4_stereo_left_gsam_v4_finetuned/label_candidates_refined
```


select_real_label_candidates.py gives you numeric comparison files. 
It writes these into --output-dir:
frame_transform_gigapose_to_epnp.json
The estimated fixed frame transform X such that: 
```bash 
T_epnp ≈ X @ T_gigapose
```
all_candidate_pairs.csv
Every matched GigaPose prediction vs EPnP label candidate. Important columns
```bash 
match_key
scene_id
im_id
score
epnp_label_path
translation_error_mm
rotation_error_deg
T_gigapose_cam_obj
T_gigapose_aligned_epnp_obj
T_epnp_obj
```

best_candidate_per_epnp_label.csv
Best GigaPose prediction for each EPnP label.

selected_samples.csv
Only samples passing your thresholds:
```bash 
--max-translation-error-mm
--max-rotation-error-deg
```

selection_report.json
Summary like number of candidate pairs, number selected, median/mean/p90 translation and rotation error.

To plot the numeric comparison:
```bash 
python -m fine_tuning.plot_epnp_gigapose_comparison \
  --input-csv gigaPose_datasets/results/real_world_data/large_real_20260518v2v4_front_gsam_v4_finetuned/label_candidates_refined/best_candidate_per_epnp_label.csv

python -m fine_tuning.plot_epnp_gigapose_comparison \
  --input-csv gigaPose_datasets/results/real_world_data/large_real_20260518v2v4_rear_gsam_v4_finetuned/label_candidates_refined/best_candidate_per_epnp_label.csv

python -m fine_tuning.plot_epnp_gigapose_comparison \
  --input-csv gigaPose_datasets/results/real_world_data/large_real_20260518v2v4_stereo_left_gsam_v4_finetuned/label_candidates_refined/best_candidate_per_epnp_label.csv
```
This writes plots to:
gigaPose_datasets/results/real_20260505_front_gsam_v4_label_candidates/plots_epnp_gigapose/

To visualize EPnPv2 vs GigaPose on the actual images:
```bash 
python -m fine_tuning.visualize_epnp_gigapose_comparison \
  --candidate-csv gigaPose_datasets/results/real_world_data/large_real_20260518v2v4_front_gsam_v4_finetuned/label_candidates_refined/best_candidate_per_epnp_label.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v2v4_front_gsam_v4 \
  --output-dir gigaPose_datasets/results/real_world_data/large_real_20260518v2v4_front_gsam_v4_finetuned/label_candidates_refined/visual_overlays \
  --max-images 100 \
  --draw-mask-bbox

python -m fine_tuning.visualize_epnp_gigapose_comparison \
  --candidate-csv gigaPose_datasets/results/real_world_data/large_real_20260518v2v4_rear_gsam_v4_finetuned/label_candidates_refined/best_candidate_per_epnp_label.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v2v4_rear_gsam_v4 \
  --output-dir gigaPose_datasets/results/real_world_data/large_real_20260518v2v4_rear_gsam_v4_finetuned/label_candidates_refined/visual_overlays \
  --max-images 100 \
  --draw-mask-bbox


  python -m fine_tuning.visualize_epnp_gigapose_comparison \
  --candidate-csv gigaPose_datasets/results/real_world_data/large_real_20260518v2v4_stereo_left_gsam_v4_finetuned/label_candidates_refined/best_candidate_per_epnp_label.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v2v4_stereo_left_gsam_v4 \
  --output-dir gigaPose_datasets/results/real_world_data/large_real_20260518v2v4_stereo_left_gsam_v4_finetuned/label_candidates_refined/visual_overlays \
  --max-images 100 \
  --draw-mask-bbox


```
If you want to look at the worst cases first:
```bash
python -m fine_tuning.visualize_epnp_gigapose_comparison \
  --candidate-csv gigaPose_datasets/results/real_20260505_front_gsam_v4_label_candidates/best_candidate_per_epnp_label.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505_front_gsam_v4 \
  --output-dir gigaPose_datasets/results/real_20260505_front_gsam_v4_label_candidates/visual_overlays_worst_translation \
  --sort-by translation_error \
  --max-images 100 \
  --draw-mask-bbox
```


Now, time for optimizing the camera extrinsics: 
1. Use selected_samples.csv only as the trusted GigaPose/EPnP matches.
2. Optimize T_map_cam using the map-frame EPnP pose from each selected label JSON.

First, check that the EPnP labels have a map pose key:
```bash
python - <<'PY'
import csv, json

selected = "gigaPose_datasets/results/real_world_data/large_real_20260505v2_front_gsam_v4_finetuned/label_candidates_refined/selected_samples.csv"

row = next(csv.DictReader(open(selected)))
label_path = row["epnp_label_path"]
data = json.load(open(label_path))

print("label:", label_path)
print("available keys:")
for k in data.keys():
    if "T_" in k or "pose" in k.lower() or "map" in k.lower():
        print(" ", k)
PY
```


If you have multiple selected files, pass multiple --input arguments:
Important: only combine files for the same camera.

```bash
For only front: 

python -m fine_tuning.combine_selected_samples \
  --input gigaPose_datasets/results/real_world_data/large_real_20260505v1_front_gsam_v4_finetuned/label_candidates_refined/selected_samples.csv \
  --input gigaPose_datasets/results/real_world_data/large_real_20260518v0_front_gsam_v4_finetuned/label_candidates_refined/selected_samples.csv \
  --input gigaPose_datasets/results/real_world_data/large_real_20260518v1v4_front_gsam_v4_finetuned/label_candidates_refined/selected_samples.csv \
  --input gigaPose_datasets/results/real_world_data/large_real_20260518v2v4_front_gsam_v4_finetuned/label_candidates_refined/selected_samples.csv \
  --output gigaPose_datasets/results/real_world_data/combined_front_selected_samples.csv \
  --dedupe-by match_key_epnp \
  --keep lowest-error


For only rear: 
python -m fine_tuning.combine_selected_samples \
  --input gigaPose_datasets/results/real_world_data/large_real_20260505v1_rear_gsam_v4_finetuned/label_candidates_refined/selected_samples.csv \
  --input gigaPose_datasets/results/real_world_data/large_real_20260505v2_rear_gsam_v4_finetuned/label_candidates_refined/selected_samples.csv \
  --input gigaPose_datasets/results/real_world_data/large_real_20260518v1v4_rear_gsam_v4_finetuned/label_candidates_refined/selected_samples.csv \
  --output gigaPose_datasets/results/real_world_data/combined_rear_selected_samples.csv \
  --dedupe-by match_key_epnp \
  --keep lowest-error


For only stereo_left: 
python -m fine_tuning.combine_selected_samples \
  --input gigaPose_datasets/results/real_world_data/large_real_20260505v1_stereo_left_gsam_v4_finetuned/label_candidates_refined/selected_samples.csv \
  --input gigaPose_datasets/results/real_world_data/large_real_20260518v0_stereo_left_gsam_v4_finetuned/label_candidates_refined/selected_samples.csv \
  --input gigaPose_datasets/results/real_world_data/large_real_20260518v1v4_stereo_left_gsam_v4_finetuned/label_candidates_refined/selected_samples.csv \
  --input gigaPose_datasets/results/real_world_data/large_real_20260518v2v4_stereo_left_gsam_v4_finetuned/label_candidates_refined/selected_samples.csv \
  --output gigaPose_datasets/results/real_world_data/combined_stereo_left_selected_samples.csv \
  --dedupe-by match_key_epnp \
  --keep lowest-error
```


Then the combined file can be used here:
```bash
python -m fine_tuning.optimize_camera_map_extrinsics \
  --selected-samples gigaPose_datasets/results/real_world_data/combined_front_selected_samples.csv \

```

Then run extrinsic optimization using the map pose key:
```bash
python -m fine_tuning.optimize_camera_map_extrinsics \
  --selected-samples gigaPose_datasets/results/real_world_data/combined_front_selected_samples.csv \
  --initial-extrinsic path/to/initial_T_map_cam_front.json \
  --initial-unit m \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-map-pose-unit auto \
  --translation-sigma-mm 1000 \
  --rotation-sigma-deg 10 \
  --translation-prior-weight 25 \
  --rotation-prior-weight 1 \
  --robust-loss soft_l1 \
  --output-dir gigaPose_datasets/results/real_world_data/extrinsic_optimization_front

```
Important: the candidate selection still uses T_camera_object_centered internally for camera-frame GigaPose-vs-EPnP agreement. But because the selected CSV keeps epnp_label_path, the optimizer can then open the same JSON and read T_map_object_raw


python -m fine_tuning.optimize_camera_map_extrinsics \
  --selected-samples gigaPose_datasets/results/real_world_data/combined_front_selected_samples.csv \
  --use-sample-metadata \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-map-pose-unit auto \
  --translation-sigma-mm 1000 \
  --rotation-sigma-deg 10 \
  --translation-prior-weight 25 \
  --rotation-prior-weight 1 \
  --robust-loss soft_l1 \
  --output-dir gigaPose_datasets/results/real_world_data/extrinsic_optimization_front_metadata


  After it runs, plot it with:
  ```bash
python -m fine_tuning.plot_extrinsic_optimization \
  --optimization-dir gigaPose_datasets/results/real_world_data/extrinsic_optimization_front_metadata_xy \
  --selected-samples gigaPose_datasets/results/real_world_data/combined_front_selected_samples.csv
  ```



It draws projected CAD boxes on the actual images:
red = map pose projected using original metadata extrinsic
blue = map pose projected using optimized extrinsic
green = selected GigaPose pose, optional
yellow = detector bbox, optional
  ```bash
python -m fine_tuning.visualize_extrinsic_optimization_on_images \
  --selected-samples gigaPose_datasets/results/real_world_data/combined_front_selected_samples.csv \
  --optimization-dir gigaPose_datasets/results/real_world_data/extrinsic_optimization_front_metadata_xy \
  --mesh gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
  --output-dir gigaPose_datasets/results/real_world_data/extrinsic_optimization_front_metadata_xy/image_overlays \
  --draw-gigapose \
  --draw-detection-bbox \
  --max-images 100

  ```
What it does for each selected sample:
Original:
T_cam_obj = inverse(t_map_lidar @ t_lidar_camera_prior) @ T_map_object_raw

Optimized:
T_cam_obj = inverse(t_map_lidar @ correction @ t_lidar_camera_prior) @ T_map_object_raw

Then it projects the CAD box into the image using the metadata camera intrinsics.