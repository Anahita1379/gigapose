Compare GigaPose predictions to the actual EPnPv2 label files:
<!-- 
for each camera seperately and for the predictions:
# 20260505v1 
front:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/front/EPnPv2_labels
rear:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/rear/EPnPv2_labels 
stereo_left:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/stereo_left/EPnPv2_labels 


# 20260505v2  
rear:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v2/rear/EPnPv2_gt_mesh_z_hybrid_labels


# 20260518v0 
front:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v0/front/EPnPv2_gt_mesh_z_hybrid_labels
stereo_left:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v0/stereo_left/EPnPv2_gt_mesh_z_labels

# 20260518v1v4 
front:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4/front/EPnPv2_gt_mesh_z_labels
rear:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4/rear/EPnPv2_gt_mesh_z_labels
stereo_left: /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4/stereo_left/EPnPv2_gt_mesh_z_labels

# 20260526 
None

20260518v2v4
front: /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v2-v4/front/EPnPv2_gt_mesh_z_labels
rear: /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v2-v4/rear/EPnPv2_gt_mesh_z_labels NOT AVAILABLE
stereo_left:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v2-v4/stereo_left/EPnPv2_gt_mesh_z_labels
 -->
 
```bash 
 python -m fine_tuning.select_real_label_candidates \
  --gigapose-predictions gigaPose_datasets/results/real_world_data/large_real_20260518v2v4_front_gsam_v4_finetuned/predictions/large-pbrreal-rgb-mmodel_real_20260518v2v4_front_gsam_v4-test_real_20260518v2v4_front_gsam_v4_finetunedMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v2v4_front_gsam_v4 \
  --epnp-root  /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v2-v4/front/EPnPv2_gt_mesh_z_labels \
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
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4/rear/EPnPv2_gt_mesh_z_labels \
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
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v2-v4/stereo_left/EPnPv2_gt_mesh_z_labels \
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