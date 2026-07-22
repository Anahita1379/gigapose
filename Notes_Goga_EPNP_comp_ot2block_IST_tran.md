Compare GigaPose predictions to the actual EPnPv2 label files:


for each camera seperately and for the predictions:
```bash 
------------------------------------------------------------------------------------------------
# 20260505v1 Done, need to do the other cameras
# front:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/front/EPnPv2_gt_mesh_z_hybrid_labels
# rear:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/rear/EPnPv2_gt_mesh_z_hybrid_labels
# stereo_left:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels
# gigapose/

 python -m fine_tuning.select_real_label_candidates \
  --gigapose-predictions gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260505v1_front_gsam_v4-test_large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_front_gsam_v4 \
  --epnp-root  /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/front/EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --frame-transform-side left \
  --frame-transform-refine-iterations 5 \
  --frame-transform-inlier-translation-mm 5000 \
  --frame-transform-inlier-rotation-deg 60 \
  --min-score 0.05 \
  --max-translation-error-mm 4000 \
  --max-rotation-error-deg 30 \
  --max-roll-error-deg 10 \
  --max-pitch-error-deg 10 \
  --max-yaw-error-deg 15 \
  --epnp-translation-unit m \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new_left


python -m fine_tuning.visualize_epnp_gigapose_comparison \
  --candidate-csv gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new_left/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_front_gsam_v4 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new_left/visual_overlays \
  --max-images 100 \
  --draw-mask-bbox


python -m fine_tuning.select_real_label_candidates \
  --gigapose-predictions gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_rear_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260505v1_rear_gsam_v4-test_large_real_20260505v1_rear_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_rear_gsam_v4 \
  --epnp-root  /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/rear/EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --frame-transform-side right \
  --frame-transform-refine-iterations 5 \
  --frame-transform-inlier-translation-mm 5000 \
  --frame-transform-inlier-rotation-deg 60 \
  --min-score 0.05 \
  --max-translation-error-mm 4000 \
  --max-rotation-error-deg 30 \
    --max-roll-error-deg 10 \
  --max-pitch-error-deg 10 \
  --max-yaw-error-deg 15 \
  --epnp-translation-unit m \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_rear_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new


python -m fine_tuning.visualize_epnp_gigapose_comparison \
  --candidate-csv gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_rear_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_rear_gsam_v4 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_rear_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/visual_overlays \
  --max-images 100 \
  --draw-mask-bbox


python -m fine_tuning.select_real_label_candidates \
  --gigapose-predictions gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260505v1_stereo_left_gsam_v4-test_large_real_20260505v1_stereo_left_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_stereo_left_gsam_v4 \
  --epnp-root  /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --frame-transform-side right \
  --frame-transform-refine-iterations 5 \
  --frame-transform-inlier-translation-mm 5000 \
  --frame-transform-inlier-rotation-deg 60 \
  --min-score 0.05 \
  --max-translation-error-mm 4000 \
  --max-rotation-error-deg 30 \
    --max-roll-error-deg 10 \
  --max-pitch-error-deg 10 \
  --max-yaw-error-deg 15 \
  --epnp-translation-unit m \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new


python -m fine_tuning.visualize_epnp_gigapose_comparison \
  --candidate-csv gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_stereo_left_gsam_v4 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/visual_overlays \
  --max-images 100 \
  --draw-mask-bbox

------------------------------------------------------------------------------------------------
# 20260505v2  
# rear:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v2/rear/EPnPv2_gt_mesh_z_hybrid_labels


  python -m fine_tuning.select_real_label_candidates \
  --gigapose-predictions gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v2_rear_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260505v2_rear_gsam_v4-test_large_real_20260505v2_rear_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v2_rear_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v2/rear/EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --frame-transform-side right \
  --frame-transform-refine-iterations 5 \
  --frame-transform-inlier-translation-mm 5000 \
  --frame-transform-inlier-rotation-deg 60 \
  --min-score 0.05 \
  --max-translation-error-mm 4000 \
  --max-rotation-error-deg 30 \
  --max-roll-error-deg 10 \
  --max-pitch-error-deg 10 \
  --max-yaw-error-deg 15 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v2_rear_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new



python -m fine_tuning.visualize_epnp_gigapose_comparison \
  --candidate-csv gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v2_rear_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v2_rear_gsam_v4 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v2_rear_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/visual_overlays \
  --max-images 100 \
  --draw-mask-bbox
-------------------------------------------------------------------------------------------------
# 20260518v0  
# front:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v0/front/EPnPv2_gt_mesh_z_hybrid_labels (No need to rerun)
# stereo_left:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v0/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels 

  python -m fine_tuning.select_real_label_candidates \
  --gigapose-predictions gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v0_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v0_front_gsam_v4-test_large_real_20260518v0_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v0_front_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v0/front/EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --frame-transform-side right \
  --frame-transform-refine-iterations 5 \
  --frame-transform-inlier-translation-mm 5000 \
  --frame-transform-inlier-rotation-deg 60 \
  --min-score 0.05 \
  --max-translation-error-mm 4000 \
  --max-rotation-error-deg 30 \
  --max-roll-error-deg 10 \
  --max-pitch-error-deg 10 \
  --max-yaw-error-deg 15 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v0_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new


python -m fine_tuning.visualize_epnp_gigapose_comparison \
  --candidate-csv gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v0_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v0_front_gsam_v4 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v0_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/visual_overlays \
  --max-images 100 \
  --draw-mask-bbox


  python -m fine_tuning.select_real_label_candidates \
  --gigapose-predictions gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v0_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v0_stereo_left_gsam_v4-test_large_real_20260518v0_stereo_left_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v0_stereo_left_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v0/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels  \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --frame-transform-side right \
  --frame-transform-refine-iterations 5 \
  --frame-transform-inlier-translation-mm 5000 \
  --frame-transform-inlier-rotation-deg 60 \
  --min-score 0.05 \
  --max-translation-error-mm 4000 \
  --max-rotation-error-deg 30 \
  --max-roll-error-deg 10 \
  --max-pitch-error-deg 10 \
  --max-yaw-error-deg 15 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v0_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new


python -m fine_tuning.visualize_epnp_gigapose_comparison \
  --candidate-csv gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v0_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v0_stereo_left_gsam_v4 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v0_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/visual_overlays \
  --max-images 100 \
  --draw-mask-bbox


------------------------------------------------------------------------------
# 20260518v1v4  
# front:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4/front/EPnPv2_gt_mesh_z_hybrid_labels
# rear:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4/rear/EPnPv2_gt_mesh_z_hybrid_labels
# stereo_left: /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels

python -m fine_tuning.select_real_label_candidates \
  --gigapose-predictions gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v1v4_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v1v4_front_gsam_v4-test_large_real_20260518v1v4_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_front_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4/front/EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --frame-transform-side right \
  --frame-transform-refine-iterations 5 \
  --frame-transform-inlier-translation-mm 5000 \
  --frame-transform-inlier-rotation-deg 60 \
  --min-score 0.05 \
  --max-translation-error-mm 4000 \
  --max-rotation-error-deg 30 \
  --max-roll-error-deg 10 \
  --max-pitch-error-deg 10 \
  --max-yaw-error-deg 15 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v1v4_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new



python -m fine_tuning.visualize_epnp_gigapose_comparison \
  --candidate-csv gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v1v4_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_front_gsam_v4 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v1v4_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/visual_overlays \
  --max-images 100 \
  --draw-mask-bbox


python -m fine_tuning.select_real_label_candidates \
  --gigapose-predictions gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v1v4_rear_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v1v4_rear_gsam_v4-test_large_real_20260518v1v4_rear_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_rear_gsam_v4 \
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
  --max-translation-error-mm 4000 \
  --max-rotation-error-deg 30 \
  --max-roll-error-deg 10 \
  --max-pitch-error-deg 10 \
  --max-yaw-error-deg 15 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v1v4_rear_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new

python -m fine_tuning.visualize_epnp_gigapose_comparison \
  --candidate-csv gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v1v4_rear_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_rear_gsam_v4 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v1v4_rear_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/visual_overlays \
  --max-images 100 \
  --draw-mask-bbox

python -m fine_tuning.select_real_label_candidates \
  --gigapose-predictions gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v1v4_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v1v4_stereo_left_gsam_v4-test_large_real_20260518v1v4_stereo_left_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_stereo_left_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --frame-transform-side right \
  --frame-transform-refine-iterations 5 \
  --frame-transform-inlier-translation-mm 5000 \
  --frame-transform-inlier-rotation-deg 60 \
  --min-score 0.05 \
  --max-translation-error-mm 4000 \
  --max-rotation-error-deg 30 \
  --max-roll-error-deg 10 \
  --max-pitch-error-deg 10 \
  --max-yaw-error-deg 15 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v1v4_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new



python -m fine_tuning.visualize_epnp_gigapose_comparison \
  --candidate-csv gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v1v4_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_stereo_left_gsam_v4 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v1v4_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/visual_overlays \
  --max-images 100 \
  --draw-mask-bbox

------------------------------------------------------------------------------
# 20260526 
# None


------------------------------------------------------------------------------
# 20260518v2v4 
# front: /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v2-v4/front/EPnPv2_gt_mesh_z_hybrid_labels
# stereo_left:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v2-v4/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels

  python -m fine_tuning.select_real_label_candidates \
  --gigapose-predictions gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v2v4_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v2v4_front_gsam_v4-test_large_real_20260518v2v4_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v2v4_front_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v2-v4/front/EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --frame-transform-side right \
  --frame-transform-refine-iterations 5 \
  --frame-transform-inlier-translation-mm 5000 \
  --frame-transform-inlier-rotation-deg 60 \
  --min-score 0.05 \
  --max-translation-error-mm 4000 \
  --max-rotation-error-deg 30 \
  --max-roll-error-deg 10 \
  --max-pitch-error-deg 10 \
  --max-yaw-error-deg 15 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v2v4_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new


python -m fine_tuning.visualize_epnp_gigapose_comparison \
  --candidate-csv gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v2v4_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v2v4_front_gsam_v4 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v2v4_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/visual_overlays \
  --max-images 100 \
  --draw-mask-bbox



   python -m fine_tuning.select_real_label_candidates \
  --gigapose-predictions gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v2v4_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v2v4_stereo_left_gsam_v4-test_large_real_20260518v2v4_stereo_left_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
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
  --max-translation-error-mm 4000 \
  --max-rotation-error-deg 30 \
  --max-roll-error-deg 10 \
  --max-pitch-error-deg 10 \
  --max-yaw-error-deg 15 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v2v4_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new




python -m fine_tuning.visualize_epnp_gigapose_comparison \
  --candidate-csv gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v2v4_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v2v4_stereo_left_gsam_v4 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v2v4_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/visual_overlays \
  --max-images 100 \
  --draw-mask-bbox

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
20260505v1
selection_report.json
Summary like number of candidate pairs, number selected, median/mean/p90 translation and rotation error.

To plot the numeric comparison:
```bash 
python -m fine_tuning.plot_epnp_gigapose_comparison \
  --input-csv gigaPose_datasets/results/real_world_data_IST_AE/large_real_20260505v1_front_gsam_v4_finetuned_AE_IST/label_candidates_refined/best_candidate_per_epnp_label.csv

python -m fine_tuning.plot_epnp_gigapose_comparison \
  --input-csv gigaPose_datasets/results/real_world_data_IST_AE/large_real_20260505v1_rear_gsam_v4_finetuned_AE_IST/label_candidates_refined/best_candidate_per_epnp_label.csv

python -m fine_tuning.plot_epnp_gigapose_comparison \
  --input-csv gigaPose_datasets/results/real_world_data_IST_AE/large_real_20260505v1_stereo_left_gsam_v4_finetuned_AE_IST/label_candidates_refined/best_candidate_per_epnp_label.csv
```
This writes plots to:
gigaPose_datasets/results/real_20260505_front_gsam_v4_label_candidates/plots_epnp_gigapose/

To visualize EPnPv2 vs GigaPose on the actual images:


gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_left/selected_samples.csv

gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_right/selected_samples.csv
```bash 
python -m fine_tuning.visualize_epnp_gigapose_comparison \
  --candidate-csv gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_right/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_front_gsam_v4 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_right/visual_overlays \
  --max-images 100 \
  --draw-mask-bbox


--output-dir gigaPose_datasets/results/real_world_data_IST_AE/large_real_20260505v1_front_gsam_v4_finetuned_AE_IST/label_candidates_refined_left


python -m fine_tuning.visualize_epnp_gigapose_comparison \
  --candidate-csv gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_rear_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_right/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_rear_gsam_v4 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_rear_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_right/visual_overlays \
  --max-images 100 \
  --draw-mask-bbox


  python -m fine_tuning.visualize_epnp_gigapose_comparison \
  --candidate-csv gigaPose_datasets/results/real_world_data_IST_AE/large_real_20260505v1_stereo_left_gsam_v4_finetuned_AE_IST/label_candidates_refined/best_candidate_per_epnp_label.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_stereo_left_gsam_v4 \
  --output-dir gigaPose_datasets/results/real_world_data_IST_AE/large_real_20260505v1_stereo_left_gsam_v4_finetuned_AE_IST/label_candidates_refined/visual_overlays \
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

selected = "gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v2v4_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization/selected_samples.csv"

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
gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/selected_samples.csv

python -m fine_tuning.combine_selected_samples \
  --input gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/selected_samples.csv \
  --input gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v0_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/selected_samples.csv \
  --input gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v1v4_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/selected_samples.csv \
  --input gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v2v4_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/selected_samples.csv \
  --output gigaPose_datasets/results/real_world_ot2_IST_tran/combined_front_selected_samples_for_optimization_new.csv \
  --dedupe-by match_key_epnp \
  --keep lowest-error


For only rear: 
gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_rear_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new

python -m fine_tuning.combine_selected_samples \
  --input gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_rear_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/selected_samples.csv \
  --input gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v2_rear_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/selected_samples.csv \
  --input gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v1v4_rear_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/selected_samples.csv \
  --output gigaPose_datasets/results/real_world_ot2_IST_tran/combined_rear_selected_samples_for_optimization_new.csv \
  --dedupe-by match_key_epnp \
  --keep lowest-error


For only stereo_left: 
gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new

python -m fine_tuning.combine_selected_samples \
  --input gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/selected_samples.csv \
  --input gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v0_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/selected_samples.csv \
  --input gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v1v4_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/selected_samples.csv \
  --input gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v2v4_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/selected_samples.csv \
  --output gigaPose_datasets/results/real_world_ot2_IST_tran/combined_stereo_left_selected_samples_for_optimization_new.csv \
  --dedupe-by match_key_epnp \
  --keep lowest-error
```




python optimize_extrinsics_updated.py \
    --selected-samples path/to/selected_samples.csv \
    --use-sample-metadata \
    --epnp-map-pose-key T_map_object_raw \
    --translation-prior-weight 0 \
    --rotation-prior-weight 0 \
    --robust-loss soft_l1 \
    --output-dir path/to/results


python optimize_extrinsics.py \
    --selected-samples path/to/selected_samples.csv \
    --use-sample-metadata \
    --epnp-map-pose-key T_map_object_raw \
    --translation-prior-weight 0 \
    --rotation-prior-weight 0 \
    --max-samples 10 \
    --output-dir path/to/test_output

recommanded optimizer code: 
```bash
# for front
python -m fine_tuning.optimize_camera_map_extrinsics \
  --selected-samples gigaPose_datasets/results/real_world_ot2_IST_tran/combined_front_selected_samples_for_optimization_new.csv \
  --use-epnp-label-extrinsics \
  --epnp-label-dir-name EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-map-pose-unit auto \
  --epnp-camera-pose-key T_camera_object_centered \
  --epnp-camera-pose-unit auto \
  --translation-residual-components xyz \
  --translation-sigma-mm 1000 \
  --rotation-sigma-deg 10 \
  --image-center-weight 5 \
  --image-center-sigma-px 50 \
  --projection-model metadata \
  --translation-prior-weight 1000 \
  --rotation-prior-weight 20 \
  --robust-loss soft_l1 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/extrinsic_optimization_front_epnp_hybrid_xyz_new


  python optimize_extrinsics_updated.py \
    --selected-samples path/to/selected_samples.csv \
    --use-sample-metadata \
    --epnp-map-pose-key T_map_object_raw \
    --translation-prior-weight 0 \
    --rotation-prior-weight 0 \
    --robust-loss soft_l1 \
    --output-dir path/to/results

# for rear
python -m fine_tuning.optimize_camera_map_extrinsics \
  --selected-samples gigaPose_datasets/results/real_world_ot2_IST_tran/combined_rear_selected_samples_for_optimization_new.csv \
  --use-epnp-label-extrinsics \
  --epnp-label-dir-name EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-map-pose-unit auto \
  --epnp-camera-pose-key T_camera_object_centered \
  --epnp-camera-pose-unit auto \
  --translation-residual-components xyz \
  --translation-sigma-mm 1000 \
  --rotation-sigma-deg 10 \
  --image-center-weight 5 \
  --image-center-sigma-px 50 \
  --projection-model metadata \
  --translation-prior-weight 1000 \
  --rotation-prior-weight 20 \
  --robust-loss soft_l1 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/extrinsic_optimization_rear_epnp_hybrid_xyz_new


# for Stereo_left
python -m fine_tuning.optimize_camera_map_extrinsics \
  --selected-samples gigaPose_datasets/results/real_world_ot2_IST_tran/combined_stereo_left_selected_samples_for_optimization_new.csv \
  --use-epnp-label-extrinsics \
  --epnp-label-dir-name EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-map-pose-unit auto \
  --epnp-camera-pose-key T_camera_object_centered \
  --epnp-camera-pose-unit auto \
  --translation-residual-components xyz \
  --translation-sigma-mm 1000 \
  --rotation-sigma-deg 10 \
  --image-center-weight 5 \
  --image-center-sigma-px 50 \
  --projection-model metadata \
  --translation-prior-weight 1000 \
  --rotation-prior-weight 20 \
  --robust-loss soft_l1 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/extrinsic_optimization_stereo_left_epnp_hybrid_xyz_new

```
  After it runs, plot it with:
  ```bash
python -m fine_tuning.plot_extrinsic_optimization \
  --optimization-dir gigaPose_datasets/results/real_world_data/extrinsic_optimization_stereo_left_metadata_xy_image \
  --selected-samples gigaPose_datasets/results/real_world_data/combined_stereo_left_selected_samples.csv
  ```

It draws projected CAD boxes on the actual images:
red = map pose projected using original metadata extrinsic
blue = map pose projected using optimized extrinsic
green = selected GigaPose pose, optional
yellow = detector bbox, optional

For visualization of that new run:
Key point: for this EPnP-label-only mode, use: 
--map-z-mode raw
```bash
# for front
python -m fine_tuning.visualize_extrinsic_optimization_on_images \
  --selected-samples gigaPose_datasets/results/real_world_ot2_IST_tran/combined_front_selected_samples_for_optimization_new.csv  \
  --optimization-dir gigaPose_datasets/results/real_world_ot2_IST_tran/extrinsic_optimization_front_epnp_hybrid_xyz_new \
  --mesh gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/extrinsic_optimization_front_epnp_hybrid_xyz_new \
  --epnp-label-dir-name EPnPv2_gt_mesh_z_hybrid_labels \
  --map-z-mode raw \
  --draw-gigapose \
  --draw-detection-bbox \
  --write-debug-projections \
  --projection-model metadata \
  --max-images 200



# for rear
  python -m fine_tuning.visualize_extrinsic_optimization_on_images \
  --selected-samples gigaPose_datasets/results/real_world_ot2_IST_tran/combined_rear_selected_samples_for_optimization_new.csv  \
  --optimization-dir gigaPose_datasets/results/real_world_ot2_IST_tran/extrinsic_optimization_rear_epnp_hybrid_xyz_new \
  --mesh gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/extrinsic_optimization_rear_epnp_hybrid_xyz_new \
  --epnp-label-dir-name EPnPv2_gt_mesh_z_hybrid_labels \
  --map-z-mode raw \
  --draw-gigapose \
  --draw-detection-bbox \
  --write-debug-projections \
  --projection-model metadata \
  --max-images 200



# for stereo_left
python -m fine_tuning.visualize_extrinsic_optimization_on_images \
  --selected-samples gigaPose_datasets/results/real_world_ot2_IST_tran/combined_stereo_left_selected_samples_for_optimization_new.csv  \
  --optimization-dir gigaPose_datasets/results/real_world_ot2_IST_tran/extrinsic_optimization_stereo_left_epnp_hybrid_xyz_new \
  --mesh gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/extrinsic_optimization_stereo_left_epnp_hybrid_xyz_new \
  --epnp-label-dir-name EPnPv2_gt_mesh_z_hybrid_labels \
  --map-z-mode raw \
  --draw-gigapose \
  --draw-detection-bbox \
  --write-debug-projections \
  --projection-model metadata \
  --max-images 200

```


we can plot the errors:
```bash
python -m fine_tuning.plot_extrinsic_optimization \
  --optimization-dir gigaPose_datasets/results/real_world_data_IST_AE/extrinsic_optimization_stereo_left_epnp_hybrid_xyz_stronger \
  --output-dir gigaPose_datasets/results/real_world_data_IST_AE/extrinsic_optimization_stereo_left_epnp_hybrid_xyz_stronger/plots \
  --selected-samples gigaPose_datasets/results/real_world_data_IST_AE/combined_front_selected_samples.csv
```

  
Important: the candidate selection still uses T_camera_object_centered internally for camera-frame GigaPose-vs-EPnP agreement. But because the selected CSV keeps epnp_label_path, the optimizer can then open the same JSON and read T_map_object_raw


After running and getting the optimized entrinsics, we rerun the lable selection code, 
but this time, with the optimized extrinsic values: 


I guess we need to rerun the optimization:
```bash
python -m fine_tuning.optimize_camera_map_extrinsics_updated \
  --selected-samples gigaPose_datasets/results/real_world_ot2_IST_tran/combined_front_selected_samples_for_optimization_new.csv \
  --gigapose-pose-source aligned \
  --use-epnp-label-extrinsics \
  --epnp-label-dir-name EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-map-pose-unit m \
  --epnp-camera-pose-key T_camera_object_centered \
  --epnp-camera-pose-unit m \
  --translation-residual-components xyz \
  --translation-sigma-mm 1000 \
  --rotation-sigma-deg 10 \
  --image-center-weight 5 \
  --image-center-sigma-px 50 \
  --projection-model metadata \
  --image-center-weight 0 \
  --translation-prior-weight 10000 \
  --rotation-prior-weight 100 \
  --robust-loss soft_l1 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/extrinsic_optimization_front_epnp_hybrid_xyz_fixed

python -m fine_tuning.optimize_camera_map_extrinsics_updated_new \
    --selected-samples gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/combined_front_selected_samples_for_optimization_new.csv \
    --use-sample-metadata \
    --epnp-map-pose-key T_map_object_raw \
    --translation-prior-weight 0 \
    --rotation-prior-weight 0 \
    --robust-loss soft_l1 \
    --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/extrinsic_optimization_front_updated



python -m fine_tuning.optimize_camera_map_extrinsics_updated_new \
    --selected-samples gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/combined_front_selected_samples_for_optimization_new.csv
  --gigapose-pose-source aligned \
  --use-sample-metadata \
  --epnp-label-dir-name EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-map-pose-unit m \
  --translation-residual-components xy \
  --translation-sigma-mm 1000 \
  --rotation-sigma-deg 10 \
  --image-center-weight 5 \
  --image-center-sigma-px 50 \
  --image-center-map-z-mode session_lidar_offset \
  --projection-model metadata \
  --translation-prior-weight 5000 \
  --rotation-prior-weight 100 \
  --robust-loss soft_l1 \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/extrinsic_optimization_front_metadata_xy



python -m fine_tuning.optimize_camera_map_extrinsics_updated \
  --selected-samples gigaPose_datasets/results/real_world_ot2_IST_tran/combined_rear_selected_samples_for_optimization_new.csv \
  --gigapose-pose-source aligned \
  --use-epnp-label-extrinsics \
  --epnp-label-dir-name EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-map-pose-unit m \
  --epnp-camera-pose-key T_camera_object_centered \
  --epnp-camera-pose-unit m \
  --translation-residual-components xyz \
  --translation-sigma-mm 1000 \
  --rotation-sigma-deg 10 \
  --image-center-weight 5 \
  --image-center-sigma-px 50 \
  --projection-model metadata \
  --image-center-weight 0 \
  --translation-prior-weight 10000 \
  --rotation-prior-weight 100 \
  --robust-loss soft_l1 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/extrinsic_optimization_rear_epnp_hybrid_xyz_fixed



python -m fine_tuning.optimize_camera_map_extrinsics_updated \
  --selected-samples gigaPose_datasets/results/real_world_ot2_IST_tran/combined_stereo_left_selected_samples_for_optimization_new.csv \
  --gigapose-pose-source aligned \
  --use-epnp-label-extrinsics \
  --epnp-label-dir-name EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-map-pose-unit m \
  --epnp-camera-pose-key T_camera_object_centered \
  --epnp-camera-pose-unit m \
  --translation-residual-components xyz \
  --translation-sigma-mm 1000 \
  --rotation-sigma-deg 10 \
  --image-center-weight 5 \
  --image-center-sigma-px 50 \
  --projection-model metadata \
  --image-center-weight 0 \
  --translation-prior-weight 10000 \
  --rotation-prior-weight 100 \
  --robust-loss soft_l1 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/extrinsic_optimization_stereo_left_epnp_hybrid_xyz_fixed


  python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv <NEW_LABEL_CANDIDATE_DIR>/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_front_gsam_v4 \
  --output-dir <NEW_LABEL_CANDIDATE_DIR>/visual_overlays_native_cad \
  --frame-transform-side right \
  --max-images 100 \
  --draw-mask-bbox
  

   new one: 
   
# --gigapose-pose-source raw \
python -m fine_tuning.optimize_camera_lidar_extrinsics \
  --selected-samples gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/combined_front_selected_samples_for_optimization_new.csv \
  --gigapose-pose-source aligned \
  --use-sample-metadata \
  --epnp-label-dir-name EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-map-pose-unit m \
  --translation-residual-components xy \
  --translation-sigma-mm 1000 \
  --rotation-sigma-deg 10 \
  --image-center-weight 5 \
  --image-center-sigma-px 50 \
  --image-center-map-z-mode session_lidar_offset \
  --projection-model metadata \
  --translation-prior-weight 5000 \
  --rotation-prior-weight 100 \
  --robust-loss soft_l1 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/camera_lidar_extrinsics_front










 

```
----------------------------------------------------
Done so far, need to rerun the selection using the new extrinsics and then visulaize it to check if correct

```bash

# 20260505v1 Done, need to do the other cameras
# front:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/front/EPnPv2_gt_mesh_z_hybrid_labels
# rear:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/rear/EPnPv2_gt_mesh_z_hybrid_labels
# stereo_left:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels
# gigapose/

python -m fine_tuning.select_real_label_candidates_with_extrinsics \
  --gigapose-predictions gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260505v1_front_gsam_v4-test_large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_front_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/front/EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --optimized-extrinsics gigaPose_datasets/results/real_world_ot2_IST_tran/extrinsic_optimization_front_epnp_hybrid_xyz_fixed/optimized_extrinsics.json \
  --frame-transform-side right \
  --frame-transform-refine-iterations 5 \
  --frame-transform-inlier-translation-mm 5000 \
  --frame-transform-inlier-rotation-deg 60 \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-camera-pose-key T_camera_object_centered \
  --epnp-map-pose-unit m \
  --epnp-camera-pose-unit m \
  --min-score 0.05 \
  --max-translation-error-mm 2500 \
  --max-rotation-error-deg 30 \
  --max-roll-error-deg 10 \
  --max-pitch-error-deg 10 \
  --max-yaw-error-deg 20 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed

gigapose/gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_new/selected_samples.csv


python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_front_gsam_v4 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed/visual_overlays_native_cad \
  --frame-transform-side right \
  --max-images 200 \
  --draw-mask-bbox

# -------------

python -m fine_tuning.select_real_label_candidates_with_extrinsics \
  --gigapose-predictions gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_rear_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260505v1_rear_gsam_v4-test_large_real_20260505v1_rear_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_rear_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/rear/EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --optimized-extrinsics gigaPose_datasets/results/real_world_ot2_IST_tran/extrinsic_optimization_rear_epnp_hybrid_xyz_fixed/optimized_extrinsics.json \
  --frame-transform-side right \
  --frame-transform-refine-iterations 5 \
  --frame-transform-inlier-translation-mm 5000 \
  --frame-transform-inlier-rotation-deg 60 \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-camera-pose-key T_camera_object_centered \
  --epnp-map-pose-unit m \
  --epnp-camera-pose-unit m \
  --min-score 0.05 \
  --max-translation-error-mm 2500 \
  --max-rotation-error-deg 30 \
  --max-roll-error-deg 10 \
  --max-pitch-error-deg 10 \
  --max-yaw-error-deg 20 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_rear_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed


python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_rear_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_rear_gsam_v4 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_rear_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed/visual_overlays_native_cad \
  --frame-transform-side right \
  --max-images 100 \
  --draw-mask-bbox

# -------------
python -m fine_tuning.select_real_label_candidates_with_extrinsics \
  --gigapose-predictions gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260505v1_stereo_left_gsam_v4-test_large_real_20260505v1_stereo_left_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_stereo_left_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --optimized-extrinsics gigaPose_datasets/results/real_world_ot2_IST_tran/extrinsic_optimization_stereo_left_epnp_hybrid_xyz_fixed/optimized_extrinsics.json \
  --frame-transform-side right \
  --frame-transform-refine-iterations 5 \
  --frame-transform-inlier-translation-mm 5000 \
  --frame-transform-inlier-rotation-deg 60 \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-camera-pose-key T_camera_object_centered \
  --epnp-map-pose-unit m \
  --epnp-camera-pose-unit m \
  --min-score 0.05 \
  --max-translation-error-mm 2500 \
  --max-rotation-error-deg 30 \
  --max-roll-error-deg 10 \
  --max-pitch-error-deg 10 \
  --max-yaw-error-deg 20 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed


python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_stereo_left_gsam_v4 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed/visual_overlays_native_cad \
  --frame-transform-side right \
  --max-images 100 \
  --draw-mask-bbox


---------------------------------------------------------------------

# 20260505v2  
# rear:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v2/rear/EPnPv2_gt_mesh_z_hybrid_labels

python -m fine_tuning.select_real_label_candidates_with_extrinsics \
  --gigapose-predictions gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v2_rear_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260505v2_rear_gsam_v4-test_large_real_20260505v2_rear_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v2_rear_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v2/rear/EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --optimized-extrinsics gigaPose_datasets/results/real_world_ot2_IST_tran/extrinsic_optimization_rear_epnp_hybrid_xyz_fixed/optimized_extrinsics.json \
  --frame-transform-side right \
  --frame-transform-refine-iterations 5 \
  --frame-transform-inlier-translation-mm 5000 \
  --frame-transform-inlier-rotation-deg 60 \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-camera-pose-key T_camera_object_centered \
  --epnp-map-pose-unit m \
  --epnp-camera-pose-unit m \
  --min-score 0.05 \
  --max-translation-error-mm 2500 \
  --max-rotation-error-deg 30 \
  --max-roll-error-deg 10 \
  --max-pitch-error-deg 10 \
  --max-yaw-error-deg 20 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v2_rear_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed


python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v2_rear_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v2_rear_gsam_v4 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260505v2_rear_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed/visual_overlays_native_cad \
  --frame-transform-side right \
  --max-images 100 \
  --draw-mask-bbox

-------------------------------------------------------------------------------------------------
# 20260518v0  
# front:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v0/front/EPnPv2_gt_mesh_z_hybrid_labels (No need to rerun)
# stereo_left:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v0/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels 




python -m fine_tuning.select_real_label_candidates_with_extrinsics \
  --gigapose-predictions gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v0_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v0_front_gsam_v4-test_large_real_20260518v0_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v0_front_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v0/front/EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --optimized-extrinsics gigaPose_datasets/results/real_world_ot2_IST_tran/extrinsic_optimization_front_epnp_hybrid_xyz_fixed/optimized_extrinsics.json \
  --frame-transform-side right \
  --frame-transform-refine-iterations 5 \
  --frame-transform-inlier-translation-mm 5000 \
  --frame-transform-inlier-rotation-deg 60 \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-camera-pose-key T_camera_object_centered \
  --epnp-map-pose-unit m \
  --epnp-camera-pose-unit m \
  --min-score 0.05 \
  --max-translation-error-mm 2500 \
  --max-rotation-error-deg 30 \
  --max-roll-error-deg 10 \
  --max-pitch-error-deg 10 \
  --max-yaw-error-deg 20 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v0_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed


python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v0_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v0_front_gsam_v4 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v0_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed/visual_overlays_native_cad \
  --frame-transform-side right \
  --max-images 100 \
  --draw-mask-bbox


python -m fine_tuning.select_real_label_candidates_with_extrinsics \
  --gigapose-predictions gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v0_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v0_stereo_left_gsam_v4-test_large_real_20260518v0_stereo_left_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v0_stereo_left_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v0/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --optimized-extrinsics gigaPose_datasets/results/real_world_ot2_IST_tran/extrinsic_optimization_stereo_left_epnp_hybrid_xyz_fixed/optimized_extrinsics.json \
  --frame-transform-side right \
  --frame-transform-refine-iterations 5 \
  --frame-transform-inlier-translation-mm 5000 \
  --frame-transform-inlier-rotation-deg 60 \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-camera-pose-key T_camera_object_centered \
  --epnp-map-pose-unit m \
  --epnp-camera-pose-unit m \
  --min-score 0.05 \
  --max-translation-error-mm 2500 \
  --max-rotation-error-deg 30 \
  --max-roll-error-deg 10 \
  --max-pitch-error-deg 10 \
  --max-yaw-error-deg 20 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v0_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed


python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v0_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v0_stereo_left_gsam_v4 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v0_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed/visual_overlays_native_cad \
  --frame-transform-side right \
  --max-images 100 \
  --draw-mask-bbox


------------------------------------------------------------------------------
# 20260518v1v4  
# front:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4/front/EPnPv2_gt_mesh_z_hybrid_labels
# rear:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4/rear/EPnPv2_gt_mesh_z_hybrid_labels
# stereo_left: /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels

python -m fine_tuning.select_real_label_candidates_with_extrinsics \
  --gigapose-predictions gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v1v4_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v1v4_front_gsam_v4-test_large_real_20260518v1v4_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_front_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4/front/EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --optimized-extrinsics gigaPose_datasets/results/real_world_ot2_IST_tran/extrinsic_optimization_front_epnp_hybrid_xyz_fixed/optimized_extrinsics.json \
  --frame-transform-side right \
  --frame-transform-refine-iterations 5 \
  --frame-transform-inlier-translation-mm 5000 \
  --frame-transform-inlier-rotation-deg 60 \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-camera-pose-key T_camera_object_centered \
  --epnp-map-pose-unit m \
  --epnp-camera-pose-unit m \
  --min-score 0.05 \
  --max-translation-error-mm 2500 \
  --max-rotation-error-deg 30 \
  --max-roll-error-deg 10 \
  --max-pitch-error-deg 10 \
  --max-yaw-error-deg 20 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v1v4_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed


python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v1v4_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_front_gsam_v4 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v1v4_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed/visual_overlays_native_cad \
  --frame-transform-side right \
  --max-images 100 \
  --draw-mask-bbox

# ----------

python -m fine_tuning.select_real_label_candidates_with_extrinsics \
  --gigapose-predictions gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v1v4_rear_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v1v4_rear_gsam_v4-test_large_real_20260518v1v4_rear_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_rear_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4/rear/EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --optimized-extrinsics gigaPose_datasets/results/real_world_ot2_IST_tran/extrinsic_optimization_rear_epnp_hybrid_xyz_fixed/optimized_extrinsics.json \
  --frame-transform-side right \
  --frame-transform-refine-iterations 5 \
  --frame-transform-inlier-translation-mm 5000 \
  --frame-transform-inlier-rotation-deg 60 \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-camera-pose-key T_camera_object_centered \
  --epnp-map-pose-unit m \
  --epnp-camera-pose-unit m \
  --min-score 0.05 \
  --max-translation-error-mm 2500 \
  --max-rotation-error-deg 30 \
  --max-roll-error-deg 10 \
  --max-pitch-error-deg 10 \
  --max-yaw-error-deg 20 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v1v4_rear_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed

python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v1v4_rear_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_rear_gsam_v4 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v1v4_rear_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed/visual_overlays_native_cad \
  --frame-transform-side right \
  --max-images 100 \
  --draw-mask-bbox


# ------------------------------
python -m fine_tuning.select_real_label_candidates_with_extrinsics \
  --gigapose-predictions gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v1v4_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v1v4_stereo_left_gsam_v4-test_large_real_20260518v1v4_stereo_left_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_stereo_left_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --optimized-extrinsics gigaPose_datasets/results/real_world_ot2_IST_tran/extrinsic_optimization_stereo_left_epnp_hybrid_xyz_fixed/optimized_extrinsics.json \
  --frame-transform-side right \
  --frame-transform-refine-iterations 5 \
  --frame-transform-inlier-translation-mm 5000 \
  --frame-transform-inlier-rotation-deg 60 \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-camera-pose-key T_camera_object_centered \
  --epnp-map-pose-unit m \
  --epnp-camera-pose-unit m \
  --min-score 0.05 \
  --max-translation-error-mm 2500 \
  --max-rotation-error-deg 30 \
  --max-roll-error-deg 10 \
  --max-pitch-error-deg 10 \
  --max-yaw-error-deg 20 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v1v4_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed



python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v1v4_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_stereo_left_gsam_v4 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v1v4_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed/visual_overlays_native_cad \
  --frame-transform-side right \
  --max-images 100 \
  --draw-mask-bbox

------------------------------------------------------------------------------
# 20260526 
# None


------------------------------------------------------------------------------
# 20260518v2v4 
# front: /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v2-v4/front/EPnPv2_gt_mesh_z_hybrid_labels
# stereo_left:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v2-v4/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels




python -m fine_tuning.select_real_label_candidates_with_extrinsics \
  --gigapose-predictions gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v2v4_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v2v4_front_gsam_v4-test_large_real_20260518v2v4_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v2v4_front_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v2-v4/front/EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --optimized-extrinsics gigaPose_datasets/results/real_world_ot2_IST_tran/extrinsic_optimization_front_epnp_hybrid_xyz_fixed/optimized_extrinsics.json \
  --frame-transform-side right \
  --frame-transform-refine-iterations 5 \
  --frame-transform-inlier-translation-mm 5000 \
  --frame-transform-inlier-rotation-deg 60 \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-camera-pose-key T_camera_object_centered \
  --epnp-map-pose-unit m \
  --epnp-camera-pose-unit m \
  --min-score 0.05 \
  --max-translation-error-mm 2500 \
  --max-rotation-error-deg 30 \
  --max-roll-error-deg 10 \
  --max-pitch-error-deg 10 \
  --max-yaw-error-deg 20 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v2v4_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed


python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v2v4_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v2v4_front_gsam_v4 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v2v4_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed/visual_overlays_native_cad \
  --frame-transform-side right \
  --max-images 100 \
  --draw-mask-bbox



python -m fine_tuning.select_real_label_candidates_with_extrinsics \
  --gigapose-predictions gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v2v4_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v2v4_stereo_left_gsam_v4-test_large_real_20260518v2v4_stereo_left_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v2v4_stereo_left_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v2-v4/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --optimized-extrinsics gigaPose_datasets/results/real_world_ot2_IST_tran/extrinsic_optimization_stereo_left_epnp_hybrid_xyz_fixed/optimized_extrinsics.json \
  --frame-transform-side right \
  --frame-transform-refine-iterations 5 \
  --frame-transform-inlier-translation-mm 5000 \
  --frame-transform-inlier-rotation-deg 60 \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-camera-pose-key T_camera_object_centered \
  --epnp-map-pose-unit m \
  --epnp-camera-pose-unit m \
  --min-score 0.05 \
  --max-translation-error-mm 2500 \
  --max-rotation-error-deg 30 \
  --max-roll-error-deg 10 \
  --max-pitch-error-deg 10 \
  --max-yaw-error-deg 20 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v2v4_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed




python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v2v4_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v2v4_stereo_left_gsam_v4 \
  --output-dir gigaPose_datasets/results/real_world_ot2_IST_tran/large_real_20260518v2v4_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_with_optimized_extrinsics_fixed/visual_overlays_centered_cad \
  --frame-transform-side right \
  --max-images 100 \
  --draw-mask-bbox

# That confirms the green-box problem was caused by the visualizer’s incorrect assumption. I fixed only [visualize_epnp_gigapose_comparison_extrinsics.py (line 85)](/home/anahita/gigapose/fine_tuning/visualize_epnp_gigapose_comparison_extrinsics.py:85).
# It now:
# Centers the CAD by [6.89, 0.003, 498.77] mm.
# Draws corrected EPnP using the centered CAD.
# Draws aligned GigaPose using the same centered CAD.
# Does not apply the large empirical right transform to the mesh or EPnP pose.
# Uses the aligned pose when matching the yellow detection box.
# Keeps the right transform only as a diagnostic consistency check.
# Rerun only the visualizer; selection does not need rerunning:

ask the codex if this effects the use of extrinsics and if there is anything I need to do when wanting to use the gigapose adb epnp selected samples for a diffrent trainig






```





