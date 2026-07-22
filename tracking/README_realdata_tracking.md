<!--
/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-26-12-19-49/ Done
 /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4/ Done
 /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v0 Done
 /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v2 Done
  /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1 Done
   /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v2-v4 only front for now:
  -->

  Preditions are: 


```bash
# 20260505v1 

dataset: gigaPose_datasets/datasets/real_20260505v1_front_gsam_v4

front: /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/front/EPnPv2_gt_mesh_z_hybrid_labels

pred: gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260505v1_front_gsam_v4-test_large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv


gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt
python -m tracking.rgb_self_recovery.run \
  --predictions gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260505v1_front_gsam_v4-test_large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_front_gsam_v4 \
  --split test \
  --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
  --association-config tracking/configs/improved.json \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_real_20260505v1_front_gsam_v4 \
  --device cuda \
  --top-k-gigapose 5 \
  --beam-size 4 \
  --max-candidates 48 \
  --refinement-iterations 2 \
  --global-interval 5 \
  --broad-recovery-confidence 0.55 \
  --normal-confidence 0.65 \
  --lost-confidence 0.25 \
  --save-overlays \
  --overlay-every 10 \
  --overwrite


gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_gsam_v4/tracked_predictions.csv

python -m tracking.select_real_label_candidates \
  --tracked-predictions gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_gsam_v4/tracked_predictions.csv \
  --allowed-tracking-modes normal \
    --min-tracking-confidence 0.65 \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_front_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/front/EPnPv2_gt_mesh_z_hybrid_labels \
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
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_gsam_v4/label_candidates_for_optimization




python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_gsam_v4/label_candidates_for_optimization/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_front_gsam_v4 \
  --split test \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_gsam_v4/label_candidates_for_optimization/visual_overlays \
  --max-images 100 \
  --sort-by translation_error \
  --draw-mask-bbox \
  --bbox-match-mode nearest_projected_center \
  --frame-transform-side right

```

```bash 
# 20260505v1 
dataset: gigaPose_datasets/datasets/real_20260505v1_rear_gsam_v4

rear: /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/rear/EPnPv2_gt_mesh_z_hybrid_labels

pred: gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260505v1_rear_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260505v1_rear_gsam_v4-test_large_real_20260505v1_rear_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv


  python -m tracking.rgb_self_recovery.run \
  --predictions gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260505v1_rear_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260505v1_rear_gsam_v4-test_large_real_20260505v1_rear_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_rear_gsam_v4 \
  --split test \
  --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
  --association-config tracking/configs/improved.json \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_real_20260505v1_rear_gsam_v4 \
  --device cuda \
  --top-k-gigapose 5 \
  --beam-size 4 \
  --max-candidates 48 \
  --refinement-iterations 2 \
  --global-interval 5 \
  --broad-recovery-confidence 0.55 \
  --normal-confidence 0.65 \
  --lost-confidence 0.25 \
  --save-overlays \
  --overlay-every 10 \
  --overwrite


gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_rear_gsam_v4/tracked_predictions.csv

python -m tracking.select_real_label_candidates \
  --tracked-predictions gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_rear_gsam_v4/tracked_predictions.csv \
  --allowed-tracking-modes normal \
    --min-tracking-confidence 0.65 \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_rear_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/rear/EPnPv2_gt_mesh_z_hybrid_labels \
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
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_rear_gsam_v4/label_candidates_for_optimization



python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_rear_gsam_v4/label_candidates_for_optimization/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_rear_gsam_v4 \
  --split test \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_rear_gsam_v4/label_candidates_for_optimization/visual_overlays \
  --max-images 100 \
  --sort-by translation_error \
  --draw-mask-bbox \
  --bbox-match-mode nearest_projected_center \
  --frame-transform-side right


```

```bash 
# 20260505v1 
dataset: gigaPose_datasets/datasets/real_20260505v1_stereo_left_gsam_v4

stereo_left: /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels

 pred:  gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260505v1_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260505v1_stereo_left_gsam_v4-test_large_real_20260505v1_stereo_left_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv


    python -m tracking.rgb_self_recovery.run \
  --predictions gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260505v1_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260505v1_stereo_left_gsam_v4-test_large_real_20260505v1_stereo_left_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_stereo_left_gsam_v4 \
  --split test \
  --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
  --association-config tracking/configs/improved.json \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_stereo_left_gsam_v4 \
  --device cuda \
  --top-k-gigapose 5 \
  --beam-size 4 \
  --max-candidates 48 \
  --refinement-iterations 2 \
  --global-interval 5 \
  --broad-recovery-confidence 0.55 \
  --normal-confidence 0.65 \
  --lost-confidence 0.25 \
  --save-overlays \
  --overlay-every 10 \
  --overwrite


gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_stereo_left_gsam_v4/tracked_predictions.csv

  python -m tracking.select_real_label_candidates \
  --tracked-predictions gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_stereo_left_gsam_v4/tracked_predictions.csv \
  --allowed-tracking-modes normal \
    --min-tracking-confidence 0.65 \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_stereo_left_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels \
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
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_stereo_left_gsam_v4/label_candidates_for_optimization









python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_stereo_left_gsam_v4/label_candidates_for_optimization/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_stereo_left_gsam_v4 \
  --split test \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_stereo_left_gsam_v4/label_candidates_for_optimization/visual_overlays \
  --max-images 100 \
  --sort-by translation_error \
  --draw-mask-bbox \
  --bbox-match-mode nearest_projected_center \
  --frame-transform-side right
```
-------------------------------------------------------------------
-------------------------------------------------------------------

```bash 
# 20260505v2
dataset: real_20260505v2_rear_gsam_v4
rear: /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v2/rear/EPnPv2_gt_mesh_z_hybrid_labels

 pred:  gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260505v2_rear_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260505v2_rear_gsam_v4-test_large_real_20260505v2_rear_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv


    python -m tracking.rgb_self_recovery.run \
  --predictions gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260505v2_rear_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260505v2_rear_gsam_v4-test_large_real_20260505v2_rear_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v2_rear_gsam_v4 \
  --split test \
  --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
  --association-config tracking/configs/improved.json \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v2_rear_gsam_v4 \
  --device cuda \
  --top-k-gigapose 5 \
  --beam-size 4 \
  --max-candidates 48 \
  --refinement-iterations 2 \
  --global-interval 5 \
  --broad-recovery-confidence 0.55 \
  --normal-confidence 0.65 \
  --lost-confidence 0.25 \
  --save-overlays \
  --overlay-every 10 \
  --overwrite


gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v2_rear_gsam_v4/tracked_predictions.csv

  python -m tracking.select_real_label_candidates \
  --tracked-predictions gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v2_rear_gsam_v4/tracked_predictions.csv \
  --allowed-tracking-modes normal \
    --min-tracking-confidence 0.65 \
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
  --epnp-translation-unit m \
  --min-score 0.05 \
  --max-translation-error-mm 3000 \
  --max-rotation-error-deg 30 \
        --max-roll-error-deg 5 \
    --max-pitch-error-deg 5 \
    --max-yaw-error-deg 15 \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v2_rear_gsam_v4/label_candidates_for_optimization


python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v2_rear_gsam_v4/label_candidates_for_optimization/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v2_rear_gsam_v4 \
  --split test \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v2_rear_gsam_v4/label_candidates_for_optimization/visual_overlays \
  --max-images 100 \
  --sort-by translation_error \
  --draw-mask-bbox \
  --bbox-match-mode nearest_projected_center \
  --frame-transform-side right

```

```bash 
# 20260518v0 
real_20260518v0_front_gsam_v4
front: /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v0/front/EPnPv2_gt_mesh_z_hybrid_labels 

pred: gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v0_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v0_front_gsam_v4-test_large_real_20260518v0_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv

    python -m tracking.rgb_self_recovery.run \
  --predictions gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v0_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v0_front_gsam_v4-test_large_real_20260518v0_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v0_front_gsam_v4 \
  --split test \
  --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
  --association-config tracking/configs/improved.json \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v0_front_gsam_v4 \
  --device cuda \
  --top-k-gigapose 5 \
  --beam-size 4 \
  --max-candidates 48 \
  --refinement-iterations 2 \
  --global-interval 5 \
  --broad-recovery-confidence 0.55 \
  --normal-confidence 0.65 \
  --lost-confidence 0.25 \
  --save-overlays \
  --overlay-every 10 \
  --overwrite

gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v0_front_gsam_v4/tracked_predictions.csv


  python -m tracking.select_real_label_candidates \
  --tracked-predictions gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v0_front_gsam_v4/tracked_predictions.csv \
  --allowed-tracking-modes normal \
    --min-tracking-confidence 0.65 \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v0_front_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v0/front/EPnPv2_gt_mesh_z_hybrid_labels  \
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
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v0_front_gsam_v4/label_candidates_for_optimization


python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v0_front_gsam_v4/label_candidates_for_optimization/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v0_front_gsam_v4 \
  --split test \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v0_front_gsam_v4/label_candidates_for_optimization/visual_overlays \
  --max-images 100 \
  --sort-by translation_error \
  --draw-mask-bbox \
  --bbox-match-mode nearest_projected_center \
  --frame-transform-side right


```

```bash 
# 20260518v0  
real_20260518v0_stereo_left_gsam_v4

stereo_left: /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v0/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels 

 pred:  gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v0_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v0_stereo_left_gsam_v4-test_large_real_20260518v0_stereo_left_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv

python -m tracking.rgb_self_recovery.run \
  --predictions gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v0_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v0_stereo_left_gsam_v4-test_large_real_20260518v0_stereo_left_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v0_stereo_left_gsam_v4 \
  --split test \
  --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
  --association-config tracking/configs/improved.json \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v0_stereo_left_gsam_v4 \
  --device cuda \
  --top-k-gigapose 5 \
  --beam-size 4 \
  --max-candidates 48 \
  --refinement-iterations 2 \
  --global-interval 5 \
  --broad-recovery-confidence 0.55 \
  --normal-confidence 0.65 \
  --lost-confidence 0.25 \
  --save-overlays \
  --overlay-every 10 \
  --overwrite

gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v0_stereo_left_gsam_v4

  python -m tracking.select_real_label_candidates \
  --tracked-predictions gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v0_stereo_left_gsam_v4/tracked_predictions.csv \
  --allowed-tracking-modes normal \
    --min-tracking-confidence 0.65 \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v0_stereo_left_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v0/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels \
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
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v0_stereo_left_gsam_v4/label_candidates_for_optimization


python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v0_stereo_left_gsam_v4/label_candidates_for_optimization/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v0_stereo_left_gsam_v4 \
  --split test \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v0_stereo_left_gsam_v4/label_candidates_for_optimization/visual_overlays \
  --max-images 100 \
  --sort-by translation_error \
  --draw-mask-bbox \
  --bbox-match-mode nearest_projected_center \
  --frame-transform-side right


```

----------
```bash 
# 20260518v1v4  
# front:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4/front/EPnPv2_gt_mesh_z_hybrid_labels

real_20260518v1v4_front_gsam_v4

 pred: gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v1v4_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v1v4_front_gsam_v4-test_large_real_20260518v1v4_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv



 python -m tracking.rgb_self_recovery.run \
  --predictions gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v1v4_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v1v4_front_gsam_v4-test_large_real_20260518v1v4_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_front_gsam_v4 \
  --split test \
  --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
  --association-config tracking/configs/improved.json \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_front_gsam_v4 \
  --device cuda \
  --top-k-gigapose 5 \
  --beam-size 4 \
  --max-candidates 48 \
  --refinement-iterations 2 \
  --global-interval 5 \
  --broad-recovery-confidence 0.55 \
  --normal-confidence 0.65 \
  --lost-confidence 0.25 \
  --save-overlays \
  --overlay-every 10 \
  --overwrite

  python -m tracking.select_real_label_candidates \
  --tracked-predictions gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_front_gsam_v4/tracked_predictions.csv \
  --allowed-tracking-modes normal \
    --min-tracking-confidence 0.65 \
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
  --epnp-translation-unit m \
  --min-score 0.05 \
  --max-translation-error-mm 3000 \
  --max-rotation-error-deg 30 \
        --max-roll-error-deg 5 \
    --max-pitch-error-deg 5 \
    --max-yaw-error-deg 15 \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_front_gsam_v4/label_candidates_for_optimization


python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_front_gsam_v4/label_candidates_for_optimization/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_front_gsam_v4 \
  --split test \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_front_gsam_v4/label_candidates_for_optimization/visual_overlays \
  --max-images 100 \
  --sort-by translation_error \
  --draw-mask-bbox \
  --bbox-match-mode nearest_projected_center \
  --frame-transform-side right



```

```bash 

# rear:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4/rear/EPnPv2_gt_mesh_z_hybrid_labels

real_20260518v1v4_rear_gsam_v4
 pred:  gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v1v4_rear_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v1v4_rear_gsam_v4-test_large_real_20260518v1v4_rear_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv


 python -m tracking.rgb_self_recovery.run \
  --predictions gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v1v4_rear_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v1v4_rear_gsam_v4-test_large_real_20260518v1v4_rear_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_rear_gsam_v4 \
  --split test \
  --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
  --association-config tracking/configs/improved.json \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_rear_gsam_v4 \
  --device cuda \
  --top-k-gigapose 5 \
  --beam-size 4 \
  --max-candidates 48 \
  --refinement-iterations 2 \
  --global-interval 5 \
  --broad-recovery-confidence 0.55 \
  --normal-confidence 0.65 \
  --lost-confidence 0.25 \
  --save-overlays \
  --overlay-every 10 \
  --overwrite


    python -m tracking.select_real_label_candidates \
  --tracked-predictions gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_rear_gsam_v4/tracked_predictions.csv \
  --allowed-tracking-modes normal \
    --min-tracking-confidence 0.65 \
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
  --epnp-translation-unit m \
  --min-score 0.05 \
  --max-translation-error-mm 3000 \
  --max-rotation-error-deg 30 \
        --max-roll-error-deg 5 \
    --max-pitch-error-deg 5 \
    --max-yaw-error-deg 15 \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_rear_gsam_v4/label_candidates_for_optimization


python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_rear_gsam_v4/label_candidates_for_optimization/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_rear_gsam_v4 \
  --split test \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_rear_gsam_v4/label_candidates_for_optimization/visual_overlays \
  --max-images 100 \
  --sort-by translation_error \
  --draw-mask-bbox \
  --bbox-match-mode nearest_projected_center \
  --frame-transform-side right
```

```bash 
stereo_left: /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels
real_20260518v1v4_stereo_left_gsam_v4
 pred:  gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v1v4_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v1v4_stereo_left_gsam_v4-test_large_real_20260518v1v4_stereo_left_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv



 python -m tracking.rgb_self_recovery.run \
  --predictions gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v1v4_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v1v4_stereo_left_gsam_v4-test_large_real_20260518v1v4_stereo_left_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_stereo_left_gsam_v4 \
  --split test \
  --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
  --association-config tracking/configs/improved.json \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_stereo_left_gsam_v4 \
  --device cuda \
  --top-k-gigapose 5 \
  --beam-size 4 \
  --max-candidates 48 \
  --refinement-iterations 2 \
  --global-interval 5 \
  --broad-recovery-confidence 0.55 \
  --normal-confidence 0.65 \
  --lost-confidence 0.25 \
  --save-overlays \
  --overlay-every 10 \
  --overwrite

    python -m tracking.select_real_label_candidates \
  --tracked-predictions gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_stereo_left_gsam_v4/tracked_predictions.csv \
  --allowed-tracking-modes normal \
    --min-tracking-confidence 0.65 \
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
  --epnp-translation-unit m \
  --min-score 0.05 \
  --max-translation-error-mm 3000 \
  --max-rotation-error-deg 30 \
        --max-roll-error-deg 5 \
    --max-pitch-error-deg 5 \
    --max-yaw-error-deg 15 \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_stereo_left_gsam_v4/label_candidates_for_optimization


python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_stereo_left_gsam_v4/label_candidates_for_optimization/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_stereo_left_gsam_v4 \
  --split test \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_stereo_left_gsam_v4/label_candidates_for_optimization/visual_overlays \
  --max-images 100 \
  --sort-by translation_error \
  --draw-mask-bbox \
  --bbox-match-mode nearest_projected_center \
  --frame-transform-side right


```

```bash 
# 20260518v2v4 
real_20260518v2v4_front_gsam_v4
front: /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v2-v4/front/EPnPv2_gt_mesh_z_hybrid_labels

 pred:  gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v2v4_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v2v4_front_gsam_v4-test_large_real_20260518v2v4_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv


 CUDA_VISIBLE_DEVICES=1 python -m tracking.rgb_self_recovery.run \
  --predictions gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v2v4_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v2v4_front_gsam_v4-test_large_real_20260518v2v4_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v2v4_front_gsam_v4 \
  --split test \
  --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
  --association-config tracking/configs/improved.json \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v2v4_front_gsam_v4 \
  --device cuda \
  --top-k-gigapose 5 \
  --beam-size 4 \
  --max-candidates 48 \
  --refinement-iterations 2 \
  --global-interval 5 \
  --broad-recovery-confidence 0.55 \
  --normal-confidence 0.65 \
  --lost-confidence 0.25 \
  --save-overlays \
  --overlay-every 10 \
  --overwrite

    in progress


python -m tracking.select_real_label_candidates \
  --tracked-predictions gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v2v4_front_gsam_v4/tracked_predictions.csv \
  --allowed-tracking-modes normal \
    --min-tracking-confidence 0.65 \
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
  --epnp-translation-unit m \
  --min-score 0.05 \
  --max-translation-error-mm 3000 \
  --max-rotation-error-deg 30 \
        --max-roll-error-deg 5 \
    --max-pitch-error-deg 5 \
    --max-yaw-error-deg 15 \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v2v4_front_gsam_v4/label_candidates_for_optimization


python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v2v4_front_gsam_v4/label_candidates_for_optimization/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v2v4_front_gsam_v4 \
  --split test \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v2v4_front_gsam_v4/label_candidates_for_optimization/visual_overlays \
  --max-images 100 \
  --sort-by translation_error \
  --draw-mask-bbox \
  --bbox-match-mode nearest_projected_center \
  --frame-transform-side right
```


```bash 
stereo_left:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v2-v4/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels

real_20260518v2v4_stereo_left_gsam_v4

 pred:  gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v2v4_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v2v4_stereo_left_gsam_v4-test_large_real_20260518v2v4_stereo_left_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv


 python -m tracking.rgb_self_recovery.run \
  --predictions gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v2v4_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v2v4_stereo_left_gsam_v4-test_large_real_20260518v2v4_stereo_left_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v2v4_stereo_left_gsam_v4 \
  --split test \
  --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
  --association-config tracking/configs/improved.json \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v2v4_stereo_left_gsam_v4 \
  --device cuda \
  --top-k-gigapose 5 \
  --beam-size 4 \
  --max-candidates 48 \
  --refinement-iterations 2 \
  --global-interval 5 \
  --broad-recovery-confidence 0.55 \
  --normal-confidence 0.65 \
  --lost-confidence 0.25 \
  --save-overlays \
  --overlay-every 10 \
  --overwrite


python -m tracking.select_real_label_candidates \
  --tracked-predictions gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v2v4_stereo_left_gsam_v4/tracked_predictions.csv \
  --allowed-tracking-modes normal \
    --min-tracking-confidence 0.65 \
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
  --epnp-translation-unit m \
  --min-score 0.05 \
  --max-translation-error-mm 3000 \
  --max-rotation-error-deg 30 \
        --max-roll-error-deg 5 \
    --max-pitch-error-deg 5 \
    --max-yaw-error-deg 15 \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v2v4_stereo_left_gsam_v4/label_candidates_for_optimization


python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v2v4_stereo_left_gsam_v4/label_candidates_for_optimization/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v2v4_stereo_left_gsam_v4 \
  --split test \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v2v4_stereo_left_gsam_v4/label_candidates_for_optimization/visual_overlays \
  --max-images 100 \
  --sort-by translation_error \
  --draw-mask-bbox \
  --bbox-match-mode nearest_projected_center \
  --frame-transform-side right



 ```



 If you have multiple selected files, pass multiple --input arguments:
Important: only combine files for the same camera.

```bash
For only front: 
gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v2v4_front_gsam_v4


python -m fine_tuning.combine_selected_samples \
  --input  gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_gsam_v4/label_candidates_for_optimization/selected_samples.csv \
  --input gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v0_front_gsam_v4/label_candidates_for_optimization/selected_samples.csv \
  --input  gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_front_gsam_v4/label_candidates_for_optimization/selected_samples.csv \
  --input gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v2v4_front_gsam_v4/label_candidates_for_optimization/selected_samples.csv \
  --output gigaPose_datasets/results/rgb_self_recovery_realData_dataset/combined_front_selected_samples_for_optimization.csv \
  --dedupe-by match_key_epnp \
  --keep lowest-error

e_datasets/results/real_world_ot2_IST_tran/large_real_20260505v1_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/selected_samples.csv
For only rear: 
gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_rear_gsam_v4




python -m fine_tuning.combine_selected_samples \
  --input  gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_rear_gsam_v4/label_candidates_for_optimization/selected_samples.csv \
  --input gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v2_rear_gsam_v4/label_candidates_for_optimization/selected_samples.csv \
  --input gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_rear_gsam_v4/label_candidates_for_optimization/selected_samples.csv \
  --output gigaPose_datasets/results/rgb_self_recovery_realData_dataset/combined_rear_selected_samples_for_optimization.csv \
  --dedupe-by match_key_epnp \
  --keep lowest-error


For only stereo_left: 
gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_stereo_left_gsam_v4/label_candidates_for_optimization

python -m fine_tuning.combine_selected_samples \
  --input  gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_stereo_left_gsam_v4/label_candidates_for_optimization/selected_samples.csv \
  --input gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v0_stereo_left_gsam_v4/label_candidates_for_optimization/selected_samples.csv \
  --input gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_stereo_left_gsam_v4/label_candidates_for_optimization/selected_samples.csv \
  --input gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v2v4_stereo_left_gsam_v4/label_candidates_for_optimization/selected_samples.csv \
  --output gigaPose_datasets/results/rgb_self_recovery_realData_dataset/combined_stereo_left_selected_samples_for_optimization_new.csv \
  --dedupe-by match_key_epnp \
  --keep lowest-error
```



