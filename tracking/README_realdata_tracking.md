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





python -m tracking.rgb_self_recovery.run \
  --predictions gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260505v1_front_gsam_v4-test_large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_front_gsam_v4 \
  --split test \
  --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
  --association-config tracking/configs/improved.json \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_rotation_gated \
  --device cuda \
  --orientation-gates \
  --no-allow-flip-hypotheses \
  --normal-max-rotation-step-deg 30 \
  --uncertain-max-rotation-step-deg 60 \
  --max-rank0-rotation-disagreement-deg 90 \
  --save-overlays \
  --overlay-every 10 \
  --overwrite

gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_gsam_v4/tracked_predictions.csv

python -m tracking.select_real_label_candidates \
  --tracked-predictions gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_rotation_gated/tracked_predictions.csv \
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
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_rotation_gated/label_candidates_for_optimization

selection: 135 out of 360


python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_rotation_gated/label_candidates_for_optimization/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_front_gsam_v4 \
  --split test \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_rotation_gated/label_candidates_for_optimization/visual_overlays \
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



python -m tracking.rgb_self_recovery.run \
  --predictions gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260505v1_rear_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260505v1_rear_gsam_v4-test_large_real_20260505v1_rear_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_rear_gsam_v4 \
  --split test \
  --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
  --association-config tracking/configs/improved.json \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_rear_rotation_gated \
  --device cuda \
  --top-k-gigapose 5 \
  --beam-size 4 \
  --max-candidates 48 \
  --refinement-iterations 2 \
  --global-interval 5 \
  --broad-recovery-confidence 0.55 \
  --normal-confidence 0.65 \
  --lost-confidence 0.25 \
  --orientation-gates \
  --no-allow-flip-hypotheses \
  --normal-max-rotation-step-deg 30 \
  --uncertain-max-rotation-step-deg 60 \
  --max-rank0-rotation-disagreement-deg 90 \
  --save-overlays \
  --overlay-every 10 \
  --overlay-axis-length-m 1.0 \
  --overwrite

 


gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_rear_gsam_v4/tracked_predictions.csv
gigapose/gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_rear_rotation_gated
python -m tracking.select_real_label_candidates \
  --tracked-predictions gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_rear_rotation_gated/tracked_predictions.csv \
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
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_rear_rotation_gated/label_candidates_for_optimization

selection: 79 out of 522

python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_rear_rotation_gated/label_candidates_for_optimization/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_rear_gsam_v4 \
  --split test \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_rear_rotation_gated/label_candidates_for_optimization/visual_overlays \
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



python -m tracking.rgb_self_recovery.run \
  --predictions  gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260505v1_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260505v1_stereo_left_gsam_v4-test_large_real_20260505v1_stereo_left_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_stereo_left_gsam_v4 \
  --split test \
  --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
  --association-config tracking/configs/improved.json \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_stereo_left_rotation_gated \
  --device cuda \
  --top-k-gigapose 5 \
  --beam-size 4 \
  --max-candidates 48 \
  --refinement-iterations 2 \
  --global-interval 5 \
  --broad-recovery-confidence 0.55 \
  --normal-confidence 0.65 \
  --lost-confidence 0.25 \
  --orientation-gates \
  --no-allow-flip-hypotheses \
  --normal-max-rotation-step-deg 30 \
  --uncertain-max-rotation-step-deg 60 \
  --max-rank0-rotation-disagreement-deg 90 \
  --save-overlays \
  --overlay-every 10 \
  --overlay-axis-length-m 1.0 \
  --overwrite




gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_stereo_left_gsam_v4/tracked_predictions.csv

  python -m tracking.select_real_label_candidates \
  --tracked-predictions gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_stereo_left_rotation_gated/tracked_predictions.csv \
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
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_stereo_left_rotation_gated/label_candidates_for_optimization

selection: 56 out of 113

python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_stereo_left_rotation_gated/label_candidates_for_optimization/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_stereo_left_gsam_v4 \
  --split test \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_stereo_left_rotation_gated/label_candidates_for_optimization/visual_overlays \
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



python -m tracking.rgb_self_recovery.run \
  --predictions  gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260505v2_rear_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260505v2_rear_gsam_v4-test_large_real_20260505v2_rear_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv  \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v2_rear_gsam_v4 \
  --split test \
  --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
  --association-config tracking/configs/improved.json \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v2_rear_rotation_gated \
  --device cuda \
  --top-k-gigapose 5 \
  --beam-size 4 \
  --max-candidates 48 \
  --refinement-iterations 2 \
  --global-interval 5 \
  --broad-recovery-confidence 0.55 \
  --normal-confidence 0.65 \
  --lost-confidence 0.25 \
  --orientation-gates \
  --no-allow-flip-hypotheses \
  --normal-max-rotation-step-deg 30 \
  --uncertain-max-rotation-step-deg 60 \
  --max-rank0-rotation-disagreement-deg 90 \
  --save-overlays \
  --overlay-every 10 \
  --overlay-axis-length-m 1.0 \
  --overwrite





gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v2_rear_gsam_v4/tracked_predictions.csv

  python -m tracking.select_real_label_candidates \
  --tracked-predictions gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v2_rear_rotation_gated/tracked_predictions.csv \
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
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v2_rear_rotation_gated/label_candidates_for_optimization

selection: 38 out of 206 

python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v2_rear_rotation_gated/label_candidates_for_optimization/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v2_rear_gsam_v4 \
  --split test \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v2_rear_rotation_gated/label_candidates_for_optimization/visual_overlays \
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



python -m tracking.rgb_self_recovery.run \
  --predictions  gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v0_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v0_front_gsam_v4-test_large_real_20260518v0_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv  \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v0_front_gsam_v4 \
  --split test \
  --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
  --association-config tracking/configs/improved.json \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v0_front_rotation_gated \
  --device cuda \
  --top-k-gigapose 5 \
  --beam-size 4 \
  --max-candidates 48 \
  --refinement-iterations 2 \
  --global-interval 5 \
  --broad-recovery-confidence 0.55 \
  --normal-confidence 0.65 \
  --lost-confidence 0.25 \
  --orientation-gates \
  --no-allow-flip-hypotheses \
  --normal-max-rotation-step-deg 30 \
  --uncertain-max-rotation-step-deg 60 \
  --max-rank0-rotation-disagreement-deg 90 \
  --save-overlays \
  --overlay-every 10 \
  --overlay-axis-length-m 1.0 \
  --overwrite


gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v0_front_gsam_v4/tracked_predictions.csv


  python -m tracking.select_real_label_candidates \
  --tracked-predictions gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v0_front_rotation_gated/tracked_predictions.csv \
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
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v0_front_rotation_gated/label_candidates_for_optimization

selection: 233 out of 382 


python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v0_front_rotation_gated/label_candidates_for_optimization/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v0_front_gsam_v4 \
  --split test \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v0_front_rotation_gated/label_candidates_for_optimization/visual_overlays \
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


CUDA_VISIBLE_DEVICES=0 python -m tracking.rgb_self_recovery.run \
  --predictions  gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v0_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v0_stereo_left_gsam_v4-test_large_real_20260518v0_stereo_left_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv  \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v0_stereo_left_gsam_v4 \
  --split test \
  --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
  --association-config tracking/configs/improved.json \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v0_stereo_left_rotation_gated \
  --device cuda \
  --top-k-gigapose 5 \
  --beam-size 4 \
  --max-candidates 48 \
  --refinement-iterations 2 \
  --global-interval 5 \
  --broad-recovery-confidence 0.55 \
  --normal-confidence 0.65 \
  --lost-confidence 0.25 \
  --orientation-gates \
  --no-allow-flip-hypotheses \
  --normal-max-rotation-step-deg 30 \
  --uncertain-max-rotation-step-deg 60 \
  --max-rank0-rotation-disagreement-deg 90 \
  --save-overlays \
  --overlay-every 10 \
  --overlay-axis-length-m 1.0 \
  --overwrite

in progress ..  
(fisnish runig gigapose on the new dataset, just need to run the select labels on it. 
and finish this tracking prep. need to run the visualize and select lables for this run)

gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v0_stereo_left_gsam_v4

  python -m tracking.select_real_label_candidates \
  --tracked-predictions gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v0_stereo_left_rotation_gated/tracked_predictions.csv \
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
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v0_stereo_left_rotation_gated/label_candidates_for_optimization


selection: 334 out of 390 


python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v0_stereo_left_rotation_gated/label_candidates_for_optimization/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v0_stereo_left_gsam_v4 \
  --split test \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v0_stereo_left_rotation_gated/label_candidates_for_optimization/visual_overlays \
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



#  python -m tracking.rgb_self_recovery.run \
#   --predictions gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v1v4_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v1v4_front_gsam_v4-test_large_real_20260518v1v4_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
#   --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_front_gsam_v4 \
#   --split test \
#   --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
#   --association-config tracking/configs/improved.json \
#   --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_front_gsam_v4 \
#   --device cuda \
#   --top-k-gigapose 5 \
#   --beam-size 4 \
#   --max-candidates 48 \
#   --refinement-iterations 2 \
#   --global-interval 5 \
#   --broad-recovery-confidence 0.55 \
#   --normal-confidence 0.65 \
#   --lost-confidence 0.25 \
#   --save-overlays \
#   --overlay-every 10 \
#   --overwrite

CUDA_VISIBLE_DEVICES=0 python -m tracking.rgb_self_recovery.run \
  --predictions  gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v1v4_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v1v4_front_gsam_v4-test_large_real_20260518v1v4_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv  \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_front_gsam_v4 \
  --split test \
  --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
  --association-config tracking/configs/improved.json \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_front_rotation_gated \
  --device cuda \
  --top-k-gigapose 5 \
  --beam-size 4 \
  --max-candidates 48 \
  --refinement-iterations 2 \
  --global-interval 5 \
  --broad-recovery-confidence 0.55 \
  --normal-confidence 0.65 \
  --lost-confidence 0.25 \
  --orientation-gates \
  --no-allow-flip-hypotheses \
  --normal-max-rotation-step-deg 30 \
  --uncertain-max-rotation-step-deg 60 \
  --max-rank0-rotation-disagreement-deg 90 \
  --save-overlays \
  --overlay-every 10 \
  --overlay-axis-length-m 1.0 \
  --overwrite


# still need to run the selection and visualizer after

  python -m tracking.select_real_label_candidates \
  --tracked-predictions gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_front_rotation_gated/tracked_predictions.csv \
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
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_front_rotation_gated/label_candidates_for_optimization



selection: 70 out of 111 

python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_front_rotation_gated/label_candidates_for_optimization/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_front_gsam_v4 \
  --split test \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_front_rotation_gated/label_candidates_for_optimization/visual_overlays \
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


#  python -m tracking.rgb_self_recovery.run \
#   --predictions gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v1v4_rear_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v1v4_rear_gsam_v4-test_large_real_20260518v1v4_rear_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
#   --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_rear_gsam_v4 \
#   --split test \
#   --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
#   --association-config tracking/configs/improved.json \
#   --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_rear_gsam_v4 \
#   --device cuda \
#   --top-k-gigapose 5 \
#   --beam-size 4 \
#   --max-candidates 48 \
#   --refinement-iterations 2 \
#   --global-interval 5 \
#   --broad-recovery-confidence 0.55 \
#   --normal-confidence 0.65 \
#   --lost-confidence 0.25 \
#   --save-overlays \
#   --overlay-every 10 \
#   --overwrite



CUDA_VISIBLE_DEVICES=1 python -m tracking.rgb_self_recovery.run \
  --predictions  gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v1v4_rear_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v1v4_rear_gsam_v4-test_large_real_20260518v1v4_rear_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv  \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_rear_gsam_v4 \
  --split test \
  --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
  --association-config tracking/configs/improved.json \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_rear_rotation_gated \
  --device cuda \
  --top-k-gigapose 5 \
  --beam-size 4 \
  --max-candidates 48 \
  --refinement-iterations 2 \
  --global-interval 5 \
  --broad-recovery-confidence 0.55 \
  --normal-confidence 0.65 \
  --lost-confidence 0.25 \
  --orientation-gates \
  --no-allow-flip-hypotheses \
  --normal-max-rotation-step-deg 30 \
  --uncertain-max-rotation-step-deg 60 \
  --max-rank0-rotation-disagreement-deg 90 \
  --save-overlays \
  --overlay-every 10 \
  --overlay-axis-length-m 1.0 \
  --overwrite



    python -m tracking.select_real_label_candidates \
  --tracked-predictions gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_rear_rotation_gated/tracked_predictions.csv \
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
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_rear_rotation_gated/label_candidates_for_optimization

still in progress

candidates: 442   out of 3348


run a diffrent one as well: 

python -m label_selection.select_absolute \
  --gigapose-predictions gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_rear_rotation_gated/tracked_predictions.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_rear_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4/rear/EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-camera-pose-key T_camera_object_centered \
  --epnp-map-pose-unit m \
  --epnp-camera-pose-unit m \
  --raw-object-center-m -0.2411941141 0.0009010172 0.3329219520 \
  --prediction-translation-unit mm \
  --min-score 0.05 \
  --max-translation-error-mm 4000 \
  --max-rotation-error-deg 30 \
  --max-roll-error-deg 10 \
  --max-pitch-error-deg 10 \
  --max-yaw-error-deg 15 \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_rear_rotation_gated/absolute_labels \
  --overwrite

candidates: 260  out of 2857



python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_rear_rotation_gated/label_candidates_for_optimization/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_rear_gsam_v4 \
  --split test \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_rear_rotation_gated/label_candidates_for_optimization/visual_overlays \
  --max-images 100 \
  --sort-by translation_error \
  --draw-mask-bbox \
  --bbox-match-mode nearest_projected_center \
  --frame-transform-side right



  another vis:
  python -m label_selection.visualize \
  --candidate-csv gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_rear_rotation_gated/absolute_labels/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_rear_gsam_v4 \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_rear_rotation_gated/absolute_labels/visual_overlays \
  --projection-model metadata \
  --draw-mask-bbox \
  --max-images 100 \
  --overwrite
```

 











gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_rotation_gated/tracked_predictions.csv

python3 -m teacher_pipeline.build_observations   --predictions gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260505v1_front_gsam_v4-test_large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv   --selected-samples gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tran/predictions/label_candidates_for_optimization_new/selected_samples.csv   --tracks gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_rotation_gated/tracked_predictions.csv  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_front_gsam_v4   --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/front/EPnPv2_gt_mesh_z_hybrid_labels   --metadata-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/front/metadata   --prediction-translation-unit mm   --epnp-map-pose-unit m   --epnp-camera-pose-unit m   --output gigaPose_datasets/results/teacher_v1_20260505_front/observations.jsonl   --strict



RUN=gigaPose_datasets/results/teacher_v1_extended_20260505_front

TRACKED=gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_rotation_gated/tracked_predictions.csv 

DATASET=gigaPose_datasets/datasets/real_20260505v1_front_gsam_v4

EPNP=/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/front/EPnPv2_gt_mesh_z_hybrid_labels

META=/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/front/metadata

TRACK_SURFACE=/media/hdd2/ARCL_multicar_bags/camera_dataset/Track_info/sim_track_info/putnam_park-no_chicanes_track_info/track_scene.ply

CAR_MESH="$DATASET/models/obj_000001.ply"

mkdir -p "$RUN"

```bash 
stereo_left: /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v1-v4/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels
real_20260518v1v4_stereo_left_gsam_v4
 pred:  gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v1v4_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v1v4_stereo_left_gsam_v4-test_large_real_20260518v1v4_stereo_left_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv

#  python -m tracking.rgb_self_recovery.run \
#   --predictions gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v1v4_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v1v4_stereo_left_gsam_v4-test_large_real_20260518v1v4_stereo_left_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
#   --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_stereo_left_gsam_v4 \
#   --split test \
#   --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
#   --association-config tracking/configs/improved.json \
#   --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_stereo_left_gsam_v4 \
#   --device cuda \
#   --top-k-gigapose 5 \
#   --beam-size 4 \
#   --max-candidates 48 \
#   --refinement-iterations 2 \
#   --global-interval 5 \
#   --broad-recovery-confidence 0.55 \
#   --normal-confidence 0.65 \
#   --lost-confidence 0.25 \
#   --save-overlays \
#   --overlay-every 10 \
#   --overwrite

CUDA_VISIBLE_DEVICES=0 python -m tracking.rgb_self_recovery.run \
  --predictions  gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v1v4_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v1v4_stereo_left_gsam_v4-test_large_real_20260518v1v4_stereo_left_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv  \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_stereo_left_gsam_v4 \
  --split test \
  --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
  --association-config tracking/configs/improved.json \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_stereo_left_rotation_gated \
  --device cuda \
  --top-k-gigapose 5 \
  --beam-size 4 \
  --max-candidates 48 \
  --refinement-iterations 2 \
  --global-interval 5 \
  --broad-recovery-confidence 0.55 \
  --normal-confidence 0.65 \
  --lost-confidence 0.25 \
  --orientation-gates \
  --no-allow-flip-hypotheses \
  --normal-max-rotation-step-deg 30 \
  --uncertain-max-rotation-step-deg 60 \
  --max-rank0-rotation-disagreement-deg 90 \
  --save-overlays \
  --overlay-every 10 \
  --overlay-axis-length-m 1.0 \
  --overwrite



    python -m tracking.select_real_label_candidates \
  --tracked-predictions gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_stereo_left_rotation_gated/tracked_predictions.csv \
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
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_stereo_left_rotation_gated/label_candidates_for_optimization


candidates:  46  out of 170

python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_stereo_left_rotation_gated/label_candidates_for_optimization/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v1v4_stereo_left_gsam_v4 \
  --split test \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v1v4_stereo_left_rotation_gated/label_candidates_for_optimization/visual_overlays \
  --max-images 100 \
  --sort-by translation_error \
  --draw-mask-bbox \
  --bbox-match-mode nearest_projected_center \
  --frame-transform-side right


```
-----------------------------------------------------------
----------------------------------------------------------------

```bash 
# 20260518v2v4 
real_20260518v2v4_front_gsam_v4
front: /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v2-v4/front/EPnPv2_gt_mesh_z_hybrid_labels

 pred:  gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v2v4_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v2v4_front_gsam_v4-test_large_real_20260518v2v4_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv

#  CUDA_VISIBLE_DEVICES=1 python -m tracking.rgb_self_recovery.run \
#   --predictions gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v2v4_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v2v4_front_gsam_v4-test_large_real_20260518v2v4_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
#   --dataset-dir gigaPose_datasets/datasets/real_20260518v2v4_front_gsam_v4 \
#   --split test \
#   --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
#   --association-config tracking/configs/improved.json \
#   --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v2v4_front_gsam_v4 \
#   --device cuda \
#   --top-k-gigapose 5 \
#   --beam-size 4 \
#   --max-candidates 48 \
#   --refinement-iterations 2 \
#   --global-interval 5 \
#   --broad-recovery-confidence 0.55 \
#   --normal-confidence 0.65 \
#   --lost-confidence 0.25 \
#   --save-overlays \
#   --overlay-every 10 \
#   --overwrite

    

  CUDA_VISIBLE_DEVICES=0 python -m tracking.rgb_self_recovery.run \
  --predictions  gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v2v4_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v2v4_front_gsam_v4-test_large_real_20260518v2v4_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv  \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v2v4_front_gsam_v4 \
  --split test \
  --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
  --association-config tracking/configs/improved.json \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v2v4_front_rotation_gated \
  --device cuda \
  --top-k-gigapose 5 \
  --beam-size 4 \
  --max-candidates 48 \
  --refinement-iterations 2 \
  --global-interval 5 \
  --broad-recovery-confidence 0.55 \
  --normal-confidence 0.65 \
  --lost-confidence 0.25 \
  --orientation-gates \
  --no-allow-flip-hypotheses \
  --normal-max-rotation-step-deg 30 \
  --uncertain-max-rotation-step-deg 60 \
  --max-rank0-rotation-disagreement-deg 90 \
  --save-overlays \
  --overlay-every 10 \
  --overlay-axis-length-m 1.0 \
  --overwrite


python -m tracking.select_real_label_candidates \
  --tracked-predictions gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v2v4_front_rotation_gated/tracked_predictions.csv \
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
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v2v4_front_rotation_gated/label_candidates_for_optimization



candidates: out of

python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v2v4_front_rotation_gated/label_candidates_for_optimization/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v2v4_front_gsam_v4 \
  --split test \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v2v4_front_rotation_gated/label_candidates_for_optimization/visual_overlays \
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


#  python -m tracking.rgb_self_recovery.run \
#   --predictions gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v2v4_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v2v4_stereo_left_gsam_v4-test_large_real_20260518v2v4_stereo_left_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
#   --dataset-dir gigaPose_datasets/datasets/real_20260518v2v4_stereo_left_gsam_v4 \
#   --split test \
#   --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
#   --association-config tracking/configs/improved.json \
#   --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v2v4_stereo_left_gsam_v4 \
#   --device cuda \
#   --top-k-gigapose 5 \
#   --beam-size 4 \
#   --max-candidates 48 \
#   --refinement-iterations 2 \
#   --global-interval 5 \
#   --broad-recovery-confidence 0.55 \
#   --normal-confidence 0.65 \
#   --lost-confidence 0.25 \
#   --save-overlays \
#   --overlay-every 10 \
#   --overwrite


  CUDA_VISIBLE_DEVICES=1 python -m tracking.rgb_self_recovery.run \
  --predictions  gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v2v4_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v2v4_stereo_left_gsam_v4-test_large_real_20260518v2v4_stereo_left_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv  \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v2v4_stereo_left_gsam_v4 \
  --split test \
  --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
  --association-config tracking/configs/improved.json \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v2v4_stereo_left_rotation_gated \
  --device cuda \
  --top-k-gigapose 5 \
  --beam-size 4 \
  --max-candidates 48 \
  --refinement-iterations 2 \
  --global-interval 5 \
  --broad-recovery-confidence 0.55 \
  --normal-confidence 0.65 \
  --lost-confidence 0.25 \
  --orientation-gates \
  --no-allow-flip-hypotheses \
  --normal-max-rotation-step-deg 30 \
  --uncertain-max-rotation-step-deg 60 \
  --max-rank0-rotation-disagreement-deg 90 \
  --save-overlays \
  --overlay-every 10 \
  --overlay-axis-length-m 1.0 \
  --overwrite



python -m tracking.select_real_label_candidates \
  --tracked-predictions gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v2v4_stereo_left_rotation_gated/tracked_predictions.csv \
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
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v2v4_stereo_left_rotation_gated/label_candidates_for_optimization


candidates:   out of 

python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v2v4_stereo_left_rotation_gated/label_candidates_for_optimization/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v2v4_stereo_left_gsam_v4 \
  --split test \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260518v2v4_stereo_left_rotation_gated/label_candidates_for_optimization/visual_overlays \
  --max-images 100 \
  --sort-by translation_error \
  --draw-mask-bbox \
  --bbox-match-mode nearest_projected_center \
  --frame-transform-side right

 ```


lets run the same thing for the new dataset, but won't include it in the optimization

```bash 
# 20260718 
# front: /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-07-18/front/EPnPv2_gt_mesh_z_hybrid_labels
dataset: real_20260718_front_gsam_v4
preds: gigaPose_datasets/results/large_real_20260718_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260718_front_gsam_v4-test_large_real_20260718_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv


  CUDA_VISIBLE_DEVICES=0 python -m tracking.rgb_self_recovery.run \
  --predictions  gigaPose_datasets/results/large_real_20260718_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260718_front_gsam_v4-test_large_real_20260718_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv  \
  --dataset-dir gigaPose_datasets/datasets/real_20260718_front_gsam_v4 \
  --split test \
  --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
  --association-config tracking/configs/improved.json \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260718_front_rotation_gated \
  --device cuda \
  --top-k-gigapose 5 \
  --beam-size 4 \
  --max-candidates 48 \
  --refinement-iterations 2 \
  --global-interval 5 \
  --broad-recovery-confidence 0.55 \
  --normal-confidence 0.65 \
  --lost-confidence 0.25 \
  --orientation-gates \
  --no-allow-flip-hypotheses \
  --normal-max-rotation-step-deg 30 \
  --uncertain-max-rotation-step-deg 60 \
  --max-rank0-rotation-disagreement-deg 90 \
  --save-overlays \
  --overlay-every 10 \
  --overlay-axis-length-m 1.0 \
  --overwrite


python -m tracking.select_real_label_candidates \
  --tracked-predictions gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260718_front_rotation_gated/tracked_predictions.csv \
  --allowed-tracking-modes normal \
    --min-tracking-confidence 0.65 \
  --dataset-dir gigaPose_datasets/datasets/real_20260718_front_gsam_v4 \
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
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260718_front_rotation_gated/label_candidates_for_optimization


candidates: out of

python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260718_front_rotation_gated/label_candidates_for_optimization/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v2v4_front_gsam_v4 \
  --split test \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260718_front_rotation_gated/label_candidates_for_optimization/visual_overlays \
  --max-images 100 \
  --sort-by translation_error \
  --draw-mask-bbox \
  --bbox-match-mode nearest_projected_center \
  --frame-transform-side right
```
<!-- ------------------------ -->

```bash 
# 20260718 
dataset: real_20260718_rear_gsam_v4
# rear:/media/hdd2/ARCL_multicar_bags/camera_dataset/2026-07-18/rear/EPnPv2_gt_mesh_z_hybrid_labels

preds: gigaPose_datasets/results/large_real_20260718_rear_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260718_rear_gsam_v4-test_large_real_20260718_rear_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv


  CUDA_VISIBLE_DEVICES=1 python -m tracking.rgb_self_recovery.run \
  --predictions  gigaPose_datasets/results/large_real_20260718_rear_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260718_rear_gsam_v4-test_large_real_20260718_rear_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv  \
  --dataset-dir gigaPose_datasets/datasets/real_20260718_rear_gsam_v4 \
  --split test \
  --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
  --association-config tracking/configs/improved.json \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260718_rear_rotation_gated \
  --device cuda \
  --top-k-gigapose 5 \
  --beam-size 4 \
  --max-candidates 48 \
  --refinement-iterations 2 \
  --global-interval 5 \
  --broad-recovery-confidence 0.55 \
  --normal-confidence 0.65 \
  --lost-confidence 0.25 \
  --orientation-gates \
  --no-allow-flip-hypotheses \
  --normal-max-rotation-step-deg 30 \
  --uncertain-max-rotation-step-deg 60 \
  --max-rank0-rotation-disagreement-deg 90 \
  --save-overlays \
  --overlay-every 10 \
  --overlay-axis-length-m 1.0 \
  --overwrite



python -m tracking.select_real_label_candidates \
  --tracked-predictions gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260718_rear_rotation_gated/tracked_predictions.csv \
  --allowed-tracking-modes normal \
    --min-tracking-confidence 0.65 \
  --dataset-dir gigaPose_datasets/datasets/real_20260718_rear_gsam_v4 \
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
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260718_rear_rotation_gated/label_candidates_for_optimization


candidates:   out of 

python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260718_rear_rotation_gated/label_candidates_for_optimization/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260518v2v4_stereo_left_gsam_v4 \
  --split test \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260718_rear_rotation_gated/label_candidates_for_optimization/visual_overlays \
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



python -m fine_tuning.optimize_camera_lidar_extrinsics \
  --selected-samples gigaPose_datasets/results/rgb_self_recovery_realData_dataset/combined_front_selected_samples_for_optimization.csv \
  --gigapose-pose-source aligned \
  --use-sample-metadata \
  --epnp-label-dir-name EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-map-pose-unit m \
  --target-lidar-z-mode epnp_corrected \
  --timestamp-alignment interpolate_metadata \
  --interpolate-missing-lidar-timestamps \
  --timestamp-max-imputation-gap-ms 1500 \
  --timestamp-max-offset-jump-ms 250 \
  --timestamp-max-bracket-gap-ms 500 \
  --timestamp-fallback skip \
  --translation-residual-components xyz \
  --translation-sigma-mm 1000 \
  --rotation-sigma-deg 5 \
  --image-center-weight 2 \
  --image-center-sigma-px 50 \
  --image-center-map-z-mode raw \
  --projection-model metadata \
  --translation-prior-weight 10000 \
  --rotation-prior-weight 100 \
  --robust-loss soft_l1 \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/extrinsic_optimization_front_time_aligned_fixed


lets try optimization again: 
python -m fine_tuning.optimize_camera_lidar_extrinsics_centered \
  --selected-samples gigaPose_datasets/results/rgb_self_recovery_realData_dataset/combined_front_selected_samples_for_optimization.csv \
  --use-sample-metadata \
  --epnp-label-dir-name EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-map-pose-unit m \
  --raw-object-center-m -0.2411941141 0.0009010172 0.3329219520 \
  --gigapose-pose-source raw \
  --timestamp-alignment raw \
  --target-lidar-z-mode raw \
  --translation-residual-components xyz \
  --translation-sigma-mm 1000 \
  --rotation-sigma-deg 10 \
  --image-center-weight 0 \
  --projection-model metadata \
  --translation-prior-weight 1000 \
  --rotation-prior-weight 20 \
  --robust-loss soft_l1 \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/extrinsic_optimization_front_direct_centered






  python -m fine_tuning.select_real_label_candidates_with_camera_lidar_extrinsics \
  --gigapose-predictions gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_gsam_v4/tracked_predictions.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_front_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/front/EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --optimized-extrinsics gigaPose_datasets/results/rgb_self_recovery_realData_dataset/extrinsic_optimization_front_time_aligned_fixed/optimized_extrinsics.json \
  --frame-transform-side right \
  --frame-transform-refine-iterations 5 \
  --frame-transform-inlier-translation-mm 5000 \
  --frame-transform-inlier-rotation-deg 60 \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-camera-pose-key T_camera_object_centered \
  --epnp-map-pose-unit m \
  --epnp-camera-pose-unit m \
  --min-score 0.05 \
  --max-translation-error-mm 3000 \
  --max-rotation-error-deg 30 \
  --max-roll-error-deg 10 \
  --max-pitch-error-deg 10 \
  --max-yaw-error-deg 20 \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_gsam_v4/label_candidates_camera_lidar_time_aligned
 


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





---------------------------------------------------------------

python -m fine_tuning.optimize_camera_lidar_extrinsics_centered \
  --selected-samples gigaPose_datasets/results/rgb_self_recovery_realData_dataset/combined_front_selected_samples_for_optimization.csv \
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
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/extrinsic_optimization_front_aligned_centered_z_corrected





python -m tracking.select_real_label_candidates_with_centered_camera_lidar_extrinsics \
  --tracked-predictions gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_gsam_v4/tracked_predictions.csv \
  --allowed-tracking-modes normal \
  --min-tracking-confidence 0.65 \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_front_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/front/EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --optimized-extrinsics gigaPose_datasets/results/rgb_self_recovery_realData_dataset/extrinsic_optimization_front_aligned_centered_z_corrected/optimized_extrinsics.json \
  --frame-transform-side right \
  --frame-transform-refine-iterations 5 \
  --frame-transform-inlier-translation-mm 5000 \
  --frame-transform-inlier-rotation-deg 60 \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-camera-pose-key T_camera_object_centered \
  --epnp-map-pose-unit m \
  --epnp-camera-pose-unit m \
  --min-score 0.05 \
    --max-translation-error-mm 2000 \
  --max-rotation-error-deg 30 \
  --max-roll-error-deg 5 \
  --max-pitch-error-deg 5 \
  --max-yaw-error-deg 15 \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_gsam_v4/label_candidates_centered_calibration




python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_gsam_v4/label_candidates_centered_calibration/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_front_gsam_v4 \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_gsam_v4/label_candidates_centered_calibration/visual_overlays \
  --projection-model metadata \
  --max-images 100 \
  --draw-mask-bbox




1. Original versus tracked predictions, without EPnP

python -m tracking.compare_predictions_without_gt \
  --model original=gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260505v1_front_gsam_v4-test_large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --model tracked=gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_gsam_v4/tracked_predictions.csv \
  --reference original \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_front_gsam_v4 \
  --split test \
  --confidence-thresholds 0.0 0.35 0.5 0.65 0.8 \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_gsam_v4/comparison_with_original

This produces pose differences, confidence plots, distance plots, and coverage. It measures change, not accuracy.



2. Accuracy against corrected EPnP targets
Compare these files:
<ORIGINAL_SELECTION_OUTPUT>/best_candidate_per_epnp_label.csv
<TRACKED_SELECTION_OUTPUT>/best_candidate_per_epnp_label.csv

-----------------------------------------------------------------
 python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_gsam_v4/label_candidates_camera_lidar_time_aligned/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_front_gsam_v4 \
  --split test \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_gsam_v4/label_candidates_camera_lidar_time_aligned/visual_overlays \
  --frame-transform-side right \
  --max-images 100 \
  --draw-mask-bbox


  python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_gsam_v4/label_candidates_camera_lidar_time_aligned/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_front_gsam_v4 \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_gsam_v4/label_candidates_camera_lidar_time_aligned/visual_overlays_metadata_equidistant \
  --projection-model metadata \
  --frame-transform-side right \
  --draw-mask-bbox \
  --max-images 100








For front:
  ---------------------------------------------------------------

python -m fine_tuning.optimize_camera_lidar_extrinsics_centered \
  --selected-samples gigaPose_datasets/results/rgb_self_recovery_realData_dataset/combined_front_selected_samples_for_optimization.csv \
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
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/extrinsic_optimization_front_aligned_centered_z_corrected


gigapose/gigaPose_datasets/results/rgb_self_recovery_realData_dataset/extrinsic_optimization_front_aligned_centered_z_corrected/optimized_extrinsics.json


----------------------------------------------------------------
For rear:

python -m fine_tuning.optimize_camera_lidar_extrinsics_centered \
  --selected-samples gigaPose_datasets/results/rgb_self_recovery_realData_dataset/combined_rear_selected_samples_for_optimization.csv \
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
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/extrinsic_optimization_rear_aligned_centered_z_corrected


gigapose/gigaPose_datasets/results/rgb_self_recovery_realData_dataset/extrinsic_optimization_rear_aligned_centered_z_corrected/optimized_extrinsics.json


---------------------------------------------------------
For stereo_left:

python -m fine_tuning.optimize_camera_lidar_extrinsics_centered \
  --selected-samples gigaPose_datasets/results/rgb_self_recovery_realData_dataset/combined_stereo_left_selected_samples_for_optimization_new.csv \
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
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/extrinsic_optimization_stereo_left_aligned_centered_z_corrected



gigapose/gigaPose_datasets/results/rgb_self_recovery_realData_dataset/extrinsic_optimization_stereo_left_aligned_centered_z_corrected/optimized_extrinsics.json

----------------------------------------------------------------

For selecting new labels and then visualization: 

python -m tracking.select_real_label_candidates_with_centered_camera_lidar_extrinsics \
  --tracked-predictions gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_gsam_v4/tracked_predictions.csv \
  --allowed-tracking-modes normal \
  --min-tracking-confidence 0.65 \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_front_gsam_v4 \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/front/EPnPv2_gt_mesh_z_hybrid_labels \
  --epnp-glob "*.json" \
  --epnp-strip-trailing-instance-id \
  --epnp-key-prefix image_ \
  --match-key image_stem \
  --optimized-extrinsics gigaPose_datasets/results/rgb_self_recovery_realData_dataset/extrinsic_optimization_front_aligned_centered_z_corrected/optimized_extrinsics.json \
  --frame-transform-side right \
  --frame-transform-refine-iterations 5 \
  --frame-transform-inlier-translation-mm 5000 \
  --frame-transform-inlier-rotation-deg 60 \
  --epnp-map-pose-key T_map_object_raw \
  --epnp-camera-pose-key T_camera_object_centered \
  --epnp-map-pose-unit m \
  --epnp-camera-pose-unit m \
  --min-score 0.05 \
    --max-translation-error-mm 2000 \
  --max-rotation-error-deg 30 \
  --max-roll-error-deg 5 \
  --max-pitch-error-deg 5 \
  --max-yaw-error-deg 15 \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_gsam_v4/label_candidates_centered_calibration

python -m fine_tuning.visualize_epnp_gigapose_comparison_extrinsics \
  --candidate-csv gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_gsam_v4/label_candidates_centered_calibration/selected_samples.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_front_gsam_v4 \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_realData_dataset/rgb_self_recovery_real_20260505v1_front_gsam_v4/label_candidates_centered_calibration/visual_overlays \
  --projection-model metadata \
  --max-images 100 \
  --draw-mask-bbox

-----------------------------------------------------------------