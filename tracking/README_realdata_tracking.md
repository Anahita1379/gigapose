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


python -m tracking.run_tracking \
  --predictions gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260505v1_front_gsam_v4-test_large_real_20260505v1_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_front_gsam_v4 \
  --split test \
  --config tracking/configs/improved.json \
  --output-dir gigaPose_datasets/results/real_world_ot2blocks_tracking/real_20260505v1_front_gsam_v4_tracking_rot_refine \
  --no-depth \
  --save-overlays \
  --overlay-every 10 \
  --identity-aware-association \
  --occlusion-aware-scoring \
  --overwrite

gigaPose_datasets/results/real_world_ot2blocks_tracking/real_20260505v1_front_gsam_v4_tracking/tracked_predictions.csv
#   --recovery-checkpoint gigaPose_datasets/results/tracking_recovery_head/best.ckpt \
```

```bash 
# 20260505v1 
dataset: gigaPose_datasets/datasets/real_20260505v1_rear_gsam_v4

rear: /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/rear/EPnPv2_gt_mesh_z_hybrid_labels

pred: gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260505v1_rear_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260505v1_rear_gsam_v4-test_large_real_20260505v1_rear_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv


python -m tracking.run_tracking \
  --predictions gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260505v1_rear_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260505v1_rear_gsam_v4-test_large_real_20260505v1_rear_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_rear_gsam_v4 \
  --split test \
  --config tracking/configs/improved.json \
  --output-dir gigaPose_datasets/results/real_world_ot2blocks_tracking/real_20260505v1_rear_gsam_v4_tracking \
  --recovery-checkpoint gigaPose_datasets/results/tracking_recovery_head/best.ckpt \
  --no-depth \
  --save-overlays \
  --overlay-every 10 \
  --identity-aware-association \
  --use-external-ids \
  --overwrite

```

```bash 
# 20260505v1 
dataset: gigaPose_datasets/datasets/real_20260505v1_stereo_left_gsam_v4

stereo_left: /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v1/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels

 pred:  gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260505v1_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260505v1_stereo_left_gsam_v4-test_large_real_20260505v1_stereo_left_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv


 python -m tracking.run_tracking \
  --predictions gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260505v1_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260505v1_stereo_left_gsam_v4-test_large_real_20260505v1_stereo_left_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_stereo_left_gsam_v4 \
  --split test \
  --config tracking/configs/improved.json \
  --output-dir gigaPose_datasets/results/real_world_ot2blocks_tracking/real_20260505v1_stereo_left_gsam_v4_tracking \
  --recovery-checkpoint gigaPose_datasets/results/tracking_recovery_head/best.ckpt \
  --no-depth \
  --save-overlays \
  --overlay-every 10 \
  --identity-aware-association \
  --use-external-ids \
  --overwrite

```
-------------------------------------------------------------------
-------------------------------------------------------------------

```bash 
# 20260505v2

rear: /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-05-12-28-13-v2/rear/EPnPv2_gt_mesh_z_hybrid_labels

 pred:  gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260505v2_rear_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260505v2_rear_gsam_v4-test_large_real_20260505v2_rear_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv

```

```bash 
# 20260518v0 

front: /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v0/front/EPnPv2_gt_mesh_z_hybrid_labels 

pred: gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v0_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v0_front_gsam_v4-test_large_real_20260518v0_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv
```

```bash 
# 20260518v0  

stereo_left: /media/hdd2/ARCL_multicar_bags/camera_dataset/2026-05-18-v0/stereo_left/EPnPv2_gt_mesh_z_hybrid_labels 

 pred:  gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v0_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v0_stereo_left_gsam_v4-test_large_real_20260518v0_stereo_left_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv
```


```bash 

 pred:  gigapose/gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v1v4_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v1v4_front_gsam_v4-test_large_real_20260518v1v4_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv
```

```bash 
 pred:  gigapose/gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v1v4_rear_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v1v4_rear_gsam_v4-test_large_real_20260518v1v4_rear_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv
```

```bash 
 pred:  gigapose/gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v1v4_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v1v4_stereo_left_gsam_v4-test_large_real_20260518v1v4_stereo_left_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv
```

```bash 
 pred:  gigapose/gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v2v4_front_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v2v4_front_gsam_v4-test_large_real_20260518v2v4_front_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv
```
```bash 
 pred:  gigapose/gigaPose_datasets/results/real_world_ot2_IST_tran_gigapose_results/large_real_20260518v2v4_stereo_left_gsam_v4_ot2blocks_IST_tran/predictions/large-pbrreal-rgb-mmodel_real_20260518v2v4_stereo_left_gsam_v4-test_large_real_20260518v2v4_stereo_left_gsam_v4_ot2blocks_IST_tranMultiHypothesis.csv
 ```