# Assetto Corsa 20260623+ preparation

This folder handles recordings containing `csv/camera_frames.csv`,
`csv/transforms.csv`, and `images/{front,rear,stereo_left,stereo_right}`. Run
commands from `/home/anahita/gigapose` in the `gigapose` Conda environment.

The shared geometry code reads `T_camera_opponent_visual`, converts AC's
X-right/Y-up/Z-forward camera frame to OpenCV, and uses the same centered model
and pose convention for mask generation, inference, and training.

## Quick workflow map

| Stage | Command/module | Main inputs | Main outputs |
|---|---|---|---|
| Generate simulator-derived masks | `python -m Assetto_data_prep.generate_masks` | Raw Assetto session folders, CAD mesh | `<session>/generated_masks/`, optional `<session>/generated_masks_visib/` |
| Validate generated masks | `python -m Assetto_data_prep.validate_generated_masks` | Raw sessions + generated masks | JSON report with rendered-vs-saved mask IoU |
| Validate camera geometry | `python -m Assetto_data_prep.validate_camera_geometry` | Raw sessions, CAD mesh, masks | JSON report grouped by camera |
| Prepare benchmark/inference dataset | `python -m Assetto_data_prep.prepare_inference` | Raw sessions, CAD mesh, masks | `gigaPose_datasets/datasets/<dataset_name>/test/` and CNOS/FastSAM detections JSON |
| Render templates | `python -m src.scripts.render_custom_templates` | Prepared dataset CAD/model metadata | `gigaPose_datasets/datasets/templates/<dataset_name>/` |
| Run inference | `python test.py ...` | Prepared dataset + templates + checkpoint | `gigaPose_datasets/results/<experiment>/predictions/` |
| Split predictions by camera | `python -m fine_tuning.split_predictions_by_camera` | Prediction CSV/NPZ + dataset `frame_map.json` | `predictions/by_camera/<camera>/` |
| Overlay predictions | `python -m fine_tuning.overlay_gigapose_predictions` | Prediction CSV + prepared dataset | CAD-overlay debug images + `prediction_overlay_report.json` |
| Prepare fine-tuning dataset | `python -m Assetto_data_prep.prepare_training` | Raw sessions, CAD mesh, masks | `gigaPose_datasets/datasets/<dataset_name>/train_pbr_web/` and `val_pbr_web/` |
| Validate fine-tuning dataset | `python -m fine_tuning.validate_training_data` | Prepared fine-tuning dataset | Training-data sanity report/errors |
| Fine-tune GigaPose | `python -m fine_tuning.train` | Prepared fine-tuning dataset + checkpoint | `gigaPose_datasets/results/<run_name>/checkpoints/`, TensorBoard/W&B logs |
| Evaluate models | `python -m fine_tuning.evaluate_gigapose_models` | One or more prediction CSVs + benchmark dataset | Overall/camera summaries, per-instance metrics, pairwise comparisons |

## Important folders

| Folder/path pattern | What it contains |
|---|---|
| `/media/hdd2/ARCL_multicar_bags/camera_dataset/<session>/` | Raw Assetto Corsa session folders |
| `<session>/csv/camera_frames.csv` | Frame/image metadata from Assetto |
| `<session>/csv/transforms.csv` | Camera/object transform metadata from Assetto |
| `<session>/images/{front,rear,stereo_left,stereo_right}/` | Source RGB frames |
| `<session>/generated_masks/<camera>/` | Packed RGB instance masks; pixel value encodes opponent id |
| `<session>/generated_masks_visib/<camera>/` | Optional binary visible-mask PNGs, one per opponent |
| `gigaPose_datasets/datasets/<dataset_name>/test/` | Prepared inference/benchmark WebDataset split |
| `gigaPose_datasets/datasets/<dataset_name>/train_pbr_web/` | Prepared fine-tuning train split |
| `gigaPose_datasets/datasets/<dataset_name>/val_pbr_web/` | Prepared fine-tuning validation split |
| `gigaPose_datasets/datasets/cnos-fastsam/` | Detection JSON files consumed by GigaPose test loader |
| `gigaPose_datasets/datasets/templates/<dataset_name>/` | Rendered CAD templates |
| `gigaPose_datasets/results/<experiment>/predictions/` | GigaPose `.npz`, CSV, and MultiHypothesis CSV prediction outputs |
| `gigaPose_datasets/results/final_results/` | Final benchmark result folders and metric summaries |

## Common preparation options

| Option | Used by | Meaning / when to change it |
|---|---|---|
| `--cameras` | mask generation, inference prep, training prep | Choose cameras, e.g. `front,stereo`, `front,rear`, or `all`. |
| `--frame-stride` | inference prep, training prep | Subsample frames. Lower values give more data but more compute and temporal redundancy. |
| `--max-frames-per-session` | mask generation, training prep | Cap frames per source session. Useful for debugging or balancing large sessions. |
| `--mask-dir-name` | inference prep, training prep | Folder containing saved instance masks, e.g. `generated_masks`. |
| `--visible-mask-dir-name` | mask generation | Optional folder for per-opponent visible binary masks; pass empty string to disable. |
| `--min-mask-pixels` | inference prep, training prep | Drop very tiny detections/instances. |
| `--max-depth-m` | inference prep, training prep | Drop instances farther than this depth, useful for rear/tiny far-away cars. |
| `--cad-axis-convention` | mask generation, inference prep, training prep | Keep consistent across all prep steps; current successful default is `x-forward-z-up`. |
| `--fit-aabb` | mask generation, inference prep, training prep | Keep consistent across all prep steps; current successful default is `nonuniform`. |

## 1. Generate masks

First test a few frames:

```bash
SESSION=/path/to/frames/20260623_laguna2026_clear_4opp_noMask_2Laps

python -m Assetto_data_prep.generate_masks \
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset/20260623_putnam_snow_3opp_noMask_4Laps \
  --cad-path gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
  --cameras front \
  --max-frames-per-session 20 \
  --overwrite
```

Then generate every camera:
<!-- paths:  
/media/hdd2/ARCL_multicar_bags/camera_dataset/20260623_putnam_snow_3opp_noMask_4Laps (DONE)
/media/hdd2/ARCL_multicar_bags/camera_dataset/20260623_putnam_rain_2opp_noMask_6Laps (DONE)
/media/hdd2/ARCL_multicar_bags/camera_dataset/20260623_putnam_clear_2opp_noMask_fixedSkin_6Laps (DONE)
/media/hdd2/ARCL_multicar_bags/camera_dataset/20260623_laguna2026_haze_2opp_noMask_6Laps (DONE)
/media/hdd2/ARCL_multicar_bags/camera_dataset/20260623_laguna2026_cloudyThunder_2opp_fixedSkin_noMask_6Laps (DONE)
/media/hdd2/ARCL_multicar_bags/camera_dataset/20260623_laguna2026_clear_4opp_noMask_2Laps (DONE)
/media/hdd2/ARCL_multicar_bags/camera_dataset/20260623_laguna2026_clear_2opp_noMask_6Laps (DONE)
/media/hdd2/ARCL_multicar_bags/camera_dataset/20260622_putnam_clear_2opponent_noMask (DONE)
/media/hdd2/ARCL_multicar_bags/camera_dataset/20260627_laguna2026_clear_5opp_fixedskin  (DONE)
/media/hdd2/ARCL_multicar_bags/camera_dataset/20260627_laguna2026_fog_5opp_fixedskin (DONE)
/media/hdd2/ARCL_multicar_bags/camera_dataset/20260627_putnam_fog_5opp_fixedskin (DONE)
/media/hdd2/ARCL_multicar_bags/camera_dataset/20260627_laguna2026_clear_2opp_fixedskin_BENCHMARK 
-->
```bash
python -m Assetto_data_prep.generate_masks \
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset/20260623_putnam_snow_3opp_noMask_4Laps  \
  --cad-path gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
  --visible-mask-dir-name="visible_mask" \
  --cameras rear,front


  
```
--visible-mask-dir-name "" to disable these additional files.
Repeat `--source-root` for multiple recordings. Existing masks are skipped, so
an interrupted run can be resumed. Output is written to:

```text
<session>/generated_masks/<camera>/<sim_time_ms>_<frame>.png
<session>/generated_masks_visib/<camera>/<sim_time_ms>_<frame>_opp<id>.png
```

Each lossless RGB PNG stores `opponent_id = R + 256*G + 65536*B`; zero is
background. Small IDs look nearly black in a normal viewer. All cars are
rendered together, so they occlude one another. `generated_masks_visib` stores
the same visible result as one binary PNG per opponent (`0` background, `255`
visible car). Pass `--visible-mask-dir-name ''` to omit the separate files. The
render does not know about the ego car, barriers, or static scene geometry.

Defaults match the successful `projection.py` setup:

```text
--cad-axis-convention x-forward-z-up --fit-aabb nonuniform
```

Use identical alignment options in all three preparation commands.




geometry check:
```bash
python -m Assetto_data_prep.validate_camera_geometry \
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset/20260623_putnam_snow_3opp_noMask_4Laps \
  --cad-path gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
  --cameras rear \
  --mask-dir-name generated_masks \
  --frame-stride 2 \
  --output-json fine_tuning/rear_geometry_validation_report.json \
  --output-overlays fine_tuning/rear_geometry_overlays \
  --max-overlays 100


python -m fine_tuning.confirm_cad_coordinate_frame \
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset/20260623_putnam_snow_3opp_noMask_4Laps \
  --cad-path gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
  --camera rear \
  --num-frames 30 \
  --mask-dir-name generated_masks \
  --output-dir fine_tuning/rear_coordinate_check
```


## 2. Prepare fine-tuning data

Repeat `--source-root` in the desired order. The last session is held out by
default, rather than randomly mixing adjacent frames:

The old version is 
```bash
python -m Assetto_data_prep.prepare_training \
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset/20260623_putnam_snow_3opp_noMask_4Laps \
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset/20260623_putnam_rain_2opp_noMask_6Laps \
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset/20260623_putnam_clear_2opp_noMask_fixedSkin_6Laps  \
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset/20260623_laguna2026_haze_2opp_noMask_6Laps \
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset/20260623_laguna2026_cloudyThunder_2opp_fixedSkin_noMask_6Laps \
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset/20260623_laguna2026_clear_4opp_noMask_2Laps \
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset/20260623_laguna2026_clear_2opp_noMask_6Laps \
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset/20260622_putnam_clear_2opponent_noMask \
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset/20260627_laguna2026_clear_5opp_fixedskin \
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset/20260627_laguna2026_fog_5opp_fixedskin \
  --source-root  /media/hdd2/ARCL_multicar_bags/camera_dataset/20260627_putnam_fog_5opp_fixedskin \
  --cad-path gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
  --dataset-name assettocorsa_new_dataset \
  --cameras all \
  --frame-stride 2 \
  --max-frames-per-session 10000 \
  --mask-dir-name generated_masks \
  --validation-sessions 2 \
  --min-mask-pixels 100
  --max-depth-m 150 \
  --overwrite
```

The new version is : 
```bash
python -m Assetto_data_prep.prepare_training \
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset/20260623_putnam_snow_3opp_noMask_4Laps \
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset/20260623_putnam_rain_2opp_noMask_6Laps \
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset/20260623_putnam_clear_2opp_noMask_fixedSkin_6Laps  \
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset/20260623_laguna2026_haze_2opp_noMask_6Laps \
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset/20260623_laguna2026_cloudyThunder_2opp_fixedSkin_noMask_6Laps \
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset/20260623_laguna2026_clear_4opp_noMask_2Laps \
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset/20260623_laguna2026_clear_2opp_noMask_6Laps \
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset/20260622_putnam_clear_2opponent_noMask \
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset/20260627_laguna2026_clear_5opp_fixedskin \
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset/20260627_laguna2026_fog_5opp_fixedskin \
  --source-root  /media/hdd2/ARCL_multicar_bags/camera_dataset/20260627_putnam_fog_5opp_fixedskin \
  --cad-path gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
  --dataset-name assettocorsa_new_dataset \
  --cameras all \
  --frame-stride 2 \
  --max-frames-per-session 10000 \
  --mask-dir-name generated_masks \
  --split-mode random_frames \
  --validation-fraction 0.1 \
  --validation-seed 20260707 \
  --min-mask-pixels 100 \
  --max-depth-m 150 \
  --overwrite
```

Validate, render templates, and start with IST-only fine-tuning:
```bash
python -m fine_tuning.validate_training_data \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_new_dataset

python -m src.scripts.render_custom_templates \
  custom_dataset_name=assettocorsa_new_dataset \
  machine.num_workers=4
```

## 3. Run GSAM on rendered dataset

1. Run Grounded-SAM on the existing WebDataset split 
```bash
###########  for trainig dataset: ################3
python -m grounded_sam2_tracking_demo_Assetto_version \
  --input-split /home/appuser/gigapose/gigaPose_datasets/datasets/assettocorsa_new_dataset/train_pbr_web \
  --output-dir /home/appuser/gigapose/gigaPose_datasets/datasets/assettocorsa_new_dataset/train_pbr_web_gsam_filtered \
  --sam2-checkpoint ./checkpoints/sam2.1_hiera_small.pt \
  --sam2-model-cfg configs/sam2.1/sam2.1_hiera_s.yaml \
  --allowed-label-substrings "race car" \
  --text "race car." \
  --box-threshold 0.30 \
  --text-threshold 0.30 \
  --nms-iou 0.35 \
  --mask-nms-iou 0.60 \
  --overwrite

# --ego-filter-cameras all \
# --save-overlays \

###### For validation dataset: ##########
python -m grounded_sam2_tracking_demo_Assetto_version \
  --input-split /home/appuser/gigapose/gigaPose_datasets/datasets/assettocorsa_new_dataset/val_pbr_web \
  --output-dir /home/appuser/gigapose/gigaPose_datasets/datasets/assettocorsa_new_dataset/val_pbr_web_gsam_filtered \
  --sam2-checkpoint ./checkpoints/sam2.1_hiera_small.pt \
  --sam2-model-cfg configs/sam2.1/sam2.1_hiera_s.yaml \
  --allowed-label-substrings "race car" \
  --text "race car." \
  --box-threshold 0.30 \
  --text-threshold 0.30 \
  --nms-iou 0.35 \
  --mask-nms-iou 0.60 \
  --save-overlays \
  --overwrite
```


Next, do the actual clearing: (this is the step we need to do when you get back to the lab on wednesday)
```bash

# For train images: 
python -m Assetto_data_prep.filter_webdataset_by_detection_count \
  --input-split gigaPose_datasets/datasets/assettocorsa_new_dataset/train_pbr_web \
  --output-split gigaPose_datasets/datasets/assettocorsa_new_dataset/train_pbr_web_gsam_clean \
  --detections gigaPose_datasets/datasets/assettocorsa_new_dataset/train_pbr_web_gsam_filtered/gsam_detections.json \
  --min-score 0.05 \
  --overwrite

# For val images: 
python -m Assetto_data_prep.filter_webdataset_by_detection_count \
  --input-split gigaPose_datasets/datasets/assettocorsa_new_dataset/val_pbr_web \
  --output-split gigaPose_datasets/datasets/assettocorsa_new_dataset/val_pbr_web_gsam_clean \
  --detections gigaPose_datasets/datasets/assettocorsa_new_dataset/val_pbr_web_gsam_filtered/gsam_detections.json \
  --min-score 0.05 \
  --overwrite
  
``` 


to check the rejected samples: 
``` bash
python -m grounded_sam2_tracking_demo_Assetto_version \
  --input-split /home/appuser/gigapose/gigaPose_datasets/datasets/assettocorsa_new_dataset/val_pbr_web \
  --keys-csv /home/appuser/gigapose/gigaPose_datasets/datasets/assettocorsa_new_dataset/val_pbr_web_gsam_clean/rejected_samples.csv \
  --output-dir /home/appuser/gigapose/gigaPose_datasets/datasets/assettocorsa_new_dataset/rejected_val_gsam_overlays \
  --sam2-checkpoint ./checkpoints/sam2.1_hiera_small.pt \
  --sam2-model-cfg configs/sam2.1/sam2.1_hiera_s.yaml \
  --text "race car." \
  --allowed-label-substrings "race car" \
  --box-threshold 0.40 \
  --text-threshold 0.35 \
  --nms-iou 0.25 \
  --mask-nms-iou 0.50 \
  --save-overlays \
  --overwrite

``` 


## 4. Train
---------------------------------------------------------
AE and IST training:
---------------------------------------------------------
``` bash
python -m fine_tuning.train \
  --dataset-name assettocorsa_new_dataset \
  --checkpoint gigaPose_datasets/pretrained/gigaPose_v1.ckpt \
  --nets-to-train all \
  --ist-lr 1e-5 \
  --ae-lr 5e-8 \
  --ae-train-mode block-offsets \
  --ae-train-block-offsets 2,1 \
  --batch-size 32 \
  --max-steps 30000 \
  --validation-interval 500 \
  --run-name assettocorsa_ist_penultimate_last_ae_noRear_again  \
  --logger wandb \
  --print-loss-every 50 \
  --devices all \
  --match-sim-threshold 0.2
``` 

----------------------------------------------

IST only training: we are going to start with this
``` bash
  python -m fine_tuning.train \
  --dataset-name assettocorsa_new_dataset \
  --train-split train_pbr_web_gsam_clean \
  --validation-split val_pbr_web_gsam_clean \
  --checkpoint gigaPose_datasets/pretrained/gigaPose_v1.ckpt \
  --nets-to-train ist \
  --ist-lr 1e-5 \
  --batch-size 16 \
  --num-workers 4 \
  --max-steps 25000 \
  --validation-interval 150 \
  --heavy-validation \
  --heavy-validation-interval 150 \
  --heavy-validation-images 4 \
  --checkpoint-interval 1000 \
  --run-name assettocorsa_ist_only_july8_test_run3 \
  --logger wandb \
  --print-loss-every 50 \
  --devices 0 \
  --match-sim-threshold 0.2 
```
  
  
assettocorsa_ist_only_july8_newdataset

ok, new train mode: 
```bash

python -m fine_tuning.train_val \
  --dataset-name assettocorsa_new_dataset \
  --train-split train_pbr_web_gsam_clean \
  --validation-split val_pbr_web_gsam_clean \
  --checkpoint gigaPose_datasets/pretrained/gigaPose_v1.ckpt \
  --nets-to-train ist \
  --ist-lr 1e-5 \
  --batch-size 32 \
  --num-workers 4 \
  --max-steps 25000 \
  --validation-interval 250 \
  --heavy-validation \
  --heavy-validation-interval 250 \
  --heavy-validation-images 4 \
  --heavy-validation-seed 20260707 \
  --checkpoint-interval 1000 \
  --run-name assettocorsa_ist_only_july8_heavy_val_actual_run \
  --logger wandb \
  --print-loss-every 50 \
  --devices 0 \
  --match-sim-threshold 0.2

```


we are doing a new type of training with new losses and all: 
```bash
python -m fine_tuning.pose_aware_training.train \
  --dataset-name assettocorsa_new_dataset \
  --train-split train_pbr_web_gsam_clean \
  --validation-split val_pbr_web_gsam_clean \
  --checkpoint gigaPose_datasets/pretrained/gigaPose_v1.ckpt \
  --nets-to-train ist \
  --ist-lr 1e-5 \
  --batch-size 32 \
  --num-workers 4 \
  --max-steps 25000 \
  --validation-interval 250 \
  --checkpoint-interval 1000 \
  --run-name assettocorsa_pose_aware_ist_direct_pose_soft_antiflip_test_run4 \
  --logger wandb \
  --print-loss-every 50 \
  --devices 0 \
  --optimize-pose-monitor-errors \
  --direct-translation-weight 0.005 \
  --direct-depth-weight 0.01 \
  --direct-rotation-weight 0.0 \
  --reprojection-weight 0.2 \
  --log-depth-weight 1.5 \
  --log-depth-beta 0.05 \
  --anti-flip-weight 0.0





  python -m fine_tuning.pose_aware_training.train \
  --dataset-name assettocorsa_new_dataset \
  --train-split train_pbr_web_gsam_clean \
  --validation-split val_pbr_web_gsam_clean \
  --checkpoint gigaPose_datasets/pretrained/gigaPose_v1.ckpt \
  --nets-to-train ist \
  --ist-lr 5e-6 \
  --batch-size 32 \
  --num-workers 4 \
  --max-steps 12000 \
  --validation-interval 250 \
  --checkpoint-interval 1000 \
  --run-name assettocorsa_pose_aware_instance_scale_test_run \
  --logger wandb \
  --print-loss-every 50 \
  --devices 0 \
  --log-depth-weight 1.0 \
  --instance-log-scale-weight 0.5 \
  --scale-consistency-weight 0.05 \
  --log-depth-beta 0.05 \
  --inplane-weight 0.5 \
  --reprojection-weight 0.0 \
  --anti-flip-weight 0.0 \
  --best-scale-checkpoints 3



# Start from the current overall-best checkpoint and gently enable aggregated reprojection:
  python -m fine_tuning.pose_aware_training.train \
  --dataset-name assettocorsa_new_dataset \
  --train-split train_pbr_web_gsam_clean \
  --validation-split val_pbr_web_gsam_clean \
  --checkpoint gigaPose_datasets/results/assettocorsa_pose_aware_instance_scale_test_run/checkpoints/best-scale-step011750.ckpt \
  --nets-to-train ist \
  --ist-lr 1e-6 \
  --batch-size 32 \
  --num-workers 4 \
  --max-steps 10000 \
  --validation-interval 250 \
  --checkpoint-interval 1000 \
  --run-name assettocorsa_pose_aware_instance_scale_center_refinement_redo \
  --logger wandb \
  --print-loss-every 50 \
  --devices 0 \
  --log-depth-weight 1.0 \
  --instance-log-scale-weight 0.5 \
  --scale-consistency-weight 0.05 \
  --log-depth-beta 0.05 \
  --inplane-weight 0.5 \
  --reprojection-weight 0.02 \
  --reprojection-beta 0.05 \
  --anti-flip-weight 0.0

  # 
```
Watch these validation metrics:
val/monitor_scale_median_pred_gt_ratio: should approach 1.0.
val/monitor_scale_signed_log_bias: should approach 0.
val/monitor_scale_abs_log_error: lower is better.
val/monitor_scale_within_10pct: higher is better.
val/monitor_scale_log_std: lower means patch predictions agree better.

  <!-- --direct-translation-weight 0.01 \
  --direct-depth-weight 0.01 \
  --direct-rotation-weight 0.01 \
  --reprojection-weight 0.2 \
  --log-depth-weight 1.5 \
  --log-depth-beta 0.05 \
  --anti-flip-weight 0.01 \
  --anti-flip-margin 0.25 -->

If you want to first only monitor flips without optimizing anti-flip, use:
--anti-flip-weight 0.0  

else: 
--anti-flip-weight 0.01 


or also 
--log-depth-weight 2.0: makes the log-scale/depth loss count 2× more in the total IST loss.
--log-depth-beta 0.05: makes the Smooth-L1 loss sharper/more sensitive around small scale errors.


when fine tuning all nets, f train/infoNCE improves but val/matching gets worse, AE is overfitting; lower ae-lr or train fewer steps.

Then open Tensorboard with 
```bash
tensorboard --logdir /home/anahita/gigapose/gigaPose_datasets/results/assettocorsa_20260623_ist/tensorboard
```

Checkpoints will be saved here:
```bash
gigapose/gigaPose_datasets/results/assettocorsa_20260623_ist/checkpoints/
```

Validation images will be saved here:
```bash
gigapose/gigaPose_datasets/results/assettocorsa_20260623_ist/validation_images/
```




there is a totally new way of training, at first we update the AEnet/Dino and then use that to train the ISt: 
```bash
python -m fine_tuning.ot_training.train \
  --dataset-name assettocorsa_new_dataset \
  --train-split train_pbr_web_gsam_clean \
  --validation-split val_pbr_web_gsam_clean \
  --checkpoint gigaPose_datasets/results/assettocorsa_ot_ae_full_run/checkpoints/best-ot-step004750.ckpt \
  --ae-lr 1e-6 \
  --ae-train-mode last-blocks \
  --ae-train-last-n-blocks 1 \
  --batch-size 32 \
  --num-workers 4 \
  --max-steps 10000 \
  --validation-interval 250 \
  --checkpoint-interval 1000 \
  --feature-temperature 0.07 \
  --sinkhorn-iterations 30 \
  --correspondence-weight 1.0 \
  --soft-patch-reprojection-weight 0.25 \
  --soft-affine-center-weight 0.25 \
  --entropy-weight 0.0 \
  --hard-match-confidence 0.05 \
  --best-ot-checkpoints 3 \
  --run-name assettocorsa_ot_ae_full_run_cont \
  --logger wandb \
  --print-loss-every 50 \
  --devices all

```
After selecting the best OT checkpoint: we train IST: 
```bash
python -m fine_tuning.pose_aware_training.train \
  --dataset-name assettocorsa_new_dataset \
  --train-split train_pbr_web_gsam_clean \
  --validation-split val_pbr_web_gsam_clean \
  --checkpoint gigaPose_datasets/results/assettocorsa_ot_ae_smoke/checkpoints/best-ot-step002000.ckpt \
  --nets-to-train ist \
  --ist-lr 5e-6 \
  --batch-size 32 \
  --num-workers 4 \
  --max-steps 12000 \
  --validation-interval 250 \
  --checkpoint-interval 1000 \
  --run-name assettocorsa_ot_then_pose_aware_ist \
  --logger wandb \
  --print-loss-every 50 \
  --devices 0 \
  --log-depth-weight 1.0 \
  --instance-log-scale-weight 0.5 \
  --scale-consistency-weight 0.05 \
  --log-depth-beta 0.05 \
  --inplane-weight 0.5 \
  --reprojection-weight 0.0



python -m fine_tuning.pose_aware_training.train \
  --dataset-name assettocorsa_new_dataset \
  --train-split train_pbr_web_gsam_clean \
  --validation-split val_pbr_web_gsam_clean \
  --checkpoint gigaPose_datasets/results/assettocorsa_ot_ae_full_run/checkpoints/best-ot-step004750.ckpt \
  --nets-to-train ist \
  --ist-lr 5e-6 \
  --batch-size 16 \
  --num-workers 2 \
  --max-steps 20000 \
  --validation-interval 250 \
  --checkpoint-interval 1000 \
  --run-name assettocorsa_ot_fullrun_then_instance_scale_ist \
  --logger wandb \
  --print-loss-every 50 \
  --devices all \
  --log-depth-weight 1.0 \
  --instance-log-scale-weight 0.5 \
  --scale-consistency-weight 0.05 \
  --log-depth-beta 0.05 \
  --inplane-weight 0.75 \
  --reprojection-weight 0.0 \
  --anti-flip-weight 0.0 \
  --best-scale-checkpoints 3
```



## 4. Prepare and run inference

Use a held-out session:

```bash
BENCHMARK=/media/hdd2/ARCL_multicar_bags/camera_dataset/20260627_laguna2026_clear_2opp_fixedskin_BENCHMARK 

python -m Assetto_data_prep.prepare_inference \
  --source-root "$BENCHMARK" \
  --cad-path gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
  --dataset-name assettocorsa_benchmark_new_dataset \
  --cameras all \
  --frame-stride 2 \
  --max-depth-m 150 \
  --overwrite


python -m src.scripts.render_custom_templates \
  custom_dataset_name=assettocorsa_benchmark_new_dataset \
  machine.num_workers=4


###### For Benchmark dataset: ########## Run GSAM
python -m grounded_sam2_tracking_demo_Assetto_version \
  --input-split /home/appuser/gigapose/gigaPose_datasets/datasets/assettocorsa_benchmark_new_dataset/test \
  --output-dir /home/appuser/gigapose/gigaPose_datasets/datasets/assettocorsa_benchmark_new_dataset/test_gsam_filtered \
  --sam2-checkpoint ./checkpoints/sam2.1_hiera_small.pt \
  --sam2-model-cfg configs/sam2.1/sam2.1_hiera_s.yaml \
  --allowed-label-substrings "race car" \
  --text "race car." \
  --box-threshold 0.30 \
  --text-threshold 0.30 \
  --nms-iou 0.35 \
  --mask-nms-iou 0.60 \
  --save-overlays \
  --overwrite




Next, do the actual clearing: (this is the step we need to do when you get back to the lab on wednesday)

```bash
python -m Assetto_data_prep.filter_webdataset_by_detection_count \
  --input-split gigaPose_datasets/datasets/assettocorsa_benchmark_new_dataset/test \
  --output-split gigaPose_datasets/datasets/assettocorsa_benchmark_new_dataset/test_gsam_clean \
  --detections gigaPose_datasets/datasets/assettocorsa_benchmark_new_dataset/test_gsam_filtered/gsam_detections.json \
  --min-score 0.05 \
  --overwrite




# new chcekpoints are :_____________________
# for the  assettocorsa_pose_aware_instance_scale_center_refinement:

# --checkpoint \
# gigaPose_datasets/results/assettocorsa_pose_aware_instance_scale_center_refinement/checkpoints/best-scale-step002250.ckpt
  python test.py \
  test_dataset_name=assettocorsa_benchmark_new_dataset \
  "model.checkpoint_path='gigaPose_datasets/results/assettocorsa_pose_aware_instance_scale_center_refinement/checkpoints/best-scale-step002250.ckpt'" \
  run_id=assettocorsa_pose_aware_instance_scale_center_refinement_benchmark_new_dataset \
  name_exp=large_assettocorsa_pose_aware_instance_scale_center_refinement_benchmark_new_dataset



# prediction is : gigaPose_datasets/results/large_assettocorsa_pose_aware_instance_scale_center_refinement_benchmark_new_dataset/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_new_dataset-test_assettocorsa_pose_aware_instance_scale_center_refinement_benchmark_new_datasetMultiHypothesis.csv

 

# ----------------------------------------
# for assettocorsa_ot_then_instance_scale_ist
# gigaPose_datasets/results/assettocorsa_ot_then_instance_scale_ist/checkpoints/best-scale-step010250.ckpt
  python test.py \
  test_dataset_name=assettocorsa_benchmark_new_dataset \
  "model.checkpoint_path='gigaPose_datasets/results/assettocorsa_ot_then_instance_scale_ist/checkpoints/best-scale-step010250.ckpt'" \
  run_id=assettocorsa_ot_then_instance_scale_ist_benchmark_new_dataset \
  name_exp=large_assettocorsa_ot_then_instance_scale_ist_benchmark_new_dataset 

# prediction is : gigaPose_datasets/results/large_assettocorsa_ot_then_instance_scale_ist_benchmark_new_dataset/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_new_dataset-test_assettocorsa_ot_then_instance_scale_ist_benchmark_new_datasetMultiHypothesis.csv


# --------------------------------------------------
# gigaPose_datasets/results/new_dataset_ckeckpoints/assettocorsa_pose_aware_ist_new_dataset/checkpoints/last.ckpt
  python test.py \
  test_dataset_name=assettocorsa_benchmark_new_dataset \
  "model.checkpoint_path='gigaPose_datasets/results/new_dataset_ckeckpoints/assettocorsa_pose_aware_ist_new_dataset/checkpoints/last.ckpt'" \
  run_id=assettocorsa_pose_aware_ist_benchmark_new_dataset \
  name_exp=large_assettocorsa_pose_aware_ist_benchmark_new_dataset 


# prediction is : gigaPose_datasets/results/new_dataset_ckeckpoints/large_assettocorsa_pose_aware_ist_benchmark_new_dataset/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_new_dataset-test_assettocorsa_pose_aware_ist_benchmark_new_datasetMultiHypothesis.csv

# ------------------------------------------------------------
#  assettocorsa_ist_penultimate_last_ae_corrected ISTlr5e6
# gigaPose_datasets/results/last_picks/assettocorsa_ist_penultimate_last_ae_corrected_good/checkpoints/last.ckpt
  python test.py \
  test_dataset_name=assettocorsa_benchmark_new_dataset \
  "model.checkpoint_path='gigaPose_datasets/results/last_picks/assettocorsa_ist_penultimate_last_ae_corrected_good/checkpoints/last.ckpt'" \
  run_id=assettocorsa_ist_penultimate_last_ae_benchmark_new_dataset \
  name_exp=large_assettocorsa_ist_penultimate_last_ae_benchmark_new_dataset 

# prediction is: gigaPose_datasets/results/large_assettocorsa_ist_penultimate_last_ae_benchmark_new_dataset/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_new_dataset-test_assettocorsa_ist_penultimate_last_ae_benchmark_new_datasetMultiHypothesis.csv


--------------------------------------------------------------
# gigaPose_datasets/results/new_dataset_ckeckpoints/assettocorsa_pose_aware_instance_scale_center_refinement_redo/checkpoints/best-scale-step006250.ckpt

  python test.py \
  test_dataset_name=assettocorsa_benchmark_new_dataset \
  "model.checkpoint_path='gigaPose_datasets/results/new_dataset_ckeckpoints/assettocorsa_pose_aware_instance_scale_center_refinement_redo/checkpoints/best-scale-step006250.ckpt'" \
  run_id=assettocorsa_pose_aware_instance_scale_center_refinement_redo_benchmark_new_dataset \
  name_exp=large_assettocorsa_pose_aware_instance_scale_center_refinement_redo_benchmark_new_dataset
# prediction is: gigapose/gigaPose_datasets/results/large_assettocorsa_pose_aware_instance_scale_center_refinement_redo_benchmark_new_dataset/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_new_dataset-test_assettocorsa_pose_aware_instance_scale_center_refinement_redo_benchmark_new_datasetMultiHypothesis.csv

# --------------------------------------------------------
# gigaPose_datasets/results/new_dataset_ckeckpoints/assettocorsa_translation_rotation_residual/checkpoints/best-residual-step010000.ckpt

# This is the new testing:
  python test.py \
  test_dataset_name=assettocorsa_benchmark_new_dataset \
  "model.checkpoint_path='gigaPose_datasets/results/new_dataset_ckeckpoints/assettocorsa_translation_rotation_residual/checkpoints/best-residual-step010000.ckpt'" \
  run_id=assettocorsa_translation_rotation_residual_benchmark_new_dataset \
  name_exp=large_assettocorsa_translation_rotation_residual_benchmark_new_dataset


python -m fine_tuning.residual_pose_training.infer \
  --dataset-name assettocorsa_benchmark_new_dataset \
  --checkpoint gigaPose_datasets/results/new_dataset_ckeckpoints/assettocorsa_translation_rotation_residual/checkpoints/best-residual-step010000.ckpt \
  --run-name assettocorsa_translation_rotation_residual_benchmark_new_dataset \
  --batch-size 8 \
  --num-workers 4 \
  --devices 0 \
  --apply-residual \
  --rotation-residual \
  --max-rotation-deg 20 \
  --max-center-offset-px 56 \
  --max-log-depth-residual 0.5


# --------------------------
python -m fine_tuning.evaluate_pose_errors_by_distance \
  --model IST_pose_aware_center_refine=gigaPose_datasets/results/large_assettocorsa_pose_aware_instance_scale_center_refinement_benchmark_new_dataset/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_new_dataset-test_assettocorsa_pose_aware_instance_scale_center_refinement_benchmark_new_datasetMultiHypothesis.csv  \
  --model OT_IST=gigaPose_datasets/results/large_assettocorsa_ot_then_instance_scale_ist_benchmark_new_dataset/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_new_dataset-test_assettocorsa_ot_then_instance_scale_ist_benchmark_new_datasetMultiHypothesis.csv \
  --model IST_AE_older=gigaPose_datasets/results/large_assettocorsa_ist_penultimate_last_ae_benchmark_new_dataset/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_new_dataset-test_assettocorsa_ist_penultimate_last_ae_benchmark_new_datasetMultiHypothesis.csv   \
  --model IST_pose_aware_center_refine_redo=gigaPose_datasets/results/large_assettocorsa_pose_aware_instance_scale_center_refinement_redo_benchmark_new_dataset/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_new_dataset-test_assettocorsa_pose_aware_instance_scale_center_refinement_redo_benchmark_new_datasetMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark_new_dataset \
  --split test \
  --distance-bin-m 10 \
  --max-distance-m 120 \
  --output-dir gigaPose_datasets/results/new_dataset_ckeckpoints/metrics/IST_new_dataset_july11_compare3/model_distance_comparison



# remove IST_only_retrain, IST_pose_aware
# --model IST_pose_aware=gigaPose_datasets/results/new_dataset_ckeckpoints/large_assettocorsa_pose_aware_ist_benchmark_new_dataset/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_new_dataset-test_assettocorsa_pose_aware_ist_benchmark_new_datasetMultiHypothesis.csv \
# --model IST_only_retrain=gigaPose_datasets/results/large_assettocorsa_ist_only_july8_heavy_val_actual_run_benchmark_new_dataset/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_new_dataset-test_assettocorsa_ist_only_july8_heavy_val_actual_run_benchmark_new_datasetMultiHypothesis.csv \
# -------------------------------------------------------
# gigaPose_datasets/results/new_dataset_ckeckpoints/assettocorsa_pose_aware_ist_direct_pose_soft_antiflip_test_run4_new_dataset/checkpoints/last.ckpt
  python test.py \
  test_dataset_name=assettocorsa_benchmark_new_dataset \
  "model.checkpoint_path='gigaPose_datasets/results/new_dataset_ckeckpoints/assettocorsa_pose_aware_ist_direct_pose_soft_antiflip_test_run4_new_dataset/checkpoints/last.ckpt'" \
  run_id=assettocorsa_pose_aware_ist_direct_pose_soft_antiflip_test_run4_benchmark_new_dataset \
  name_exp=large_assettocorsa_pose_aware_ist_direct_pose_soft_antiflip_test_run4_benchmark_new_dataset




# gigaPose_datasets/results/new_dataset_ckeckpoints/assettocorsa_ist_only_july8_heavy_val_actual_run_new_dataset/checkpoints/last.ckpt
  python test.py \
  test_dataset_name=assettocorsa_benchmark_new_dataset \
  "model.checkpoint_path='gigaPose_datasets/results/new_dataset_ckeckpoints/assettocorsa_ist_only_july8_heavy_val_actual_run_new_dataset/checkpoints/last.ckpt'" \
  run_id=assettocorsa_ist_only_july8_heavy_val_actual_run_benchmark_new_dataset \
  name_exp=large_assettocorsa_ist_only_july8_heavy_val_actual_run_benchmark_new_dataset



# pred: gigaPose_datasets/results/large_assettocorsa_ist_only_july8_heavy_val_actual_run_benchmark_new_dataset/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_new_dataset-test_assettocorsa_ist_only_july8_heavy_val_actual_run_benchmark_new_datasetMultiHypothesis.csv

  python -m fine_tuning.overlay_gigapose_predictions \
  --predictions /home/anahita/gigapose/gigaPose_datasets/results/large_assettocorsa_assettocorsa_inference_run/predictions/large-pbrreal-rgb-mmodel_assettocorsa_inference-test_assettocorsa_assettocorsa_inference_runMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark \
  --split test \
  --output-dir fine_tuning/prediction_overlays_corrected \
  --min-score 0.01
```




## Comparing the models and results:

It compares two GigaPose CSVs against the Assetto GT stored inside the prepared dataset shards:

```bash
python -m fine_tuning.compare_gigapose_predictions \
  --baseline-predictions gigaPose_datasets/results/large_assettocorsa_assettocorsa_inference_run/predictions/large-pbrreal-rgb-mmodel_assettocorsa_inference-test_assettocorsa_assettocorsa_inference_runMultiHypothesis.csv \
  --finetuned-predictions gigaPose_datasets/results/large_assettocorsa_finetuned_corrected/predictions/large-pbrreal-rgb-mmodel_assettocorsa_inference-test_assettocorsa_assettocorsa_inference_correctedMultiHypothesis.csv \
  --baseline-name original \
  --finetuned-name finetuned \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_inference \
  --split test \
  --output-dir fine_tuning/prediction_gt_comparison



  python -m fine_tuning.evaluate_gigapose_models \
  --model IST_pose_aware=gigaPose_datasets/results/new_dataset_ckeckpoints/large_assettocorsa_pose_aware_ist_benchmark_new_dataset/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_new_dataset-test_assettocorsa_pose_aware_ist_benchmark_new_datasetMultiHypothesis.csv  \
  --model IST_direct_pose=gigaPose_datasets/results/new_dataset_ckeckpoints/large_assettocorsa_pose_aware_ist_direct_pose_soft_antiflip_test_run4_benchmark_new_dataset/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_new_dataset-test_assettocorsa_pose_aware_ist_direct_pose_soft_antiflip_test_run4_benchmark_new_datasetMultiHypothesis.csv  \
  --model Ist_heavy_val=gigaPose_datasets/results/large_assettocorsa_ist_only_july8_heavy_val_actual_run_benchmark_new_dataset/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_new_dataset-test_assettocorsa_ist_only_july8_heavy_val_actual_run_benchmark_new_datasetMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark_new_dataset \
  --split test \
  --rendered-iou \
  --output-dir gigaPose_datasets/results/new_dataset_ckeckpoints/metrics/IST_new_dataset_july10


```

it writes:
```bash
fine_tuning/prediction_gt_comparison/per_instance_metrics.csv
fine_tuning/prediction_gt_comparison/summary_metrics.csv
fine_tuning/prediction_gt_comparison/summary_metrics.json
```

Compare numerical performance over distance
```bash
python -m fine_tuning.evaluate_pose_errors_by_distance \
  --model IST_pose_aware=gigaPose_datasets/results/new_dataset_ckeckpoints/large_assettocorsa_pose_aware_ist_benchmark_new_dataset/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_new_dataset-test_assettocorsa_pose_aware_ist_benchmark_new_datasetMultiHypothesis.csv  \
  --model IST_direct_pose=gigaPose_datasets/results/new_dataset_ckeckpoints/large_assettocorsa_pose_aware_ist_direct_pose_soft_antiflip_test_run4_benchmark_new_dataset/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_new_dataset-test_assettocorsa_pose_aware_ist_direct_pose_soft_antiflip_test_run4_benchmark_new_datasetMultiHypothesis.csv  \
  --model Ist_heavy_val=gigaPose_datasets/results/large_assettocorsa_ist_only_july8_heavy_val_actual_run_benchmark_new_dataset/predictions/large-pbrreal-rgb-mmodel_assettocorsa_benchmark_new_dataset-test_assettocorsa_ist_only_july8_heavy_val_actual_run_benchmark_new_datasetMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark_new_dataset \
  --split test \
  --distance-bin-m 5 \
  --max-distance-m 100 \
  --output-dir gigaPose_datasets/results/new_dataset_ckeckpoints/metrics/IST_new_dataset_july10/model_distance_comparison 
```

After running the comparison script, run:
```bash
python -m fine_tuning.plot_prediction_comparison \
  --comparison-dir fine_tuning/prediction_gt_comparison

  or 

  python -m fine_tuning.plot_prediction_comparison   --input-csv gigaPose_datasets/results/new_dataset_ckeckpoints/metrics/IST_new_dataset_july10/all_instance_metrics.csv --comparison-dir gigaPose_datasets/results/new_dataset_ckeckpoints/metrics/IST_new_dataset_july10  --output-dir gigaPose_datasets/results/new_dataset_ckeckpoints/metrics/IST_new_dataset_july10/plots

  python -m fine_tuning.plot_model_summary \
  --metrics-dir gigaPose_datasets/results/new_dataset_ckeckpoints/metrics/IST_new_dataset_july10
```
It reads:
```bash
fine_tuning/prediction_gt_comparison/per_instance_metrics.csv
```
<!-- gigapose/gigaPose_datasets/results/final_results/metrics/IST_AE_July6_2047 -->
<!-- gigapose/gigaPose_datasets/results/last_picks/metrics/IST_AE_run2_vs_last -->
and writes plots to:
```bash
fine_tuning/prediction_gt_comparison/plots
```

If you want custom paths:
```bash
python -m fine_tuning.plot_prediction_comparison \
  --input-csv fine_tuning/prediction_gt_comparison/per_instance_metrics.csv \
  --output-dir fine_tuning/prediction_gt_comparison/plots

  python -m fine_tuning.plot_prediction_comparison   --input-csv gigaPose_datasets/results/last_picks/metrics/all_models/all_instance_metrics.csv   --comparison-dir gigaPose_datasets/results/last_picks/metrics/all_models   --output-dir gigaPose_datasets/results/last_picks/metrics/all_models/plots


  python -m fine_tuning.plot_prediction_comparison   --input-csv gigaPose_datasets/results/last_picks/metrics/IST_AE_run2_vs_last/all_instance_metrics.csv   --comparison-dir gigaPose_datasets/results/last_picks/metrics/IST_AE_run2_vs_last  --output-dir gigaPose_datasets/results/last_picks/metrics/IST_AE_run2_vs_last/plots

  python -m fine_tuning.plot_model_summary \
  --metrics-dir gigaPose_datasets/results/last_picks/metrics/all_models
```

```bash
python -m fine_tuning.visualize_prediction_gt_comparison \
  --baseline-predictions gigaPose_datasets/results/large_assettocorsa_assettocorsa_inference_run/predictions/large-pbrreal-rgb-mmodel_assettocorsa_inference-test_assettocorsa_assettocorsa_inference_runMultiHypothesis.csv \
  --finetuned-predictions gigaPose_datasets/results/large_assettocorsa_finetuned_corrected/predictions/large-pbrreal-rgb-mmodel_assettocorsa_inference-test_assettocorsa_assettocorsa_inference_correctedMultiHypothesis.csv \
  --baseline-name original \
  --finetuned-name finetuned \
  --dataset-dir gigaPose_datasets/datasets/  \
  --split test \
  --output-dir fine_tuning/prediction_gt_comparison/visual_overlays \
  --max-images 100
```

green box = GT/reference
red box = original
blue box = fine-tuned

## Consistency rules

- All opponents assigned object ID 1 must have the same geometry. Preparation
  stops if their AABB geometry differs.
- Re-render templates whenever the prepared model or alignment changes.
- Do not use `20260623_194543` or `20260623_194952`; both are empty.
- Recorded AC instance masks are still preferable when available because they
  contain scene and ego-car occlusion.














-----------------------------------------------------

Now this command can split both .csv and .npz files:
# gigaPose_datasets/results/large_assettocorsa_IST_only_benchmark/predictions
# gigaPose_datasets/results/final_results/large_assettocorsa_older_corrected_IST_only_benchmark/predictions
# gigaPose_datasets/results/final_results/large_assettocorsa_older_ist_2layerAE_benchmark/predictions
# gigaPose_datasets/results/final_results/large_assettocorsa_original_benchmark/predictions
```bash
python -m fine_tuning.split_predictions_by_camera \
  --predictions gigaPose_datasets/results/final_results/large_assettocorsa_original_benchmark/predictions \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark


python -m fine_tuning.overlay_gigapose_predictions \
  --predictions gigaPose_datasets/results/large_assettocorsa_IST_only_inference_corrected/predictions/large-pbrreal-rgb-mmodel_assettocorsa_inference-test_assettocorsa_IST_only_inference_correctedMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark \
  --split test \
  --output-dir fine_tuning/prediction_overlays_IST_only_inference_corrected \
  --min-score 0.01
```

or overlay the images seperately for each camera folder: 
```bash
(front only)
python -m fine_tuning.overlay_gigapose_predictions \
  --predictions gigaPose_datasets/results/large_assettocorsa_IST_only_benchmark/predictions/by_camera/front/large-pbrreal-rgb-mmodel_assettocorsa_benchmark-test_assettocorsa_IST_only_benchmarkMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark \
  --split test \
  --output-dir gigaPose_datasets/results/large_assettocorsa_IST_only_benchmark/overlays/front \
  --min-score 0.1

(rear only)
 python -m fine_tuning.overlay_gigapose_predictions \
  --predictions gigaPose_datasets/results/large_assettocorsa_IST_only_benchmark/predictions/by_camera/rear/large-pbrreal-rgb-mmodel_assettocorsa_benchmark-test_assettocorsa_IST_only_benchmarkMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark \
  --split test \
  --output-dir gigaPose_datasets/results/large_assettocorsa_IST_only_benchmark/overlays/rear \
  --min-score 0.1 

(stereo_left)
 python -m fine_tuning.overlay_gigapose_predictions \
  --predictions gigaPose_datasets/results/large_assettocorsa_IST_only_benchmark/predictions/by_camera/stereo_left/large-pbrreal-rgb-mmodel_assettocorsa_benchmark-test_assettocorsa_IST_only_benchmarkMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark \
  --split test \
  --output-dir gigaPose_datasets/results/large_assettocorsa_IST_only_benchmark/overlays/stereo_left \
  --min-score 0.1 

(stereo_right)
 python -m fine_tuning.overlay_gigapose_predictions \
  --predictions gigaPose_datasets/results/large_assettocorsa_IST_only_benchmark/predictions/by_camera/stereo_right/large-pbrreal-rgb-mmodel_assettocorsa_benchmark-test_assettocorsa_IST_only_benchmarkMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark \
  --split test \
  --output-dir gigaPose_datasets/results/large_assettocorsa_IST_only_benchmark/overlays/stereo_right \
  --min-score 0.1 
```
we are currectly using teh follwing results/preditions: 


<!-- /media/hdd2/ARCL_multicar_bags/camera_dataset/20260627_laguna2026_clear_2opp_fixedskin_BENCHMARK
 rsync -avP /home/anahita/gigapose/gigaPose_datasets/results/large_assettocorsa_IST_only_benchmark   /media/hdd2/ARCL_multicar_bags/camera_dataset/20260627_laguna2026_clear_2opp_fixedskin_BENCHMARK/
 -->
```bash
assettocorsa_ist_only_run_newdata
gigaPose_datasets/results/large_assettocorsa_IST_only_benchmark


large_assettocorsa_assettocorsa_inference_run (not bench mark, need to rerun on benchmark)
large_assettocorsa_finetuned_corrected (not benchmark, need to rerun on benchmark)

```

This writes the `test/` shards, centered CAD, targets, GT poses, frame map, and
`cnos-fastsam_assettocorsa-test.json` detections expected by the existing
`assettocorsa` configuration.

The generated masks depend on ground-truth poses. This is an oracle pipeline
check, not an unbiased inference benchmark. For a real evaluation, replace the
detection JSON with boxes and masks predicted independently from RGB.












---------------------------------------------------------
something new again:
new tarinig packages: residual_pose_training


Start with translation only and freeze the already-good IST:
```bash
python -m fine_tuning.residual_pose_training.train \
  --dataset-name assettocorsa_new_dataset \
  --train-split train_pbr_web_gsam_clean \
  --validation-split val_pbr_web_gsam_clean \
  --checkpoint gigaPose_datasets/results/assettocorsa_pose_aware_instance_scale_center_refinement_redo/checkpoints/best-scale-step006250.ckpt \
  --nets-to-train ist \
  --no-train-ist \
  --residual-lr 1e-4 \
  --batch-size 32 \
  --num-workers 4 \
  --max-steps 10000 \
  --validation-interval 250 \
  --checkpoint-interval 1000 \
  --run-name assettocorsa_translation_rotation_residual \
  --logger wandb \
  --print-loss-every 50 \
  --devices 0 \
  --residual-center-weight 1.0 \
  --residual-log-depth-weight 1.0 \
  --residual-translation-weight 0.05 \
  --residual-regularization-weight 0.001 \
  --rotation-residual \
  --residual-rotation-weight 0.25 \
  --max-rotation-deg 20 \
  --heavy-validation \
  --heavy-validation-interval 1000 \
  --heavy-validation-images 4
```
 <!-- --no-rotation-residual \ -->
Enable rotation refinement:
For a separate experiment, add:
```bash
--rotation-residual \
--residual-rotation-weight 0.25 \
--max-rotation-deg 20
```

Watch:
val/monitor_baseline_translation_error_mm
val/monitor_refined_translation_error_mm
val/monitor_translation_improvement_mm

val/monitor_baseline_depth_error_mm
val/monitor_refined_depth_error_mm
val/monitor_depth_improvement_mm

val/monitor_baseline_center_error_px
val/monitor_refined_center_error_px
val/monitor_center_improvement_px

val/monitor_baseline_rotation_error_deg
val/monitor_refined_rotation_error_deg 

Positive improvement means the residual head improved the original IST estimate.



End-to-end inference:
Use the new inference script, not the ordinary test.py, because the original model does not know how to apply these heads:
```bash
python -m fine_tuning.residual_pose_training.infer \
  --dataset-name assettocorsa_benchmark \
  --checkpoint gigaPose_datasets/results/assettocorsa_translation_residual/checkpoints/best-residual-stepXXXXXX.ckpt \
  --run-name assettocorsa_translation_residual_benchmark \
  --batch-size 16 \
  --num-workers 4 \
  --devices 0 \
  --no-rotation-residual \
  --max-center-offset-px 56 \
  --max-log-depth-residual 0.5
```

  For a rotation-enabled checkpoint, use:
  ```bash
  --rotation-residual \
--max-rotation-deg 20
```

Run refined inference with:
python -m fine_tuning.residual_pose_training.infer \
  --dataset-name assettocorsa_benchmark \
  --checkpoint gigaPose_datasets/results/assettocorsa_translation_residual/checkpoints/best-residual-stepXXXXXX.ckpt \
  --run-name assettocorsa_translation_residual_benchmark \
  --batch-size 16 \
  --num-workers 4 \
  --devices 0 \
  --apply-residual \
  --rotation-residual \
  --max-rotation-deg 20 \
  --max-center-offset-px 56 \
  --max-log-depth-residual 0.5

For a rotation-enabled model:
--rotation-residual \
--max-rotation-deg 20
--no-rotation-residual \

You can also generate a controlled baseline from the exact same checkpoint:
python -m fine_tuning.residual_pose_training.infer \
  --dataset-name assettocorsa_benchmark \
  --checkpoint gigaPose_datasets/results/assettocorsa_translation_residual/checkpoints/best-residual-stepXXXXXX.ckpt \
  --run-name assettocorsa_translation_residual_disabled_benchmark \
  --batch-size 16 \
  --num-workers 4 \
  --devices 0 \
  --no-apply-residual \
  --no-rotation-residual


