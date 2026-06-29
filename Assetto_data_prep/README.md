# Assetto Corsa 20260623+ preparation

This folder handles recordings containing `csv/camera_frames.csv`,
`csv/transforms.csv`, and `images/{front,rear,stereo_left,stereo_right}`. Run
commands from `/home/anahita/gigapose` in the `gigapose` Conda environment.

The shared geometry code reads `T_camera_opponent_visual`, converts AC's
X-right/Y-up/Z-forward camera frame to OpenCV, and uses the same centered model
and pose convention for mask generation, inference, and training.

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
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset/20260627_laguna2026_clear_2opp_fixedskin_BENCHMARK   \
  --cad-path gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
  --visible-mask-dir-name="" \
  --cameras all
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

## 2. Prepare and run inference

Use a held-out session:

```bash
VAL_SESSION=/media/hdd2/ARCL_multicar_bags/camera_dataset/20260623_putnam_snow_3opp_noMask_4Laps
BENCHMARK=/media/hdd2/ARCL_multicar_bags/camera_dataset/20260627_laguna2026_clear_2opp_fixedskin_BENCHMARK 

python -m Assetto_data_prep.prepare_inference \
  --source-root "$BENCHMARK" \
  --cad-path gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
  --dataset-name assettocorsa_benchmark \
  --cameras all \
  --frame-stride 5 \
  --overwrite

python -m src.scripts.render_custom_templates \
  custom_dataset_name=assettocorsa_benchmark \
  machine.num_workers=1

python test.py \
  test_dataset_name=assettocorsa_benchmark \
  run_id=assettocorsa_original_benchmark_run

  python -m fine_tuning.overlay_gigapose_predictions \
  --predictions /home/anahita/gigapose/gigaPose_datasets/results/large_assettocorsa_assettocorsa_inference_run/predictions/large-pbrreal-rgb-mmodel_assettocorsa_inference-test_assettocorsa_assettocorsa_inference_runMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_benchmark \
  --split test \
  --output-dir fine_tuning/prediction_overlays_corrected \
  --min-score 0.01
```

for testing the fine tuned model: 
```bash
python test.py \
  test_dataset_name=assettocorsa_benchmark \
  "model.checkpoint_path='gigaPose_datasets/results/assettocorsa_ist_only_run_newdata_good/checkpoints/epoch=13-step=17000.ckpt'" \
  run_id=assettocorsa_IST_only_benchmark \
  name_exp=large_assettocorsa_IST_only_benchmark

for older finetuned model:---------------------------
python test.py \
  test_dataset_name=assettocorsa_benchmark \
  "model.checkpoint_path='gigaPose_datasets/results/assettocorsa_ist_only_run_corrected_good/checkpoints/epoch=17-step=14000.ckpt'" \
  run_id=assettocorsa_older_corrected_IST_only_benchmark \
  name_exp=large_assettocorsa_older_corrected_IST_only_benchmark


python test.py \
  test_dataset_name=assettocorsa_benchmark \
  "model.checkpoint_path='gigaPose_datasets/results/assettocorsa_ist_penultimate_last_ae_corrected_good/checkpoints/last.ckpt'" \
  run_id=assettocorsa_older_ist_2layerAE_benchmark \
  name_exp=large_assettocorsa_older_ist_2layerAE_benchmark
-----------------------------------------------------

Now this command can split both .csv and .npz files:
# gigaPose_datasets/results/large_assettocorsa_IST_only_benchmark/predictions
# gigaPose_datasets/results/final_results/large_assettocorsa_older_corrected_IST_only_benchmark/predictions
# gigaPose_datasets/results/final_results/large_assettocorsa_older_ist_2layerAE_benchmark/predictions
# gigaPose_datasets/results/final_results/large_assettocorsa_original_benchmark/predictions
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

## 3. Prepare fine-tuning data

Repeat `--source-root` in the desired order. The last session is held out by
default, rather than randomly mixing adjacent frames:

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
  --dataset-name assettocorsa_all_no_rear \
  --cameras front,stereo_left,stereo_right \
  --frame-stride 2 \
  --max-frames-per-session 15000 \
  --mask-dir-name generated_masks \
  --validation-sessions 2 \
  --overwrite
```



to validate the generated masks before training: 
```bash
python -m Assetto_data_prep.validate_generated_masks \
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
  --cameras all \
  --frame-stride 20 \
  --mask-dir-name generated_masks \
  --output-json fine_tuning/generated_mask_validation_report.json
```

Generate masks using the same sessions, cameras, stride, and per-session cap.
The cap prevents the 45,686-frame Laguna session from overwhelming the other
weather/track conditions.

Validate, render templates, and start with IST-only fine-tuning:

```bash
python -m fine_tuning.validate_training_data \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_all

python -m src.scripts.render_custom_templates \
  custom_dataset_name=assettocorsa_all \
  machine.num_workers=1

AE and IST training:

python -m fine_tuning.train \
  --dataset-name assettocorsa \
  --checkpoint gigaPose_datasets/pretrained/gigaPose_v1.ckpt \
  --nets-to-train all \
  --ist-lr 1e-5 \
  --ae-lr 1e-6 \
  --ae-train-mode block-offsets \
  --ae-train-block-offsets 2,1 \
  --batch-size 32 \
  --max-steps 20000 \
  --validation-interval 150 \
  --run-name assettocorsa_ist_penultimate_last_ae_corrected_run2  \
  --logger wandb \
  --print-loss-every 50 \
  --devices all \
  --match-sim-threshold 0.2

IST only training: 

  python -m fine_tuning.train \
  --dataset-name assettocorsa_all \
  --checkpoint gigaPose_datasets/pretrained/gigaPose_v1.ckpt \
  --nets-to-train ist \
  --ist-lr 1e-4 \
  --batch-size 32 \
  --max-steps 20000 \
  --validation-interval 150 \
  --run-name assettocorsa_ist_only_run_newdata  \
  --logger wandb \
  --print-loss-every 50 \
  --devices all \
  --match-sim-threshold 0.2 
```
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
```

it writes:
```bash
fine_tuning/prediction_gt_comparison/per_instance_metrics.csv
fine_tuning/prediction_gt_comparison/summary_metrics.csv
fine_tuning/prediction_gt_comparison/summary_metrics.json
```

After running the comparison script, run:
```bash
python -m fine_tuning.plot_prediction_comparison \
  --comparison-dir fine_tuning/prediction_gt_comparison
```
It reads:
```bash
fine_tuning/prediction_gt_comparison/per_instance_metrics.csv
```

and writes plots to:
```bash
fine_tuning/prediction_gt_comparison/plots
```

If you want custom paths:
```bash
python -m fine_tuning.plot_prediction_comparison \
  --input-csv fine_tuning/prediction_gt_comparison/per_instance_metrics.csv \
  --output-dir fine_tuning/prediction_gt_comparison/plots
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
