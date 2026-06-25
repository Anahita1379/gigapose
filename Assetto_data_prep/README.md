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

/media/hdd2/ARCL_multicar_bags/camera_dataset/20260622_putnam_clear_2opponent_noMask
-->
```bash
python -m Assetto_data_prep.generate_masks \
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset/20260622_putnam_clear_2opponent_noMask \
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


python -m Assetto_data_prep.prepare_inference \
  --source-root "$VAL_SESSION" \
  --cad-path gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
  --dataset-name assettocorsa \
  --cameras front \
  --frame-stride 5 \
  --overwrite

python -m src.scripts.render_custom_templates \
  custom_dataset_name=assettocorsa \
  machine.num_workers=1

python test.py \
  test_dataset_name=assettocorsa \
  run_id=assettocorsa_oracle_masks
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
  --source-root /path/to/frames/20260623_195032 \
  --source-root /path/to/frames/20260623_laguna2026_clear_2opp_noMask_6Laps \
  --source-root /path/to/frames/20260623_laguna2026_clear_4opp_noMask_2Laps \
  --source-root /path/to/frames/20260623_laguna2026_haze_2opp_noMask_6Laps \
  --source-root /path/to/frames/20260623_putnam_clear_2opp_noMask_fixedSkin_6Laps \
  --source-root /path/to/frames/20260623_putnam_rain_2opp_noMask_6Laps \
  --source-root /path/to/frames/20260623_putnam_snow_3opp_noMask_4Laps \
  --cad-path gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
  --dataset-name assettocorsa \
  --cameras front \
  --frame-stride 5 \
  --max-frames-per-session 6000 \
  --validation-sessions 1 \
  --overwrite
```

Generate masks using the same sessions, cameras, stride, and per-session cap.
The cap prevents the 45,686-frame Laguna session from overwhelming the other
weather/track conditions.

Validate, render templates, and start with IST-only fine-tuning:

```bash
python -m fine_tuning.validate_training_data \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa

python -m src.scripts.render_custom_templates \
  custom_dataset_name=assettocorsa \
  machine.num_workers=1

python -m fine_tuning.train \
  --dataset-name assettocorsa \
  --checkpoint gigaPose_datasets/pretrained/gigaPose_v1.ckpt \
  --nets-to-train ist \
  --ist-lr 1e-5 \
  --batch-size 4 \
  --max-steps 5000 \
  --validation-interval 250 \
  --run-name assettocorsa_20260623_ist
```

## Consistency rules

- All opponents assigned object ID 1 must have the same geometry. Preparation
  stops if their AABB geometry differs.
- Re-render templates whenever the prepared model or alignment changes.
- Do not use `20260623_194543` or `20260623_194952`; both are empty.
- Recorded AC instance masks are still preferable when available because they
  contain scene and ego-car occlusion.
