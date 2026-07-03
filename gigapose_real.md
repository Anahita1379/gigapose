# Real-world GigaPose label pipeline

This note is a runbook for using GigaPose to create pose labels on the real
ARCL bag-derived datasets, compare them against transponder/EPnPv2 labels, and
use the good matches to optimize the map-to-camera extrinsics.

The requested raw data root is:

```text
/media/hdd2/ARCL_multicar_bags/camera_dataset/
```

Focus on sessions from:

| Date | Meaning |
|---|---|
| `2026-05-05` | Real-world bag-derived dataset sessions |
| `2026-05-18` | Real-world bag-derived dataset sessions |
| `2026-05-26` | Real-world bag-derived dataset sessions |

The EPnPv2 labels should be read from the actual pose label files under:

```text
EPnPv2_labels/
```

Do not compare against only metadata if actual pose-label files are present.

## New helper scripts

| Script | Purpose |
|---|---|
| `python -m fine_tuning.select_real_label_candidates` | Load GigaPose predictions and EPnPv2 labels, align frames, select good candidate matches |
| `python -m fine_tuning.optimize_camera_map_extrinsics` | Optimize a correction to the map-to-camera extrinsic using selected samples |

These are templates with flexible parsers. The exact EPnPv2 schema may require
passing the correct `--epnp-key-field`, `--epnp-glob`, and initial extrinsic
file path.

## Overall workflow

| Step | Input | Output |
|---|---|---|
| Prepare real GigaPose inference dataset | Bag-extracted images, camera intrinsics, masks/detections | `gigaPose_datasets/datasets/<real_dataset_name>/test/` |
| Run GigaPose | Prepared real dataset, CAD templates, checkpoint | GigaPose prediction CSV / MultiHypothesis CSV |
| Load EPnPv2 labels | `EPnPv2_labels` actual pose files | Parsed EPnPv2 pose table |
| Align frames | GigaPose pose CSV + EPnPv2 labels | `frame_transform_gigapose_to_epnp.json` |
| Select good candidates | Aligned GigaPose labels vs EPnPv2 labels | `selected_samples.csv` |
| Optimize extrinsics | `selected_samples.csv` + initial `T_map_cam` | `optimized_extrinsics.json` |

## 1. Generate labels with GigaPose

The real-world labeling stage needs RGB images, intrinsics, CAD templates, and
detections/masks. If you do not already have car masks, run Grounded-SAM2 or
another detector/segmenter first and convert its masks into the detection format
used by the GigaPose test loader.

### Prepare a Grounded-SAM v4 folder for GigaPose

For folders with this layout:

```text
.../frames/<session>/<camera>/
  images/
  images_jpg_new/
  metadata/
  Grounded_Sam_v4/
    metadata.json
    masks_png/
    masks_npy/
```

use:

```bash
python -m Assetto_data_prep.prepare_grounded_sam_inference \
  --source-root /mnt/ssd2tb/.local_share_backup/Steam/steamapps/common/assettocorsa/apps/lua/multi_cam_obs/frames/2026-05-26-12-19-49/front \
  --cad-path gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
  --dataset-name real_20260526_front_gsam_v4 \
  --grounded-sam-dir Grounded_Sam_v4 \
  --overwrite
```

This creates:

```text
gigaPose_datasets/datasets/real_20260526_front_gsam_v4/test/
gigaPose_datasets/datasets/real_20260526_front_gsam_v4/models/
gigaPose_datasets/datasets/real_20260526_front_gsam_v4/test_targets_bop19.json
gigaPose_datasets/datasets/real_20260526_front_gsam_v4/frame_map.json
gigaPose_datasets/datasets/cnos-fastsam/cnos-fastsam_real_20260526_front_gsam_v4-test.json
```

The preparer reads camera intrinsics from the per-frame YAML files under
`metadata/`. If those are missing, pass a fallback:

```bash
--camera-k fx,fy,cx,cy
```

For a quick test, add:

```bash
--max-frames-per-session 20
```

Then render templates for the new dataset name:

```bash
python -m src.scripts.render_custom_templates \
  custom_dataset_name=real_20260526_front_gsam_v4 \
  machine.num_workers=1
```

Then run GigaPose:

```bash
python test.py \
  test_dataset_name=real_20260526_front_gsam_v4 \
  run_id=real_20260526_front_gsam_v4_original \
  name_exp=large_real_20260526_front_gsam_v4_original
```

For a fine-tuned checkpoint:

```bash
python test.py \
  test_dataset_name=real_20260526_front_gsam_v4 \
  "model.checkpoint_path='gigaPose_datasets/results/<train_run>/checkpoints/last.ckpt'" \
  run_id=real_20260526_front_gsam_v4_finetuned \
  name_exp=large_real_20260526_front_gsam_v4_finetuned
```

For this step, the practical goal is to produce a normal GigaPose prediction
CSV, such as:

```text
gigaPose_datasets/results/<real_run_name>/predictions/<...>MultiHypothesis.csv
```

Example shape of the run:

```bash
python test.py \
  test_dataset_name=<real_dataset_name> \
  "model.checkpoint_path='gigaPose_datasets/pretrained/gigaPose_v1.ckpt'" \
  run_id=<real_run_name> \
  name_exp=large_<real_run_name>
```

If using a fine-tuned model, replace the checkpoint:

```bash
python test.py \
  test_dataset_name=<real_dataset_name> \
  "model.checkpoint_path='gigaPose_datasets/results/<train_run>/checkpoints/last.ckpt'" \
  run_id=<real_run_name> \
  name_exp=large_<real_run_name>
```

## 2. Compare GigaPose labels to EPnPv2 labels

The script:

```text
fine_tuning/select_real_label_candidates.py
```

loads:

- GigaPose prediction CSV
- optional `frame_map.json` from the prepared GigaPose dataset
- actual pose files from `EPnPv2_labels`

It writes:

```text
frame_transform_gigapose_to_epnp.json
all_candidate_pairs.csv
best_candidate_per_epnp_label.csv
selected_samples.csv
selection_report.json
```

### Basic command

```bash
python -m fine_tuning.select_real_label_candidates \
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset \
  --date 2026-05-05 \
  --date 2026-05-18 \
  --date 2026-05-26 \
  --gigapose-predictions gigaPose_datasets/results/<real_run_name>/predictions/<prediction_file>MultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/<real_dataset_name> \
  --epnp-glob "**/*.csv" \
  --epnp-key-field filename \
  --min-score 0.05 \
  --max-translation-error-mm 2000 \
  --max-rotation-error-deg 30 \
  --output-dir gigaPose_datasets/results/real_world_label_candidates/<real_run_name>
```

If the EPnPv2 labels are JSON files:

```bash
python -m fine_tuning.select_real_label_candidates \
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset \
  --gigapose-predictions gigaPose_datasets/results/<real_run_name>/predictions/<prediction_file>MultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/<real_dataset_name> \
  --epnp-glob "**/*.json" \
  --epnp-key-field filename \
  --output-dir gigaPose_datasets/results/real_world_label_candidates/<real_run_name>
```

If all EPnPv2 labels are already in one folder:

```bash
python -m fine_tuning.select_real_label_candidates \
  --gigapose-predictions gigaPose_datasets/results/<real_run_name>/predictions/<prediction_file>MultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/<real_dataset_name> \
  --epnp-root /media/hdd2/ARCL_multicar_bags/camera_dataset/<session>/EPnPv2_labels \
  --epnp-glob "**/*.csv" \
  --epnp-key-field filename \
  --output-dir gigaPose_datasets/results/real_world_label_candidates/<real_run_name>
```

## 3. Matching keys

GigaPose predictions are indexed by:

```text
scene_id
im_id
instance_id
```

EPnPv2 labels are usually indexed by one of:

```text
filename
image_path
frame_id
timestamp
sim_time_ms
```

The candidate-selection script tries to normalize filenames to stems, but for
best results you should choose the actual EPnPv2 field:

| EPnPv2 label field | Recommended option |
|---|---|
| `filename` | `--epnp-key-field filename` |
| `image_path` | `--epnp-key-field image_path` |
| `frame_id` | `--epnp-key-field frame_id --match-key frame_id` |
| `sim_time_ms` | `--epnp-key-field sim_time_ms --match-key sim_time_ms` |

If the prepared GigaPose dataset has a useful `frame_map.json`, pass:

```bash
--dataset-dir gigaPose_datasets/datasets/<real_dataset_name>
```

or:

```bash
--frame-map /path/to/frame_map.json
```

## 4. Coordinate frame alignment

The selection script estimates a constant transform:

```text
T_epnp ≈ T_epnp_gigapose @ T_gigapose
```

and saves it as:

```text
frame_transform_gigapose_to_epnp.json
```

This handles the case where GigaPose and EPnPv2 use different pose frames.

The automatic estimate uses easy one-to-one matches where a frame has exactly
one GigaPose prediction and one EPnPv2 label. If that is not reliable, provide a
manually defined transform:

```bash
--frame-transform-json /path/to/frame_transform_gigapose_to_epnp.json
```

The transform file should contain:

```json
{
  "T_epnp_gigapose": [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ]
}
```

Translations are in millimeters.

## 5. Selected sample file

The main output used downstream is:

```text
selected_samples.csv
```

It contains one accepted GigaPose/EPnPv2 pair per row:

| Column | Meaning |
|---|---|
| `match_key` | Shared frame/image key |
| `scene_id`, `im_id`, `instance_id` | GigaPose prediction identifiers |
| `score` | GigaPose confidence score |
| `epnp_label_path` | Actual EPnPv2 pose-label file |
| `epnp_record_index` | Row/object index inside the EPnPv2 file |
| `translation_error_mm` | Error after frame alignment |
| `rotation_error_deg` | Rotation error after frame alignment |
| `T_gigapose_cam_obj` | Original GigaPose camera-object pose |
| `T_gigapose_aligned_epnp_obj` | GigaPose pose after frame alignment |
| `T_epnp_obj` | EPnPv2 object pose |

You can tighten or loosen candidate selection with:

```bash
--max-translation-error-mm 1000
--max-rotation-error-deg 20
--min-score 0.1
```

## 6. Optimize map-to-camera extrinsics

After selecting reasonable GigaPose candidates, run:

```text
fine_tuning/optimize_camera_map_extrinsics.py
```

This optimizes:

```text
T_map_cam_optimized = T_correction @ T_map_cam_initial
```

using the selected pairs:

```text
T_epnp_obj ≈ T_map_cam_optimized @ T_gigapose_cam_obj
```

### Initial extrinsic file

Create a JSON file with the current camera-to-map extrinsic:

```json
{
  "T_map_cam": [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ]
}
```

By default, translations are expected in millimeters. If your JSON translation
is in meters, pass:

```bash
--initial-unit m
```

### Optimization command

```bash
python -m fine_tuning.optimize_camera_map_extrinsics \
  --selected-samples gigaPose_datasets/results/real_world_label_candidates/<real_run_name>/selected_samples.csv \
  --initial-extrinsic path/to/current_T_map_cam.json \
  --initial-unit mm \
  --translation-prior-weight 10 \
  --rotation-prior-weight 1 \
  --translation-sigma-mm 1000 \
  --rotation-sigma-deg 10 \
  --output-dir gigaPose_datasets/results/real_world_extrinsics/<real_run_name>
```

The important outputs are:

```text
optimized_extrinsics.json
optimization_report.json
errors_before_optimization.csv
errors_after_optimization.csv
```

## 7. Why translation correction is penalized more

The optimization includes a prior on the extrinsic correction:

```text
translation_prior_weight = 10
rotation_prior_weight = 1
```

That means the optimizer is discouraged from changing translation too much, but
is allowed to adjust rotation more freely. This matches the assumption that the
translation part of the current extrinsics is more accurate than the rotational
part.

If the optimized translation correction is still large, that is a red flag:

- the GigaPose/EPnPv2 matching key may be wrong
- the coordinate-frame transform may be wrong
- the EPnPv2 pose unit may be wrong
- the initial `T_map_cam` direction may be inverted
- the selected samples may include bad GigaPose labels

## 8. Recommended debug order

Before trusting the optimized extrinsics:

1. Inspect `selection_report.json`.
2. Check that `selected_samples.csv` has enough rows from several sessions.
3. Sort `selected_samples.csv` by `translation_error_mm` and inspect the worst cases.
4. Visualize a handful of selected frames with GigaPose overlays.
5. Confirm whether `T_map_cam` means camera-to-map or map-to-camera in the downstream pipeline.
6. Run optimization.
7. Compare `errors_before_optimization.csv` vs `errors_after_optimization.csv`.
8. Check that the correction is physically plausible.

## 9. Common pitfalls

| Symptom | Likely cause |
|---|---|
| Very few selected samples | Matching key mismatch, strict thresholds, low GigaPose scores |
| Huge translation errors everywhere | Unit mismatch, wrong frame transform, wrong initial extrinsic direction |
| Rotation error near 180 degrees | Axis convention mismatch or front/back ambiguity |
| Good image-plane overlays but bad EPnPv2 comparison | EPnPv2 frame differs from GigaPose frame or extrinsic calibration is off |
| Optimization changes translation a lot | Bad selected samples or wrong initial `T_map_cam` direction |
| Only one date/session contributes samples | EPnPv2 label discovery/glob is missing other folders |

## 10. Minimal command skeleton

```bash
# 1) Run GigaPose on real data first; this produces a prediction CSV.

# 2) Select good GigaPose/EPnPv2 pairs.
python -m fine_tuning.select_real_label_candidates \
  --source-root /media/hdd2/ARCL_multicar_bags/camera_dataset \
  --date 2026-05-05 \
  --date 2026-05-18 \
  --date 2026-05-26 \
  --gigapose-predictions gigaPose_datasets/results/<real_run_name>/predictions/<prediction_file>MultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/<real_dataset_name> \
  --epnp-glob "**/*.csv" \
  --epnp-key-field filename \
  --output-dir gigaPose_datasets/results/real_world_label_candidates/<real_run_name>

# 3) Optimize map-camera extrinsics.
python -m fine_tuning.optimize_camera_map_extrinsics \
  --selected-samples gigaPose_datasets/results/real_world_label_candidates/<real_run_name>/selected_samples.csv \
  --initial-extrinsic path/to/current_T_map_cam.json \
  --output-dir gigaPose_datasets/results/real_world_extrinsics/<real_run_name>
```
