# Fine-tuning GigaPose on the Assetto Corsa recording

> **20260623+ schema:** recordings with `camera_frames.csv` and
> `transforms.csv` use [`Assetto_data_prep/README.md`](../Assetto_data_prep/README.md).
> The workflow below describes the older `bboxes_3d.csv` plus recorded-mask
> schema.

This folder is an isolated extension of the parent GigaPose checkout. It keeps
the dataset preparation, coordinate conversion, validation, configuration, and
training entry point together while importing the tested core model and utility
code from `src/`. Copying the complete upstream `src/` tree here would create a
second divergent GigaPose implementation, so only the components that differ
for fine-tuning are local to this folder.

## What the workflow produces

The corrected input recording supplies real RGB frames, camera intrinsics,
oriented 3D boxes, and visible per-opponent instance silhouettes. The opponent
CAD is the same model already at:

```text
gigaPose_datasets/datasets/racecar/models/obj_000001.ply
```

The preparation script centers that CAD, converts each CSV box frame to a
right-handed OpenCV CAD-to-camera pose, and renders CAD depth. Its training mask
is the intersection of the observed visible silhouette and rendered CAD
silhouette, ensuring that every supervised pixel has valid geometric depth.
It then creates:

```text
gigaPose_datasets/datasets/assettocorsa/
├── models/
│   ├── obj_000001.ply
│   ├── obj_000001.obj
│   └── models_info.json
├── models_info.json
├── train_pbr_web/
│   ├── shard-*.tar
│   └── key_to_shard.json
├── val_pbr_web/
│   ├── shard-*.tar
│   └── key_to_shard.json
├── fine_tuning_metadata.json
└── key_to_shard.json
```

Every shard sample contains RGB, rendered object depth, camera intrinsics, GT
CAD pose, visible masks, and BOP GT information.

### Which recording folders are used?

`prepare_ac_training_data.py` does not scan your disk automatically. It uses
only the recording roots passed with `--source-root`. If that option is omitted,
it uses the single built-in default:

```text
/mnt/ssd2tb/.local_share_backup/Steam/steamapps/common/assettocorsa/apps/lua/multi_cam_obs/frames/20260620_haze_3opp_withInstanceMask
```

Repeat the option once per recording to combine folders. Each root must contain
`csv/bboxes_3d.csv`, `images/<camera>/`, and lossless RGB/RGBA instance-ID PNGs
under `masks/<camera>/`. The CSV must contain `instance_id` and `mask_r/g/b`.
The processed dataset is written to `--output-root/--dataset-name`, whose
default is:

```text
gigaPose_datasets/datasets/assettocorsa
```

## 1. Activate the existing GigaPose environment

Run all commands from the repository root:

```bash
cd /home/anahita/gigapose
conda activate gigapose
```

The scripts use EGL through `pyrender`; they do not require a desktop display.

## 2. Confirm the CAD coordinate frame

Run the numerical and visual audit before generating the full dataset:

```bash
python -m fine_tuning.confirm_cad_coordinate_frame \
  --source-root /mnt/ssd2tb/.local_share_backup/Steam/steamapps/common/assettocorsa/apps/lua/multi_cam_obs/frames/20260620_haze_3opp_withInstanceMask \
  --cad-path gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
  --camera front \
  --num-frames 6
```

Inspect:

```text
fine_tuning/coordinate_check/coordinate_frame_report.json
fine_tuning/coordinate_check/recommended_front_*.jpg
fine_tuning/coordinate_check/yaw_180_alternative_front_*.jpg
```

The committed mapping was checked on frame 000000: the recommended overlay has
the CAD nose and rear wing aligned with the rendered car. Its matrix is:

```text
0,1,0, 0,0,1, 1,0,0
```

The numerical size comparison is approximate because AC records the collider or
LOD-D AABB. In the inspected recording, CAD/AC extent ratios are approximately
`[1.14, 0.82, 0.97]` after axis mapping. The CSV recorder documentation already
warns that these boxes need not tightly match the visible mesh.

If a later CAD uses another convention, pass a corrected row-major matrix with
`--cad-to-ac-body` to the preparation script.

## 3. Smoke-test pose conversion and rendering

Build a tiny dataset first:

```bash
python -m fine_tuning.prepare_ac_training_data \
  --cad-path gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
  --cameras front \
  --max-frames-per-session 20 \
  --validation-fraction 0.25 \
  --gap-frames 2 \
  --max-shard-size 10 \
  --overwrite

python -m fine_tuning.validate_training_data
```

This deliberately replaces only `train_pbr_web`, `val_pbr_web`, and the
centered model/metadata for `assettocorsa`. It does not modify the recording.

## 4. Build the full train/validation dataset

For the current single recording, validation is a contiguous tail and 50 source
frames before it are excluded as a leakage gap:

```bash
python -m fine_tuning.prepare_ac_training_data \
  --source-root /mnt/ssd2tb/.local_share_backup/Steam/steamapps/common/assettocorsa/apps/lua/multi_cam_obs/frames/20260620_haze_3opp_withInstanceMask \
  --cad-path gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
  --cameras front \
  --validation-fraction 0.20 \
  --gap-frames 50 \
  --max-shard-size 250 \
  --overwrite

python -m fine_tuning.validate_training_data
```

The default `--min-mask-overlap 0.25` rejects instances whose observed mask and
GT-pose CAD render disagree severely. On the first 10 front frames of the new
recording, the mean silhouette IoU is about `0.70`, and the intersection keeps
about `81%` of observed car pixels.

Use `--cameras front,stereo_left,stereo_right` to add those cameras. Rear frames
are useful only where an opponent annotation actually exists.

For multiple independent recordings, repeat `--source-root`. The final session
is held out in its entirety by default:

```bash
python -m fine_tuning.prepare_ac_training_data \
  --source-root /path/to/session_1 \
  --source-root /path/to/session_2 \
  --source-root /path/to/session_3 \
  --validation-sessions 1 \
  --cameras front \
  --overwrite
```

This is preferable to randomly splitting adjacent 10 Hz frames.

## Build the inference dataset with true instance masks

The inference converter now decodes the exact instance color from each PNG,
uses the visible-mask bounding box, and stores one COCO-RLE detection per
visible opponent:

```bash
python -m src.scripts.prepare_AC_metadata_dataset \
  --source-root /mnt/ssd2tb/.local_share_backup/Steam/steamapps/common/assettocorsa/apps/lua/multi_cam_obs/frames/20260620_haze_3opp_withInstanceMask \
  --cad-path gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
  --dataset-name assettocorsa \
  --cameras front \
  --overwrite
```

`--overwrite` replaces only the inference `test/` split and inference metadata;
it preserves `train_pbr_web/` and `val_pbr_web/`. Fully occluded CSV instances
with no mask pixels are skipped. The CAD is centered by default so training,
templates, inference, and visualization share one coordinate frame.

## Visualize GigaPose predictions as CAD silhouettes

`coordinate_check` uses GT poses from `bboxes_3d.csv`: it converts each GT pose,
renders the CAD with `pyrender`/EGL, and alpha-blends the rendered silhouette on
the original RGB image. To perform the same check with GigaPose's predicted
poses, run:

```bash
python -m fine_tuning.overlay_gigapose_predictions \
  --predictions /home/anahita/gigapose/gigaPose_datasets/results/large_assettocorsa_oracle_masks/predictions/large-pbrreal-rgb-mmodel_assettocorsa-test_assettocorsa_oracle_masks.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa \
  --split test \
  --output-dir fine_tuning/prediction_overlays
```

The script groups all CSV rows by `(scene_id, im_id)` and draws every predicted
car in one image. `src/scripts/visualize_racecar_predictions.py` has also been
updated to group instances, so later cars no longer overwrite earlier ones.

The output report records the number of CSV predictions per image:

```text
fine_tuning/prediction_overlays/prediction_overlay_report.json
```

Use the exact mesh that was used to render the inference templates. The default
is `<dataset-dir>/models/obj_000001.ply`. For a MultiHypothesis CSV, the script
keeps the highest-score hypothesis for each explicit `instance_id`.

If translations in the CSV are millimeters while the mesh is in meters, add
`--translation-scale 0.001`. The current custom GigaPose inference CSV normally
uses translations in the mesh's meter units, so the default scale is `1.0`.

## 5. Render templates from the centered training CAD

The preparation script writes a centered CAD. Render a fresh template bank from
that exact file; do not reuse the old uncentered `racecar` templates:

```bash
python -m src.scripts.render_custom_templates \
  custom_dataset_name=assettocorsa \
  machine.num_workers=1
```

Expected output:

```text
gigaPose_datasets/datasets/templates/assettocorsa/000001/
gigaPose_datasets/datasets/templates/assettocorsa/object_poses/000001.npy
```

## 6. Fine-tune from `gigaPose_v1.ckpt`

The local training entry point loads the complete pretrained checkpoint as an
initialization, creates a fresh optimizer, and trains only IST by default:

```bash
python -m fine_tuning.train \
  --dataset-name assettocorsa \
  --checkpoint gigaPose_datasets/pretrained/gigaPose_v1.ckpt \
  --nets-to-train ist \
  --ist-lr 1e-5 \
  --batch-size 4 \
  --max-steps 5000 \
  --validation-interval 250 \
  --run-name assettocorsa_ist_finetune
```

Checkpoints and TensorBoard logs are written beneath:

```text
gigaPose_datasets/results/assettocorsa_ist_finetune/
```

IST-only training freezes the DINO appearance/matching network and updates the
relative scale/in-plane network. This is the safer first experiment for one
highly correlated recording. If held-out validation improves, a later low-rate
joint run can use:

```bash
python -m fine_tuning.train \
  --nets-to-train all \
  --ae-lr 1e-6 \
  --ist-lr 1e-5 \
  --run-name assettocorsa_all_finetune
```


## CAD overlay on preditions: 
```bash
python -m fine_tuning.overlay_gigapose_predictions \
  --predictions /home/anahita/gigapose/gigaPose_datasets/results/large_ac_instance_masks/predictions/large-pbrreal-rgb-mmodel_assettocorsa-test_ac_instance_masks.csv \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa \
  --split test \
  --output-dir fine_tuning/prediction_overlays
```
It composites every predicted car onto one image and writes a count report:

```text
fine_tuning/prediction_overlays/prediction_overlay_report.json
```

Use the exact dataset mesh used for template rendering. Add --translation-scale 0.001 only if the prediction translations are millimeters while the CAD is in meters.


## Important limitations

- The corrected PNGs are true visible per-instance silhouettes and are used
  directly for inference. Fine-tuning intersects them with the CAD render so
  all target-mask pixels also have valid rendered depth.
- Rendered depth contains the opponent CADs only. That is intentional: GigaPose
  uses it to establish geometric correspondences.
- AC's masks already include occlusion by nearer cars, the ego car, and opaque
  scene geometry. The remaining depth is CAD-rendered because the recording
  does not contain metric depth.
- Pose correctness depends on the centered CAD representing the same car and on
  the visual coordinate audit. Always inspect several overlays from a new
  recording before training.
- A single mostly stationary recording is not enough to demonstrate useful
  generalization. Add sessions with different ranges, orientations, lighting,
  tracks, and opponent placements, then hold out entire sessions.
