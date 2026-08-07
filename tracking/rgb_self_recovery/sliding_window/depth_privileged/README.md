# Depth-privileged RGB self-recovery

This isolated package implements three DINO matching U-Net experiments. All
three use Assetto Corsa metric depth **only while training** and export the
existing `rgb_render_self_recovery_dino_matching_unet_v1` checkpoint format.
Inference therefore accepts only RGB, mask, and rendered CAD inputs.

The variants are:

1. `auxiliary_depth`: an RGB-only student has a dense depth head used only in
   the training loss. The head is removed from deployment checkpoints.
2. `teacher_student`: a depth-conditioned teacher is trained first. An RGB-only
   student then matches its pose outputs and multi-scale image features.
3. `combined`: the RGB-only student uses both teacher distillation and the
   training-only dense depth head.

The teacher and students train on independent candidate groups; they do not
consume temporal windows during training. Shuffling training shards therefore
cannot leak motion state between runs or cameras. At inference, every RGB-only
student uses the shared sequence-aware tracker: frames are grouped by
`source_run + camera_id`, ordered by physical source frame, and all track,
optical-flow, candidate-beam, and window state resets at frame/time gaps.

## 1. Prepare depth-augmented copies of the identical split

The existing recovery shards do not contain AC depth. Add it without changing
their RGB crops, candidates, or train/validation membership:

```bash
export DATA="$PWD/gigaPose_datasets/results/rgb_self_recovery_assetto_window_data"
export DEPTH_DATA="$PWD/gigaPose_datasets/results/rgb_self_recovery_assetto_depth_data"
export AC_DEPTH=/path/to/assetto/export

python3 -m tracking.rgb_self_recovery.sliding_window.depth_privileged.prepare_data \
  --input-data "$DATA/train" \
  --depth-root "$AC_DEPTH" \
  --depth-pattern '{source_run}/{camera_id}/depth/{frame_id:06d}.npy' \
  --depth-unit-scale 1.0 \
  --output-dir "$DEPTH_DATA/train"

python3 -m tracking.rgb_self_recovery.sliding_window.depth_privileged.prepare_data \
  --input-data "$DATA/validation" \
  --depth-root "$AC_DEPTH" \
  --depth-pattern '{source_run}/{camera_id}/depth/{frame_id:06d}.npy' \
  --depth-unit-scale 1.0 \
  --output-dir "$DEPTH_DATA/validation"
```

The pattern is a template, because AC exporters use different layouts. It may
contain `{source_key}`, `{source_run}`, `{camera_id}`, and `{frame_id}`. The
loader accepts `.npy`, `.npz`, PNG/TIFF, and OpenCV-readable EXR. Set
`--depth-unit-scale 0.001` for millimetres or `1.0` for metres. Verify whether
the AC depth buffer has already been linearized before generating data.

If sensor depth was not exported, generate CAD/transform-derived object
pseudo-depth in float32 metres first:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.depth_privileged.generate_pseudo_depth \
  --assetto-root /path/to/Assettocorsa_new_dataset_distance_bin_cleaned \
  --mesh gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
  --distance-bins distance0_20,distance20_40,distance40_60,distance60_80,distance80_100,distance100_120
```

This writes `depth_info/{front,rear}/*.npz` under each bin. Each file stores a
sparse object bounding box with `depth_m`, `valid`, `bbox_xyxy`, `image_shape`,
and `center_depth_m`. It is geometric pseudo-depth from the known CAD pose—not
an independent sensor measurement. Use the generated manifest coverage fields
to exclude poor mask/render alignments.

For these generated files, replace `--depth-root` and `--depth-pattern` in the
two preparation commands with:

```bash
--depth-manifest-root /path/to/Assettocorsa_new_dataset_distance_bin_cleaned
```

## 2. Train the depth-conditioned teacher

The teacher is needed by variants 2 and 3, but never by inference:

```bash
export MODELS="$PWD/gigaPose_datasets/results/rgb_self_recovery_backbone_comparison"

python3 -m tracking.rgb_self_recovery.sliding_window.depth_privileged.train_teacher \
  --data "$DEPTH_DATA/train" \
  --validation-data "$DEPTH_DATA/validation" \
  --initialize-from "$MODELS/dino_matching_unet/best_pose.ckpt" \
  --output-dir "$MODELS/depth_teacher" \
  --dino-mode frozen \
  --maximum-depth-m 120 \
  --epochs 100 --batch-size 8 --anti-flip-training \
  --patience 10 --logger wandb \
  --wandb-project rgb-self-recovery-depth --run-name depth-teacher \
  --device cuda
```

## 3. Train all three RGB-only students

For a controlled comparison, initialize every student from the same original
DINO matching U-Net checkpoint.

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.depth_privileged.auxiliary_depth.train \
  --data "$DEPTH_DATA/train" --validation-data "$DEPTH_DATA/validation" \
  --initialize-from "$MODELS/dino_matching_unet/best_pose.ckpt" \
  --output-dir "$MODELS/dino_matching_unet_aux_depth" \
  --maximum-depth-m 120 --auxiliary-depth-weight 0.25 \
  --epochs 100 --batch-size 8 --anti-flip-training --patience 10 --device cuda

python3 -m tracking.rgb_self_recovery.sliding_window.depth_privileged.teacher_student.train \
  --data "$DEPTH_DATA/train" --validation-data "$DEPTH_DATA/validation" \
  --initialize-from "$MODELS/dino_matching_unet/best_pose.ckpt" \
  --teacher-checkpoint "$MODELS/depth_teacher/best_pose.ckpt" \
  --output-dir "$MODELS/dino_matching_unet_teacher_student" \
  --maximum-depth-m 120 --distillation-output-weight 0.5 \
  --distillation-feature-weight 0.1 \
  --epochs 100 --batch-size 8 --anti-flip-training --patience 10 --device cuda

python3 -m tracking.rgb_self_recovery.sliding_window.depth_privileged.combined.train \
  --data "$DEPTH_DATA/train" --validation-data "$DEPTH_DATA/validation" \
  --initialize-from "$MODELS/dino_matching_unet/best_pose.ckpt" \
  --teacher-checkpoint "$MODELS/depth_teacher/best_pose.ckpt" \
  --output-dir "$MODELS/dino_matching_unet_depth_combined" \
  --maximum-depth-m 120 --auxiliary-depth-weight 0.25 \
  --distillation-output-weight 0.5 --distillation-feature-weight 0.1 \
  --epochs 100 --batch-size 8 --anti-flip-training --patience 10 --device cuda
```

Add the same WandB options to each student command when desired.

## 4. RGB-only inference

Use `best_pose.ckpt`. No depth argument exists in these runners:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.depth_privileged.auxiliary_depth.run \
  --predictions "$GIGAPOSE_PREDICTIONS" --dataset-dir "$EVAL_DATA" --split test \
  --checkpoint "$MODELS/dino_matching_unet_aux_depth/best_pose.ckpt" \
  --mesh "$EVAL_DATA/models/obj_000001.ply" \
  --output-dir "$EVAL_RUNS/dino_matching_unet_aux_depth" \
  --window-size 5 \
  --sequence-aware --sequence-max-frame-gap 1 --sequence-max-time-gap-s 0.5 \
  --orientation-gate-mode soft --save-overlays --device cuda
```

Replace the module/checkpoint/output names with `teacher_student` or `combined`
for the other two models. Every exported checkpoint records
`depth_required_at_inference: false`.

For the distilled teacher/student model specifically:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.depth_privileged.teacher_student.run \
  --predictions "$GIGAPOSE_PREDICTIONS" \
  --dataset-dir "$EVAL_DATA" --split test \
  --checkpoint "$MODELS/dino_matching_unet_teacher_student/best_pose.ckpt" \
  --mesh "$EVAL_DATA/models/obj_000001.ply" \
  --output-dir "$EVAL_RUNS/teacher_student_window5" \
  --window-size 5 \
  --sequence-aware --sequence-max-frame-gap 1 --sequence-max-time-gap-s 0.5 \
  --orientation-gate-mode soft --state-iou-confidence-weight 0.5 \
  --save-overlays --device cuda --overwrite
```

Use `--window-size 1` for the original sequential tracker without the added
window, or `--window-size 1 --per-frame` for independent frame-wise recovery.
Both modes use the same distilled RGB-only student checkpoint and load neither
the depth teacher nor observed depth.
