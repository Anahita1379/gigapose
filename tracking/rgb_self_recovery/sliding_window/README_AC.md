1. Generate the training data
Current unique-image counts are:
0–20:     3311
20–40:    3591
40–60:    1166
60–80:     651
80–100:    578
100–120:   519

For perfectly equal bins, use 519 from each bin: 3,114 images total. 

```bash 

cd /home/anahita/gigapose
conda activate gigapose

AC_EXPORT=/mnt/ssd2tb/.local_share_backup/Steam/steamapps/common/assettocorsa/apps/lua/multi_cam_obs/frames/Assettocorsa_new_dataset_copy
CAD=gigaPose_datasets/datasets/racecar/models/obj_000001.ply
DATA=gigaPose_datasets/results/rgb_self_recovery_assetto_window_data

BINS=distance0_20,distance20_40,distance40_60,distance60_80,distance80_100,distance100_120
```

Generate training shards:
```bash 

python3 -m tracking.rgb_self_recovery.sliding_window.generate_dataset \
  --assetto-export-root "$AC_EXPORT" \
  --assetto-split train \
  --assetto-validation-run 20260730_putnam_rain_1opp_2lap_farRear \
  --assetto-validation-run 20260730_laguna2026_fog_1opp_2laps_mostlyFront \
  --assetto-cameras front,rear \
  --assetto-distance-bins "$BINS" \
  --assetto-frames-per-bin 519 \
  --assetto-short-bin-policy error \
  --mesh "$CAD" \
  --cad-axis-convention x-forward-z-up \
  --fit-aabb nonuniform \
  --assetto-max-depth-m 120 \
  --candidates-per-instance 12 \
  --output-dir "$DATA/train" \
  --overwrite
```

Generate validation shards using the identical selection configuration:
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.generate_dataset \
  --assetto-export-root "$AC_EXPORT" \
  --assetto-split validation \
  --assetto-validation-run 20260730_putnam_rain_1opp_2lap_farRear \
  --assetto-validation-run 20260730_laguna2026_fog_1opp_2laps_mostlyFront \
  --assetto-cameras front,rear \
  --assetto-distance-bins "$BINS" \
  --assetto-frames-per-bin 519 \
  --assetto-short-bin-policy error \
  --mesh "$CAD" \
  --cad-axis-convention x-forward-z-up \
  --fit-aabb nonuniform \
  --assetto-max-depth-m 120 \
  --candidates-per-instance 12 \
  --output-dir "$DATA/validation" \
  --overwrite

```

Shaded-region cropping is off by default. No flag is needed.
If you prefer keeping 529 in the first five bins and all 519 in the last bin, use:
```bash
--assetto-frames-per-bin 529 \
--assetto-short-bin-policy all
```
Updated both isolated CNN and DINO trainers. They now save:
best_loss.ckpt and compatible alias best.ckpt
best_rotation.ckpt
best_center.ckpt
best_depth.ckpt
best_pose.ckpt — normalized center + depth + rotation
last.ckpt 

### Training:
1. CNN baseline:
```bash
DATA=gigaPose_datasets/results/rgb_self_recovery_assetto_window_data
MODELS=gigaPose_datasets/results/rgb_self_recovery_backbone_comparison

python3 -m tracking.rgb_self_recovery.sliding_window.train \
  --data "$DATA/train" \
  --validation-data "$DATA/validation" \
  --output-dir "$MODELS/cnn" \
  --epochs 200 \
  --batch-size 32 \
  --anti-flip-training \
  --patience 12 \
  --min-delta 1e-4 \
  --logger wandb \
  --wandb-project rgb-self-recovery-backbones \
  --run-name cnn \
  --device cuda
```

```bash
export DATA="$PWD/gigaPose_datasets/results/rgb_self_recovery_assetto_window_data"
export MODELS="$PWD/gigaPose_datasets/results/rgb_self_recovery_backbone_comparison"

python3 -m tracking.rgb_self_recovery.sliding_window.train \
  --data "$DATA/train" \
  --validation-data "$DATA/validation" \
  --output-dir "$MODELS/cnn_multicheckpoint" \
  --epochs 200 \
  --batch-size 32 \
  --anti-flip-training \
  --patience 12 \
  --min-delta 1e-4 \
  --logger wandb \
  --wandb-project rgb-self-recovery-backbones \
  --run-name cnn-multicheckpoint \
  --device cuda
```



2. Frozen DINOv2-S/14:
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.train_dino \
  --dino-mode frozen \
  --dino-model dinov2_vits14 \
  --data "$DATA/train" \
  --validation-data "$DATA/validation" \
  --output-dir "$MODELS/dino_frozen" \
  --epochs 200 \
  --batch-size 4 \
  --anti-flip-training \
  --patience 8 \
  --min-delta 1e-4 \
  --logger wandb \
  --wandb-project rgb-self-recovery-backbones \
  --run-name dino-frozen \
  --device cuda
```


```bash
python3 -m tracking.rgb_self_recovery.sliding_window.train_dino \
  --dino-mode frozen \
  --dino-model dinov2_vits14 \
  --data "$DATA/train" \
  --validation-data "$DATA/validation" \
  --output-dir "$MODELS/dino_frozen_multicheckpoint" \
  --epochs 200 \
  --batch-size 32 \
  --anti-flip-training \
  --patience 10 \
  --min-delta 1e-4 \
  --logger wandb \
  --wandb-project rgb-self-recovery-backbones \
  --run-name dino-frozen-multicheckpoint \
  --device cuda

```
3. Tune the final DINO block
This starts from the best frozen-DINO result:
```bash
export DATA="$PWD/gigaPose_datasets/results/rgb_self_recovery_assetto_window_data"
export MODELS="$PWD/gigaPose_datasets/results/rgb_self_recovery_backbone_comparison"

test -f "$MODELS/dino_frozen_multicheckpoint/best_pose.ckpt" && \
  echo "Frozen-DINO checkpoint found"

python3 -m tracking.rgb_self_recovery.sliding_window.train_dino \
  --dino-mode last_block \
  --dino-model dinov2_vits14 \
  --initialize-from "$MODELS/dino_frozen_multicheckpoint/best_pose.ckpt" \
  --data "$DATA/train" \
  --validation-data "$DATA/validation" \
  --output-dir "$MODELS/dino_last_block" \
  --learning-rate 5e-5 \
  --dino-learning-rate 1e-5 \
  --epochs 60 \
  --batch-size 32 \
  --anti-flip-training \
  --patience 8 \
  --min-delta 1e-4 \
  --logger wandb \
  --wandb-project rgb-self-recovery-backbones \
  --run-name dino-last-block \
  --device cuda
```
Use best_pose.ckpt as the first checkpoint for end-to-end sliding-window evaluation. Eventually compare best_pose.ckpt, best_loss.ckpt, and possibly the specialized checkpoints against real GigaPose predictions—the held-out translation/rotation errors and overlays remain the final decision criteria. 

Every CNN and DINO run saves these useful checkpoint paths:

- `best.ckpt`: backward-compatible alias of `best_loss.ckpt`.
- `best_loss.ckpt`: lowest complete validation objective.
- `best_rotation.ckpt`: lowest validation rotation error.
- `best_center.ckpt`: lowest validation crop-center error.
- `best_depth.ckpt`: lowest validation absolute log-depth error.
- `best_pose.ckpt`: lowest normalized center-plus-depth-plus-rotation score.
- `last.ckpt`: final epoch, primarily for diagnostics.

The pose score is the equal-weight mean of `center_px/max_center_crop_px`,
`log_depth_error/max_log_depth`, and `rotation_deg/max_rotation_deg`.
Early stopping still watches total validation loss, while all best
checkpoints continue to update until training stops. For the end-to-end pose
comparison, start with `best_pose.ckpt`; evaluate the specialized center and
rotation checkpoints too if their tradeoff is useful. The final model choice
must still be based on held-out GigaPose inference and overlays.


### Evaluation: 
It compares all three models on exactly matched ground-truth frames and produces:
Overall translation and rotation statistics.
Median errors by distance bin.
Paired DINO-versus-CNN improvements.
Translation, rotation, and joint win fractions.
Per-frame CSV metrics.
Translation and rotation plots.
Combined CAD overlays for visual inspection.



# Run inference first
Run the sliding-window pipeline three times on the same held-out sequential dataset:
```bash
rgb_window_eval/cnn/tracked_predictions.csv
rgb_window_eval/dino_frozen/tracked_predictions.csv
rgb_window_eval/dino_last_block/tracked_predictions.csv
```
Only change --checkpoint and --output-dir between runs. All other inference settings must remain identical.

# Evaluate
```bash
EVAL_DATA=gigaPose_datasets/datasets/PREPARED_HELD_OUT_SEQUENCE
EVAL_RUNS=gigaPose_datasets/results/rgb_window_eval

python3 -m tracking.rgb_self_recovery.sliding_window.evaluate_backbones \
  --dataset-dir "$EVAL_DATA" \
  --split test \
  --model cnn="$EVAL_RUNS/cnn" \
  --model dino_frozen="$EVAL_RUNS/dino_frozen" \
  --model dino_last_block="$EVAL_RUNS/dino_last_block" \
  --baseline cnn \
  --mesh "$EVAL_DATA/models/obj_000001.ply" \
  --distance-bins-m 0,20,40,60,80,100,120 \
  --max-overlays 120 \
  --output-dir "$EVAL_RUNS/backbone_comparison" \
  --overwrite

```
Use the aligned CAD from the prepared evaluation dataset, not the original unaligned racecar CAD.

# Outputs
backbone_comparison/
├── summary.json
├── per_frame_metrics.csv
├── translation_error_by_distance.png
├── rotation_error_by_distance.png
└── overlays/

Overlay colors follow the model argument order:
Ground truth: green
CNN: red
Frozen DINO: blue
Last-block DINO: orange

Each overlay prints translation and rotation error for every model.


In summary.json, improvement is calculated as:
```bash
CNN error − DINO error
```
Therefore:
Positive improvement means DINO is better.
Negative improvement means CNN is better.
translation_better_fraction > 0.5 means DINO wins translation on most paired frames.
both_better_fraction requires DINO to improve both translation and rotation on the same frame.
The evaluator currently expects the one-car protocol: exactly one ground-truth opponent per frame. All ten tests and static checks pass.


# The new architecture has:
Separate RGB+mask and CAD-render encoders.
Four-resolution feature matching.
U-Net decoder with spatial skip connections.
A 64×64 heatmap and soft-argmax for center correction.
Global heads for depth, rotation, confidence, and quality.
Existing ranking and anti-flip losses.
About 2.98M parameters, versus 3.64M for the current CNN.
Cached image features across CAD candidates.
The same multi-checkpoint selection system.
Its own five-frame inference entry point.

```bash
export DATA="$PWD/gigaPose_datasets/results/rgb_self_recovery_assetto_window_data"
export MODELS="$PWD/gigaPose_datasets/results/rgb_self_recovery_backbone_comparison"

python3 -m tracking.rgb_self_recovery.sliding_window.matching_unet.train \
  --data "$DATA/train" \
  --validation-data "$DATA/validation" \
  --output-dir "$MODELS/matching_unet" \
  --epochs 200 \
  --batch-size 32 \
  --anti-flip-training \
  --heatmap-weight 0.10 \
  --heatmap-sigma-px 4 \
  --patience 10 \
  --min-delta 1e-4 \
  --logger wandb \
  --wandb-project rgb-self-recovery-backbones \
  --run-name matching-unet \
  --device cuda
```

Use this checkpoint first afterward: 
```bash
"$MODELS/matching_unet/best_pose.ckpt"
```
The package includes its own inference command:
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.matching_unet.run \
  --predictions "$GIGAPOSE_PREDICTIONS" \
  --dataset-dir "$EVAL_DATA" \
  --split test \
  --checkpoint "$MODELS/matching_unet/best_pose.ckpt" \
  --mesh "$EVAL_DATA/models/obj_000001.ply" \
  --output-dir "$EVAL_RUNS/matching_unet" \
  --window-size 5 \
  --save-overlays \
  --device cuda
```


# Model 5:
What model 5 contains

RGB + mask CNN pyramid ─────────────┐
                                   │
Frozen DINO 16×16 patch features ──┼─→ matching U-Net
                                   │       │
Candidate CAD geometry pyramid ────┘       ├─ center heatmap
                                           └─ depth/rotation/ranking
                                                      │
                                           five-frame optimizer


# Train model 5
It uses the identical existing train/validation split. Do not regenerate the data.

```bash
export DATA="$PWD/gigaPose_datasets/results/rgb_self_recovery_assetto_window_data"
export MODELS="$PWD/gigaPose_datasets/results/rgb_self_recovery_backbone_comparison"

python3 -m tracking.rgb_self_recovery.sliding_window.dino_matching_unet.train \
  --dino-mode frozen \
  --dino-model dinov2_vits14 \
  --dino-input-size 224 \
  --data "$DATA/train" \
  --validation-data "$DATA/validation" \
  --output-dir "$MODELS/dino_matching_unet" \
  --epochs 200 \
  --batch-size 32 \
  --anti-flip-training \
  --heatmap-weight 0.10 \
  --heatmap-sigma-px 4 \
  --patience 10 \
  --min-delta 1e-4 \
  --logger wandb \
  --wandb-project rgb-self-recovery-backbones \
  --run-name dino-matching-unet \
  --device cuda
```

# Model 5 with trainable DIno:
```bash
export DATA="$PWD/gigaPose_datasets/results/rgb_self_recovery_assetto_window_data"
export MODELS="$PWD/gigaPose_datasets/results/rgb_self_recovery_backbone_comparison"

python3 -m tracking.rgb_self_recovery.sliding_window.dino_matching_unet.train \
  --dino-mode last_block \
  --dino-model dinov2_vits14 \
  --dino-input-size 224 \
  --initialize-from "$MODELS/dino_matching_unet/best_pose.ckpt" \
  --dino-learning-rate 1e-5 \
  --learning-rate 5e-5 \
  --data "$DATA/train" \
  --validation-data "$DATA/validation" \
  --output-dir "$MODELS/dino_matching_unet_last_block" \
  --epochs 200 \
  --batch-size 16 \
  --anti-flip-training \
  --heatmap-weight 0.10 \
  --heatmap-sigma-px 4 \
  --patience 10 \
  --min-delta 1e-4 \
  --logger wandb \
  --wandb-project rgb-self-recovery-backbones \
  --run-name dino-matching-unet-last-block \
  --device cuda
```
This now:
Loads every weight from the best frozen-DINO matching U-Net.
Preserves the trained RGB, CAD, U-Net, heatmap, and pose heads.
Unfreezes only DINO’s final transformer block and final normalization layer.
Fine-tunes DINO at 1e-5 and the remaining trainable network at 5e-5.
Saves results separately under dino_matching_unet_last_block.


Use this checkpoint first:
```bash
"$MODELS/dino_matching_unet/best_pose.ckpt"
```

# Run sliding-window inference
Use the exact same evaluation sequence and settings used for the other four models:

```bash
export EVAL_RUNS="$PWD/gigaPose_datasets/results/rgb_window_eval"

python3 -m tracking.rgb_self_recovery.sliding_window.dino_matching_unet.run \
  --predictions "$GIGAPOSE_PREDICTIONS" \
  --dataset-dir "$EVAL_DATA" \
  --split test \
  --checkpoint "$MODELS/dino_matching_unet/best_pose.ckpt" \
  --mesh "$EVAL_DATA/models/obj_000001.ply" \
  --output-dir "$EVAL_RUNS/dino_matching_unet" \
  --window-size 5 \
  --save-overlays \
  --device cuda

```

The result will be:
```bash
$EVAL_RUNS/dino_matching_unet/tracked_predictions.csv
```
You can then add it to the existing comparison evaluator:
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.evaluate_backbones \
  --dataset-dir "$EVAL_DATA" \
  --split test \
  --model cnn="$EVAL_RUNS/cnn" \
  --model dino_frozen="$EVAL_RUNS/dino_frozen" \
  --model dino_last_block="$EVAL_RUNS/dino_last_block" \
  --model matching_unet="$EVAL_RUNS/matching_unet" \
  --model dino_matching_unet="$EVAL_RUNS/dino_matching_unet" \
  --baseline cnn \
  --mesh "$EVAL_DATA/models/obj_000001.ply" \
  --distance-bins-m 0,20,40,60,80,100,120 \
  --max-overlays 120 \
  --output-dir "$EVAL_RUNS/all_five_backbones" \
  --overwrite

```

Gigapose_pred:
/mnt/ssd2tb/gigapose/gigaPose_datasets/results/gigapose_dev_rgb_recovery_val_laguna_front/predictions/large-pbrreal-rgb-mmodel_rgb_recovery_val_laguna_front-test_gigapose_dev_rgb_recovery_val_laguna_frontMultiHypothesis.csv 

Set the shared paths:
```bash
export MODELS="$PWD/gigaPose_datasets/results/rgb_self_recovery_backbone_comparison"
export DEV="$PWD/gigaPose_datasets/results/rgb_self_recovery_assetto_dev_eval"

```
Putnam rear:
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.assetto_validation_eval.run_suite \
  --dataset-dir gigaPose_datasets/datasets/rgb_recovery_val_putnam_rear \
  --predictions gigaPose_datasets/results/gigapose_dev_rgb_recovery_val_putnam_rear \
  --models-root "$MODELS" \
  --output-root "$DEV/putnam_rear/runs_rot_gated \
  --window-size 5 \
  --device cuda \
  --save-overlays \
  --overwrite \
  --orientation-gates \
  --skip-missing 
```

python3 -m tracking.rgb_self_recovery.sliding_window.assetto_validation_eval.run_suite \
  --dataset-dir gigaPose_datasets/datasets/rgb_recovery_val_putnam_rear \
  --predictions gigaPose_datasets/results/gigapose_dev_rgb_recovery_val_putnam_rear \
  --models-root "$MODELS" \
  --output-root "$DEV/putnam_rear/runs_rot_gated" \
  --models cnn_multicheckpoint,dino_frozen_multicheckpoint,dino_last_block,matching_unet,dino_matching_unet,dino_matching_unet_last_block \
  --window-size 5 \
  --device cuda \
  --overwrite \
  --save-overlays \
  --orientation-gates \
  --skip-missing



python3 -m tracking.rgb_self_recovery.sliding_window.assetto_validation_eval.run_suite \
  --dataset-dir gigaPose_datasets/datasets/rgb_recovery_val_putnam_rear \
  --predictions gigaPose_datasets/results/gigapose_dev_rgb_recovery_val_putnam_rear \
  --models-root "$MODELS" \
  --output-root "$DEV/putnam_rear/runs_visual_verified_ungated" \
  --window-size 5 \
  --device cuda \
  --no-orientation-gates \
  --save-overlays \
  --skip-missing



export MODELS="$PWD/gigaPose_datasets/results/rgb_self_recovery_backbone_comparison"
export DEV="$PWD/gigaPose_datasets/results/rgb_self_recovery_assetto_dev_eval"

python3 -m tracking.rgb_self_recovery.sliding_window.assetto_validation_eval.run_suite \
  --dataset-dir gigaPose_datasets/datasets/rgb_recovery_val_putnam_rear \
  --predictions gigaPose_datasets/results/gigapose_dev_rgb_recovery_val_putnam_rear \
  --models-root "$MODELS" \
  --output-root "$DEV/putnam_rear/runs_soft_gated" \
  --orientation-gate-mode soft \
  --state-iou-confidence-weight 0.5 \
  --device cuda \
  --save-overlays



Laguna front:
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.assetto_validation_eval.run_suite \
  --dataset-dir gigaPose_datasets/datasets/rgb_recovery_val_laguna_front \
  --predictions gigaPose_datasets/results/gigapose_dev_rgb_recovery_val_laguna_front \
  --models-root "$MODELS" \
  --output-root "$DEV/laguna_front/runs" \
  --window-size 5 \
  --device cuda \
  --skip-missing
```

The suite currently finds and runs:
CNN baseline
Frozen DINO + CNN/CAD
Last-block DINO + CNN/CAD
Matching U-Net
Frozen-DINO matching U-Net


After both run_suite commands finish, evaluate Putnam:
```bash

python3 -m tracking.rgb_self_recovery.sliding_window.assetto_validation_eval.evaluate_suite \
  --dataset-dir gigaPose_datasets/datasets/rgb_recovery_val_putnam_rear \
  --gigapose-predictions gigaPose_datasets/results/gigapose_dev_rgb_recovery_val_putnam_rear \
  --runs-root "$DEV/putnam_rear/runs_visual_verified_ungated" \
  --output-dir "$DEV/putnam_rear/comparison_visual_verified_ungated" \
  --max-overlays 120 \
  --skip-missing
```

Then evaluate Laguna:
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.assetto_validation_eval.evaluate_suite \
  --dataset-dir gigaPose_datasets/datasets/rgb_recovery_val_laguna_front \
  --gigapose-predictions gigaPose_datasets/results/gigapose_dev_rgb_recovery_val_laguna_front \
  --runs-root "$DEV/laguna_front/runs" \
  --output-dir "$DEV/laguna_front/comparison" \
  --max-overlays 120 \
  --skip-missing
```

evaluate_suite compares every recovery output against:
Ground-truth pose.
Raw GigaPose as the baseline.
The other recovery models.



1. Set paths
```bash
cd /home/anahita/gigapose

export MODELS="$PWD/gigaPose_datasets/results/rgb_self_recovery_backbone_comparison"
export DEV="$PWD/gigaPose_datasets/results/rgb_self_recovery_assetto_dev_eval"
export ALL_MODELS="cnn_multicheckpoint,dino_frozen_multicheckpoint,dino_last_block,matching_unet,dino_matching_unet,dino_matching_unet_last_block"
```

2. Run Putnam with abstention/fallback
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.assetto_validation_eval.run_suite \
  --dataset-dir "$PWD/gigaPose_datasets/datasets/rgb_recovery_val_putnam_rear" \
  --predictions "$PWD/gigaPose_datasets/results/gigapose_dev_rgb_recovery_val_putnam_rear" \
  --models-root "$MODELS" \
  --output-root "$DEV/putnam_rear/runs_soft_abstention" \
  --models "$ALL_MODELS" \
  --window-size 5 \
  --orientation-gate-mode soft \
  --state-iou-confidence-weight 0.5 \
  --device cuda \
  --save-overlays \
  --skip-missing \
  --overwrite
```
This runs the six models sequentially, so it may take a while.

3. Evaluate Putnam and generate comparison overlays
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.assetto_validation_eval.evaluate_suite \
  --dataset-dir "$PWD/gigaPose_datasets/datasets/rgb_recovery_val_putnam_rear" \
  --gigapose-predictions "$PWD/gigaPose_datasets/results/gigapose_dev_rgb_recovery_val_putnam_rear" \
  --runs-root "$DEV/putnam_rear/runs_soft_abstention" \
  --output-dir "$DEV/putnam_rear/comparison_soft_abstention" \
  --max-overlays 164 \
  --overwrite
```

Results will be in:
```bash
$DEV/putnam_rear/comparison_soft_abstention/summary.json
$DEV/putnam_rear/comparison_soft_abstention/overlays/
```

4. Run Laguna as the second evaluation condition
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.assetto_validation_eval.run_suite \
  --dataset-dir "$PWD/gigaPose_datasets/datasets/rgb_recovery_val_laguna_front" \
  --predictions "$PWD/gigaPose_datasets/results/gigapose_dev_rgb_recovery_val_laguna_front" \
  --models-root "$MODELS" \
  --output-root "$DEV/laguna_front/runs_soft_abstention" \
  --models "$ALL_MODELS" \
  --window-size 5 \
  --orientation-gate-mode soft \
  --state-iou-confidence-weight 0.5 \
  --device cuda \
  --save-overlays \
  --skip-missing \
  --overwrite
```

Then evaluate it:
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.assetto_validation_eval.evaluate_suite \
  --dataset-dir "$PWD/gigaPose_datasets/datasets/rgb_recovery_val_laguna_front" \
  --gigapose-predictions "$PWD/gigaPose_datasets/results/gigapose_dev_rgb_recovery_val_laguna_front" \
  --runs-root "$DEV/laguna_front/runs_soft_abstention" \
  --output-dir "$DEV/laguna_front/comparison_soft_abstention" \
  --max-overlays 120 \
  --overwrite
```


# #############################################
# #############################################
## Fair three-way experiment
Start with CNN only so the comparison finishes reasonably quickly.
```bash
cd /home/anahita/gigapose

export MODELS="$PWD/gigaPose_datasets/results/rgb_self_recovery_backbone_comparison"
export DEV="$PWD/gigaPose_datasets/results/rgb_self_recovery_assetto_dev_eval"
export DATASET="$PWD/gigaPose_datasets/datasets/rgb_recovery_val_putnam_rear"
export GP_ROOT="$PWD/gigaPose_datasets/results/gigapose_dev_rgb_recovery_val_putnam_rear"

```

1. Completely independent per-frame recovery
No previous pose, flow, beam, or window:
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.assetto_validation_eval.run_suite \
  --dataset-dir "$DATASET" \
  --predictions "$GP_ROOT" \
  --models-root "$MODELS" \
  --output-root "$DEV/putnam_rear/runs_ablation_per_frame" \
  --models cnn_multicheckpoint \
  --window-size 1 \
  --per-frame \
  --sequence-aware \
  --sequence-max-frame-gap 1 \
  --sequence-max-time-gap-s 0.5 \
  --orientation-gate-mode soft \
  --state-iou-confidence-weight 0.5 \
  --device cuda \
  --save-overlays \
  --overwrite

```

2. Original sequential tracker without the five-frame optimizer
This retains constant velocity, optical flow, previous pose, and candidate beam:
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.assetto_validation_eval.run_suite \
  --dataset-dir "$DATASET" \
  --predictions "$GP_ROOT" \
  --models-root "$MODELS" \
  --output-root "$DEV/putnam_rear/runs_ablation_sequential" \
  --models cnn_multicheckpoint \
  --window-size 1 \
  --sequence-aware \
  --sequence-max-frame-gap 1 \
  --sequence-max-time-gap-s 0.5 \
  --orientation-gate-mode soft \
  --state-iou-confidence-weight 0.5 \
  --device cuda \
  --save-overlays \
  --overwrite
```

3. Sequence-aware five-frame optimization
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.assetto_validation_eval.run_suite \
  --dataset-dir "$DATASET" \
  --predictions "$GP_ROOT" \
  --models-root "$MODELS" \
  --output-root "$DEV/putnam_rear/runs_ablation_window5" \
  --models cnn_multicheckpoint \
  --window-size 5 \
  --sequence-aware \
  --sequence-max-frame-gap 1 \
  --sequence-max-time-gap-s 0.5 \
  --orientation-gate-mode soft \
  --state-iou-confidence-weight 0.5 \
  --device cuda \
  --save-overlays \
  --overwrite
```

Compare all three directly
```bash
export GP_CSV="$GP_ROOT/predictions/large-pbrreal-rgb-mmodel_rgb_recovery_val_putnam_rear-test_gigapose_dev_rgb_recovery_val_putnam_rearMultiHypothesis.csv"

python3 -m tracking.rgb_self_recovery.sliding_window.evaluate_backbones \
  --dataset-dir "$DATASET" \
  --split test \
  --model gigapose="$GP_CSV" \
  --model per_frame="$DEV/putnam_rear/runs_ablation_per_frame/cnn_multicheckpoint" \
  --model sequential="$DEV/putnam_rear/runs_ablation_sequential/cnn_multicheckpoint" \
  --model window5="$DEV/putnam_rear/runs_ablation_window5/cnn_multicheckpoint" \
  --baseline gigapose \
  --mesh "$DATASET/models/obj_000001.ply" \
  --distance-bins-m 0,20,40,60,80,100,120 \
  --max-overlays 164 \
  --output-dir "$DEV/putnam_rear/comparison_temporal_ablation" \
  --overwrite

```


# #############################################
# #############################################
# Run GRU: 
Use this checkpoint first:

```bash
export CANDIDATE_CKPT="$PWD/gigaPose_datasets/results/rgb_self_recovery_backbone_comparison/dino_matching_unet/best_pose.ckpt"

```


Run GigaPose now
```bash
export ASSETTO_SOURCE="/mnt/ssd2tb/.local_share_backup/Steam/steamapps/common/assettocorsa/apps/lua/multi_cam_obs/frames/Assettocorsa_new_dataset_distance_bin_cleaned"

python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.run_gigapose \
  --dataset-name rgb_recovery_gru_train_assetto \
  --devices 0 \
  --batch-size 16 \
  --num-workers 2

```

Combined front/rear validation:
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.run_gigapose \
  --dataset-name rgb_recovery_gru_validation_laguna_putnam \
  --devices 0 \
  --batch-size 16 \
  --num-workers 2
```

Use these variables afterward:
```bash

export TRAIN_DATASET="$PWD/gigaPose_datasets/datasets/rgb_recovery_gru_train_assetto"
export VAL_DATASET="$PWD/gigaPose_datasets/datasets/rgb_recovery_gru_validation_laguna_putnam"

export TRAIN_GP="$PWD/gigaPose_datasets/results/gigapose_rgb_recovery_gru_train_assetto"
export VAL_GP="$PWD/gigaPose_datasets/results/gigapose_rgb_recovery_gru_validation_laguna_putnam"

export CANDIDATE_CKPT="$PWD/gigaPose_datasets/results/rgb_self_recovery_backbone_comparison/dino_matching_unet/best_pose.ckpt"

export GRU_ROOT="$PWD/gigaPose_datasets/results/rgb_self_recovery_gru_dino_matching"
mkdir -p "$GRU_ROOT"

```

Verify everything exists:
```bash
test -d "$TRAIN_DATASET" && echo "Training dataset found"
test -d "$VAL_DATASET" && echo "Validation dataset found"
test -d "$TRAIN_GP" && echo "Training GigaPose predictions found"
test -d "$VAL_GP" && echo "Validation GigaPose predictions found"
test -f "$CANDIDATE_CKPT" && echo "Recovery checkpoint found"
```

2. Optional smoke test
Candidate generation is the expensive step. Test 20 frames first:
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.export_candidates \
  --dataset-dir "$TRAIN_DATASET" \
  --split test \
  --predictions "$TRAIN_GP" \
  --checkpoint "$CANDIDATE_CKPT" \
  --mesh "$TRAIN_DATASET/models/obj_000001.ply" \
  --prediction-translation-unit mm \
  --saved-candidates 16 \
  --top-k-gigapose 5 \
  --max-candidates 48 \
  --refinement-iterations 2 \
  --allow-flip-hypotheses \
  --sequence-max-frame-gap 1 \
  --sequence-max-time-gap-s 0.5 \
  --max-frames 20 \
  --output-dir "$GRU_ROOT/smoke_candidates" \
  --device cuda \
  --overwrite

```

Inspect it:
```bash

python3 -m json.tool "$GRU_ROOT/smoke_candidates/manifest.json"
```
3. Export full training candidates
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.export_candidates \
  --dataset-dir "$TRAIN_DATASET" \
  --split test \
  --predictions "$TRAIN_GP" \
  --checkpoint "$CANDIDATE_CKPT" \
  --mesh "$TRAIN_DATASET/models/obj_000001.ply" \
  --prediction-translation-unit mm \
  --saved-candidates 16 \
  --top-k-gigapose 5 \
  --max-candidates 48 \
  --refinement-iterations 2 \
  --allow-flip-hypotheses \
  --sequence-max-frame-gap 1 \
  --sequence-max-time-gap-s 0.5 \
  --output-dir "$GRU_ROOT/data/train" \
  --device cuda \
  --overwrite

```

4. Export validation candidates
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.export_candidates \
  --dataset-dir "$VAL_DATASET" \
  --split test \
  --predictions "$VAL_GP" \
  --checkpoint "$CANDIDATE_CKPT" \
  --mesh "$VAL_DATASET/models/obj_000001.ply" \
  --prediction-translation-unit mm \
  --saved-candidates 16 \
  --top-k-gigapose 5 \
  --max-candidates 48 \
  --refinement-iterations 2 \
  --allow-flip-hypotheses \
  --sequence-max-frame-gap 1 \
  --sequence-max-time-gap-s 0.5 \
  --output-dir "$GRU_ROOT/data/validation" \
  --device cuda \
  --overwrite
```

5. Inspect the exported data
```bash
python3 -m json.tool "$GRU_ROOT/data/train/manifest.json" | \
  grep -E 'frame_count|segment_count|oracle_nonbaseline_fraction|skipped_count'

python3 -m json.tool "$GRU_ROOT/data/validation/manifest.json" | \
  grep -E 'frame_count|segment_count|oracle_nonbaseline_fraction|skipped_count'

```

6. Train the GRU
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.train \
  --data "$GRU_ROOT/data/train" \
  --validation-data "$GRU_ROOT/data/validation" \
  --output-dir "$GRU_ROOT/models/dino_matching_gru" \
  --clip-length 8 \
  --clip-stride 1 \
  --validation-stride 8 \
  --candidate-hidden 96 \
  --gru-hidden 128 \
  --gru-layers 1 \
  --dropout 0.1 \
  --epochs 100 \
  --batch-size 32 \
  --num-workers 4 \
  --learning-rate 2e-4 \
  --weight-decay 1e-4 \
  --expected-cost-weight 0.1 \
  --fallback-label-weight 1.0 \
  --gradient-clip 5 \
  --patience 10 \
  --min-delta 1e-4 \
  --device cuda \
  --overwrite


  python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.train \
  --data "$GRU_ROOT/data/train" \
  --validation-data "$GRU_ROOT/data/validation" \
  --output-dir "$GRU_ROOT/models/dino_matching_gru_2layer" \
  --clip-length 8 \
  --clip-stride 1 \
  --validation-stride 8 \
  --candidate-hidden 96 \
  --gru-hidden 128 \
  --gru-layers 2 \
  --dropout 0.1 \
  --epochs 100 \
  --batch-size 32 \
  --num-workers 4 \
  --learning-rate 2e-4 \
  --weight-decay 1e-4 \
  --expected-cost-weight 0.1 \
  --fallback-label-weight 1.0 \
  --gradient-clip 5 \
  --patience 10 \
  --min-delta 1e-4 \
  --device cuda \
  --overwrite
```

7. Plot training progress
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.plot_history \
  --run-dir "$GRU_ROOT/models/dino_matching_gru"

  python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.plot_history \
  --run-dir "$GRU_ROOT/models/dino_matching_gru_2layer"
```

8. Run raw GRU inference
First evaluate without abstention thresholds so we can see what the GRU actually learned:
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.select \
  --data "$GRU_ROOT/data/validation" \
  --checkpoint "$GRU_ROOT/models/dino_matching_gru/best.ckpt" \
  --minimum-recovery-probability 0.0 \
  --minimum-recovery-margin-over-gigapose 0.0 \
  --output-dir "$GRU_ROOT/runs/putnam_laguna_raw_gru" \
  --device cuda \
  --overwrite


  python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.select \
  --data "$GRU_ROOT/data/validation" \
  --checkpoint "$GRU_ROOT/models/dino_matching_gru_2layer/best.ckpt" \
  --minimum-recovery-probability 0.0 \
  --minimum-recovery-margin-over-gigapose 0.0 \
  --output-dir "$GRU_ROOT/runs/putnam_laguna_raw_gru_2layer" \
  --device cuda \
  --overwrite

#   Inspect the result:
    python3 -m json.tool "$GRU_ROOT/runs/putnam_raw_gru/run_report.json"

    python3 -m json.tool "$GRU_ROOT/runs/putnam_laguna_raw_gru_2layer/run_report.json"
```

9. Run conservative inference with GigaPose fallback
After inspecting the raw result, run an abstaining version:
```bash

python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.select \
  --data "$GRU_ROOT/data/validation" \
  --checkpoint "$GRU_ROOT/models/dino_matching_gru/best.ckpt" \
  --minimum-recovery-probability 0.15 \
  --minimum-recovery-margin-over-gigapose 0.03 \
  --output-dir "$GRU_ROOT/runs/putnam_laguna_guarded_gru" \
  --device cuda \
  --overwrite


  python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.select \
  --data "$GRU_ROOT/data/validation" \
  --checkpoint "$GRU_ROOT/models/dino_matching_gru_2layer/best.ckpt" \
  --minimum-recovery-probability 0.15 \
  --minimum-recovery-margin-over-gigapose 0.03 \
  --output-dir "$GRU_ROOT/runs/putnam_laguna_guarded_gru_2layer" \
  --device cuda \
  --overwrite
```
The GRU retains original GigaPose whenever its preferred recovery candidate does not pass those thresholds.

10. Compare raw and guarded GRU against GigaPose
```bash
export VAL_GP_CSV="$VAL_GP/predictions/large-pbrreal-rgb-mmodel_rgb_recovery_gru_validation_laguna_putnam-test_gigapose_rgb_recovery_gru_validation_laguna_putnamMultiHypothesis.csv"

python3 -m tracking.rgb_self_recovery.sliding_window.evaluate_backbones \
  --dataset-dir "$VAL_DATASET" \
  --split test \
  --model gigapose="$VAL_GP_CSV" \
  --model gru_raw="$GRU_ROOT/runs/putnam_laguna_raw_gru" \
  --model gru_guarded="$GRU_ROOT/runs/putnam_laguna_guarded_gru" \
  --baseline gigapose \
  --prediction-translation-unit mm \
  --distance-bins-m 0,20,40,60,80,100,120 \
  --output-dir "$GRU_ROOT/comparison/putnam_laguna" \
  --overwrite



  python3 -m tracking.rgb_self_recovery.sliding_window.evaluate_backbones \
  --dataset-dir "$VAL_DATASET" \
  --split test \
  --model gigapose="$VAL_GP_CSV" \
  --model gru_raw="$GRU_ROOT/runs/putnam_laguna_raw_gru_2layer" \
  --model gru_guarded="$GRU_ROOT/runs/putnam_laguna_guarded_gru_2layer" \
  --baseline gigapose \
  --prediction-translation-unit mm \
  --distance-bins-m 0,20,40,60,80,100,120 \
  --output-dir "$GRU_ROOT/comparison/putnam_laguna_2layer" \
  --overwrite

#   Read the numerical comparison:
python3 -m json.tool "$GRU_ROOT/comparison/putnam_laguna/summary.json"

# The overlays will be under:
$GRU_ROOT/comparison/putnam/overlays/
```

11. Later: run on a genuinely held-out dataset
After preparing a third run and running GigaPose on it:
```bash

export TEST_DATASET=/absolute/path/to/held_out_dataset
export TEST_GP=/absolute/path/to/held_out_gigapose_results
```
Export its candidates:
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.export_candidates \
  --dataset-dir "$TEST_DATASET" \
  --split test \
  --predictions "$TEST_GP" \
  --checkpoint "$CANDIDATE_CKPT" \
  --mesh "$TEST_DATASET/models/obj_000001.ply" \
  --prediction-translation-unit mm \
  --saved-candidates 16 \
  --top-k-gigapose 5 \
  --max-candidates 48 \
  --refinement-iterations 2 \
  --allow-flip-hypotheses \
  --sequence-max-frame-gap 1 \
  --sequence-max-time-gap-s 0.5 \
  --output-dir "$GRU_ROOT/data/held_out_test" \
  --device cuda \
  --overwrite
```

Run inference:
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.select \
  --data "$GRU_ROOT/data/held_out_test" \
  --checkpoint "$GRU_ROOT/models/dino_matching_gru/best.ckpt" \
  --minimum-recovery-probability 0.15 \
  --minimum-recovery-margin-over-gigapose 0.03 \
  --output-dir "$GRU_ROOT/runs/held_out_test" \
  --device cuda \
  --overwrite
```


After the one-layer GRU selects a visual candidate, the orientation layer:
Maintains previous accepted orientations.
Estimates constant angular velocity from up to five accepted frames.
Generates these temporal orientation hypotheses:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.select \
  --data "$GRU_ROOT/data/validation" \
  --checkpoint "$GRU_ROOT/models/dino_matching_gru/best.ckpt" \
  --temporal-orientation \
  --orientation-soft-start-deg 45 \
  --orientation-hard-limit-deg 90 \
  --flip-min-angle-deg 135 \
  --flip-confirmation-frames 2 \
  --flip-min-probability 0.15 \
  --flip-min-margin 0.03 \
  --maximum-angular-prediction-deg 30 \
  --minimum-recovery-probability 0 \
  --minimum-recovery-margin-over-gigapose 0 \
  --output-dir "$GRU_ROOT/runs/putnam_laguna_temporal_orientation_gru" \
  --device cuda \
  --overwrite



python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.select \
  --data "$GRU_ROOT/data/validation" \
  --checkpoint "$GRU_ROOT/models/dino_matching_gru/best.ckpt" \
  --temporal-orientation \
  --flip-confirmation-frames 2 \
  --stable-orientation-frames 4 \
  --stable-flip-confirmation-frames 5 \
  --orientation-soft-start-deg 45 \
  --orientation-hard-limit-deg 90 \
  --flip-min-angle-deg 135 \
  --flip-min-probability 0.15 \
  --flip-min-margin 0.03 \
  --maximum-angular-prediction-deg 30 \
  --minimum-recovery-probability 0 \
  --minimum-recovery-margin-over-gigapose 0 \
  --output-dir "$GRU_ROOT/runs/putnam_laguna_adaptive_temporal_orientation_gru" \
  --device cuda \
  --overwrite

export GRU_ROOT="$PWD/gigaPose_datasets/results/rgb_self_recovery_gru_dino_matching"

python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.select \
  --data "$GRU_ROOT/data/validation" \
  --checkpoint "$GRU_ROOT/models/dino_matching_gru/best.ckpt" \
  --temporal-orientation \
  --orientation-fixed-lag \
  --orientation-soft-start-deg 45 \
  --orientation-hard-limit-deg 90 \
  --flip-min-angle-deg 135 \
  --flip-confirmation-frames 2 \
  --stable-orientation-frames 4 \
  --stable-flip-confirmation-frames 5 \
  --flip-min-probability 0.15 \
  --flip-min-margin 0.03 \
  --maximum-angular-prediction-deg 30 \
  --minimum-recovery-probability 0 \
  --minimum-recovery-margin-over-gigapose 0 \
  --output-dir "$GRU_ROOT/runs/putnam_laguna_fixed_lag_gru" \
  --device cuda \
  --overwrite

```
Compare against the original selector
```bash

export VAL_GP_CSV="$VAL_GP/predictions/large-pbrreal-rgb-mmodel_rgb_recovery_gru_validation_laguna_putnam-test_gigapose_rgb_recovery_gru_validation_laguna_putnamMultiHypothesis.csv"

python3 -m tracking.rgb_self_recovery.sliding_window.evaluate_backbones \
  --dataset-dir "$VAL_DATASET" \
  --split test \
  --model gigapose="$VAL_GP_CSV" \
  --model gru_original="$GRU_ROOT/runs/putnam_laguna_raw_gru" \
  --model gru_temporal="$GRU_ROOT/runs/putnam_laguna_temporal_orientation_gru" \
  --baseline gigapose \
  --prediction-translation-unit mm \
  --mesh "$VAL_DATASET/models/obj_000001.ply" \
  --distance-bins-m 0,20,40,60,80,100,120 \
  --max-overlays 328 \
  --output-dir "$GRU_ROOT/comparison/putnam_laguna_temporal_orientation" \
  --overwrite
  ```

# #############################################
# #############################################
## GRU Kalman
```bash

export KF_ROOT="$PWD/gigaPose_datasets/results/rgb_self_recovery_gru_kalman"

export TRAIN_DATASET="$PWD/gigaPose_datasets/datasets/rgb_recovery_gru_train_assetto"
export VAL_DATASET="$PWD/gigaPose_datasets/datasets/rgb_recovery_gru_validation_laguna_putnam"

export TRAIN_GP="$PWD/gigaPose_datasets/results/gigapose_rgb_recovery_gru_train_assetto"
export VAL_GP="$PWD/gigaPose_datasets/results/gigapose_rgb_recovery_gru_validation_laguna_putnam"
```

Export the training sequences:
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.export_measurements \
  --dataset-dir "$TRAIN_DATASET" \
  --predictions "$TRAIN_GP" \
  --output-dir "$KF_ROOT/data/train" \
  --overwrite
```

Export validation sequences:
```bash 
python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.export_measurements \
  --dataset-dir "$VAL_DATASET" \
  --predictions "$VAL_GP" \
  --output-dir "$KF_ROOT/data/validation" \
  --overwrite
```
Train:
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.train \
  --data "$KF_ROOT/data/train" \
  --validation-data "$KF_ROOT/data/validation" \
  --output-dir "$KF_ROOT/model" \
  --clip-length 16 \
  --clip-stride 4 \
  --validation-stride 16 \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 2e-4 \
  --patience 10 \
  --min-delta 1e-4 \
  --device cuda \
  --overwrite
```

The trained weights are saved in:
```bash
$KF_ROOT/model/best.ckpt
```

Plot training:
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.plot_history \
  --run-dir "$KF_ROOT/model"
```

First run unconstrained inference to measure the model itself:
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.filter \
  --data "$KF_ROOT/data/validation" \
  --checkpoint "$KF_ROOT/model/best.ckpt" \
  --max-correction-translation-m 100 \
  --max-correction-rotation-deg 180 \
  --output-dir "$KF_ROOT/validation_raw" \
  --device cuda \
  --overwrite
```

Then run guarded deployment-style inference:
```bash


python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.filter \
  --data "$KF_ROOT/data/validation" \
  --checkpoint "$KF_ROOT/model/best.ckpt" \
  --max-correction-translation-m 5 \
  --max-correction-rotation-deg 60 \
  --output-dir "$KF_ROOT/validation_guarded" \
  --device cuda \
  --overwrite

```

 Compare raw and guarded GRU against GigaPose
```bash
export VAL_GP_CSV="$VAL_GP/predictions/large-pbrreal-rgb-mmodel_rgb_recovery_gru_validation_laguna_putnam-test_gigapose_rgb_recovery_gru_validation_laguna_putnamMultiHypothesis.csv"

python3 -m tracking.rgb_self_recovery.sliding_window.evaluate_backbones \
  --dataset-dir "$VAL_DATASET" \
  --split test \
  --model gigapose="$VAL_GP_CSV" \
  --model gru_kf_raw="$KF_ROOT/validation_raw" \
  --model gru_kf_guarded="$KF_ROOT/validation_guarded" \
  --baseline gigapose \
  --prediction-translation-unit mm \
  --distance-bins-m 0,20,40,60,80,100,120 \
  --output-dir "$KF_ROOT/comparison/putnam_laguna" \
  --overwrite
```


The important outputs are:
tracked_predictions.csv: filtered BOP-format poses.
filter_diagnostics.csv: frame-level raw/filtered errors, corrections, gains, and fallback decisions.
run_report.json: aggregate results separated by camera.
best.ckpt: best validation checkpoint.
history.png: training curves.
This model is different from the earlier GRU selector: the selector chooses among recovery candidates, while this new model directly filters raw GigaPose SE(3) measurements over time. It is implemented and tested, but whether it improves GigaPose must be determined from the validation report after training.




# GRU selector + GRU-Kalman
1. Set paths
```bash
cd /home/anahita/gigapose
conda activate gigapose

export SELECTOR_ROOT="$PWD/gigaPose_datasets/results/rgb_self_recovery_gru_dino_matching"
export CASCADE_ROOT="$PWD/gigaPose_datasets/results/rgb_self_recovery_fixed_lag_kalman"

export TRAIN_DATASET="$PWD/gigaPose_datasets/datasets/rgb_recovery_gru_train_assetto"
export VAL_DATASET="$PWD/gigaPose_datasets/datasets/rgb_recovery_gru_validation_laguna_putnam"
```

2. Run the adaptive selector on training candidates
Run fixed-lag selector on training candidates:
This creates the measurements used to train the second stage:
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.select \
  --data "$SELECTOR_ROOT/data/train" \
  --checkpoint "$SELECTOR_ROOT/models/dino_matching_gru/best.ckpt" \
  --temporal-orientation \
  --flip-confirmation-frames 2 \
  --stable-orientation-frames 4 \
  --stable-flip-confirmation-frames 5 \
  --orientation-soft-start-deg 45 \
  --orientation-hard-limit-deg 90 \
  --flip-min-angle-deg 135 \
  --flip-min-probability 0.15 \
  --flip-min-margin 0.03 \
  --maximum-angular-prediction-deg 30 \
  --minimum-recovery-probability 0 \
  --minimum-recovery-margin-over-gigapose 0 \
  --output-dir "$CASCADE_ROOT/selector/train" \
  --device cuda \
  --overwrite



  python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.select \
  --data "$SELECTOR_ROOT/data/train" \
  --checkpoint "$SELECTOR_ROOT/models/dino_matching_gru/best.ckpt" \
  --temporal-orientation \
  --orientation-fixed-lag \
  --flip-confirmation-frames 2 \
  --stable-orientation-frames 4 \
  --stable-flip-confirmation-frames 5 \
  --flip-min-probability 0.15 \
  --flip-min-margin 0.03 \
  --minimum-recovery-probability 0 \
  --minimum-recovery-margin-over-gigapose 0 \
  --output-dir "$CASCADE_ROOT/selector/train" \
  --device cuda \
  --overwrite
```

3. Run the same adaptive selector on validation
Run it on validation candidates:
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.select \
  --data "$SELECTOR_ROOT/data/validation" \
  --checkpoint "$SELECTOR_ROOT/models/dino_matching_gru/best.ckpt" \
  --temporal-orientation \
  --flip-confirmation-frames 2 \
  --stable-orientation-frames 4 \
  --stable-flip-confirmation-frames 5 \
  --orientation-soft-start-deg 45 \
  --orientation-hard-limit-deg 90 \
  --flip-min-angle-deg 135 \
  --flip-min-probability 0.15 \
  --flip-min-margin 0.03 \
  --maximum-angular-prediction-deg 30 \
  --minimum-recovery-probability 0 \
  --minimum-recovery-margin-over-gigapose 0 \
  --output-dir "$CASCADE_ROOT/selector/validation" \
  --device cuda \
  --overwrite

python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.select \
  --data "$SELECTOR_ROOT/data/validation" \
  --checkpoint "$SELECTOR_ROOT/models/dino_matching_gru/best.ckpt" \
  --temporal-orientation \
  --orientation-fixed-lag \
  --flip-confirmation-frames 2 \
  --stable-orientation-frames 4 \
  --stable-flip-confirmation-frames 5 \
  --flip-min-probability 0.15 \
  --flip-min-margin 0.03 \
  --minimum-recovery-probability 0 \
  --minimum-recovery-margin-over-gigapose 0 \
  --output-dir "$CASCADE_ROOT/selector/validation" \
  --device cuda \
  --overwrite

```

4. Export selector outputs as Kalman 
measurements
Pass the actual tracked_predictions.csv, not its parent directory:
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.export_measurements \
  --dataset-dir "$TRAIN_DATASET" \
  --predictions "$CASCADE_ROOT/selector/train/tracked_predictions.csv" \
  --prediction-translation-unit mm \
  --output-dir "$CASCADE_ROOT/data/train" \
  --overwrite

python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.export_measurements \
  --dataset-dir "$VAL_DATASET" \
  --predictions "$CASCADE_ROOT/selector/validation/tracked_predictions.csv" \
  --prediction-translation-unit mm \
  --output-dir "$CASCADE_ROOT/data/validation" \
  --overwrite
---------------------------------------------
python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.export_measurements \
  --dataset-dir "$TRAIN_DATASET" \
  --predictions "$CASCADE_ROOT/selector/train/tracked_predictions.csv" \
  --prediction-translation-unit mm \
  --output-dir "$CASCADE_ROOT/data/train" \
  --overwrite


python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.export_measurements \
  --dataset-dir "$VAL_DATASET" \
  --predictions "$CASCADE_ROOT/selector/validation/tracked_predictions.csv" \
  --prediction-translation-unit mm \
  --output-dir "$CASCADE_ROOT/data/validation" \
  --overwrite






```

Inspect the reports:
```bash
python3 -m json.tool "$CASCADE_ROOT/data/train/manifest.json"
python3 -m json.tool "$CASCADE_ROOT/data/validation/manifest.json"

# Expected validation values:
frame_count:   328
segment_count: 13
skipped_count: 0

```
Train translation-only Kalman:
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.train \
  --data "$CASCADE_ROOT/data/train" \
  --validation-data "$CASCADE_ROOT/data/validation" \
  --output-dir "$CASCADE_ROOT/model_translation_only" \
  --preserve-measurement-rotation \
  --measurement-preservation-weight 0.05 \
  --clip-length 16 \
  --clip-stride 4 \
  --validation-stride 16 \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 2e-4 \
  --patience 10 \
  --min-delta 1e-4 \
  --device cuda \
  --overwrite
  ```
Filter validation:
```bash

python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.filter \
  --data "$CASCADE_ROOT/data/validation" \
  --checkpoint "$CASCADE_ROOT/model_translation_only/best.ckpt" \
  --max-correction-translation-m 100 \
  --max-correction-rotation-deg 180 \
  --output-dir "$CASCADE_ROOT/validation_translation_only" \
  --device cuda \
  --overwrite
```

Run the full translation-and-rotation GRU–Kalman using the same fixed-lag selector measurements.
```bash
export SELECTOR_ROOT="$PWD/gigaPose_datasets/results/rgb_self_recovery_gru_dino_matching"
export CASCADE_ROOT="$PWD/gigaPose_datasets/results/rgb_self_recovery_fixed_lag_kalman"
export VAL_DATASET="$PWD/gigaPose_datasets/datasets/rgb_recovery_gru_validation_laguna_putnam"

# Set this to the actual validation GigaPose MultiHypothesis CSV:
export VAL_GP_CSV="/path/to/validation/MultiHypothesis.csv"
```

Train the full-pose model:
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.train \
  --data "$CASCADE_ROOT/data/train" \
  --validation-data "$CASCADE_ROOT/data/validation" \
  --output-dir "$CASCADE_ROOT/model_full_pose" \
  --no-preserve-measurement-rotation \
  --measurement-preservation-weight 0.05 \
  --translation-scale-m 1 \
  --rotation-scale-deg 20 \
  --clip-length 16 \
  --clip-stride 4 \
  --validation-stride 16 \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 2e-4 \
  --patience 10 \
  --min-delta 1e-4 \
  --device cuda \
  --overwrite
```

Run full-pose filtering with a 45° safety limit:
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.filter \
  --data "$CASCADE_ROOT/data/validation" \
  --checkpoint "$CASCADE_ROOT/model_full_pose/best.ckpt" \
  --max-correction-translation-m 100 \
  --max-correction-rotation-deg 45 \
  --output-dir "$CASCADE_ROOT/validation_full_pose" \
  --device cuda \
  --overwrite
```

Compare GigaPose, fixed-lag translation-only Kalman, and fixed-lag full-pose Kalman:
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.evaluate_backbones \
  --dataset-dir "$VAL_DATASET" \
  --split test \
  --model gigapose="$VAL_GP_CSV" \
  --model fixed_lag_selector="$CASCADE_ROOT/selector/validation" \
  --model kalman_translation="$CASCADE_ROOT/validation_translation_only" \
  --model kalman_full_pose="$CASCADE_ROOT/validation_full_pose" \
  --baseline gigapose \
  --prediction-translation-unit mm \
  --mesh "$VAL_DATASET/models/obj_000001.ply" \
  --distance-bins-m 0,20,40,60,80,100,120 \
  --max-overlays 328 \
  --output-dir "$CASCADE_ROOT/comparison/full_cascade_ablation" \
  --overwrite
```


train:
```bash

export CASCADE_ROOT="$PWD/gigaPose_datasets/results/rgb_self_recovery_adaptive_selector_kalman"

python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.train \
  --data "$CASCADE_ROOT/data/train" \
  --validation-data "$CASCADE_ROOT/data/validation" \
  --output-dir "$CASCADE_ROOT/model_full_pose" \
  --no-preserve-measurement-rotation \
  --measurement-preservation-weight 0.05 \
  --rotation-scale-deg 20 \
  --clip-length 16 \
  --clip-stride 4 \
  --validation-stride 16 \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 2e-4 \
  --patience 10 \
  --min-delta 1e-4 \
  --device cuda \
  --overwrite



  python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.train \
  --data "$CASCADE_ROOT/data/train" \
  --validation-data "$CASCADE_ROOT/data/validation" \
  --output-dir "$CASCADE_ROOT/model_full" \
  --no-preserve-measurement-rotation \
  --measurement-preservation-weight 0.05 \
  --clip-length 16 \
  --clip-stride 4 \
  --validation-stride 16 \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 2e-4 \
  --patience 10 \
  --min-delta 1e-4 \
  --device cuda \
  --overwrite

```
Then evaluate it with a conservative rotational correction limit:
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.filter \
  --data "$CASCADE_ROOT/data/validation" \
  --checkpoint "$CASCADE_ROOT/model_full_pose/best.ckpt" \
  --max-correction-translation-m 100 \
  --max-correction-rotation-deg 45 \
  --output-dir "$CASCADE_ROOT/validation_full_pose" \
  --device cuda \
  --overwrite
```

The \(45^\circ\) limit prevents the second stage from applying a large correction relative to the already flip-protected adaptive orientation.

Compare both variants
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.evaluate_backbones \
  --dataset-dir "$VAL_DATASET" \
  --split test \
  --model adaptive_selector="$CASCADE_ROOT/selector/validation" \
  --model kalman_translation_only="$CASCADE_ROOT/validation_raw" \
  --model kalman_full_pose="$CASCADE_ROOT/validation_full_pose" \
  --baseline adaptive_selector \
  --prediction-translation-unit mm \
  --mesh "$VAL_DATASET/models/obj_000001.ply" \
  --distance-bins-m 0,20,40,60,80,100,120 \
  --max-overlays 328 \
  --output-dir "$CASCADE_ROOT/comparison/translation_vs_full_pose" \
  --overwrite
```