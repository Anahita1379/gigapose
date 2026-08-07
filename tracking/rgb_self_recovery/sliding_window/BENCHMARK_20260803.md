# Assetto 2026-08-03 full tracking benchmark

This runbook evaluates unseen front and rear sequences from
`Assettocorsa_new_dataset_benchmark_distance_bin_cleaned` with the same trained
checkpoints used for the Laguna/Putnam validation experiments.

The 120--140 m bin is included, but is out of distribution because the models
were trained only through 120 m.

## 1. Paths

Run every command from the GigaPose repository root in the `gigapose` conda
environment.

```bash
cd /home/anahita/gigapose

export RAW_BENCH="/mnt/ssd2tb/.local_share_backup/Steam/steamapps/common/assettocorsa/apps/lua/multi_cam_obs/frames/Assettocorsa_new_dataset_benchmark_distance_bin_cleaned"
export BENCH_DATASET_NAME="rgb_recovery_benchmark_assetto_20260803"
export BENCH_DATASET="$PWD/gigaPose_datasets/datasets/$BENCH_DATASET_NAME"
export BENCH_ROOT="$PWD/gigaPose_datasets/results/rgb_self_recovery_benchmark_20260803"
export GP_RESULT="$PWD/gigaPose_datasets/results/gigapose_rgb_recovery_benchmark_assetto_20260803"
export MODELS="$PWD/gigaPose_datasets/results/rgb_self_recovery_backbone_comparison"
export SELECTOR_ROOT="$PWD/gigaPose_datasets/results/rgb_self_recovery_gru_dino_matching"
export RAW_KALMAN_ROOT="$PWD/gigaPose_datasets/results/rgb_self_recovery_gru_kalman"
export CASCADE_MODEL_ROOT="$PWD/gigaPose_datasets/results/rgb_self_recovery_fixed_lag_kalman"

mkdir -p "$BENCH_ROOT"
```

```bash 

conda activate gigapose
cd /home/anahita/gigapose

set -euo pipefail

export POSE_BENCH="$PWD/gigaPose_datasets/datasets/Assettocorsa_new_dataset_benchmark_distance_bin_pose_regenerated"
export POSE_DATASET_NAME="rgb_recovery_benchmark_assetto_20260803_pose_regenerated"
export POSE_DATASET="$PWD/gigaPose_datasets/datasets/$POSE_DATASET_NAME"

export GP_RUN_NAME="gigapose_rgb_recovery_benchmark_pose_regenerated_20260803"
export GP_RESULT="$PWD/gigaPose_datasets/results/$GP_RUN_NAME"

export BENCH_ROOT="$PWD/gigaPose_datasets/results/rgb_self_recovery_benchmark_pose_regenerated_20260803"
export MODELS="$PWD/gigaPose_datasets/results/rgb_self_recovery_backbone_comparison"
export SELECTOR_ROOT="$PWD/gigaPose_datasets/results/rgb_self_recovery_gru_dino_matching"
export RAW_KALMAN_ROOT="$PWD/gigaPose_datasets/results/rgb_self_recovery_gru_kalman"
export CASCADE_MODEL_ROOT="$PWD/gigaPose_datasets/results/rgb_self_recovery_fixed_lag_kalman"

mkdir -p "$BENCH_ROOT"

```





## 2. Prepare and validate the benchmark dataset

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.prepare_assetto_dataset \
  --assetto-export-root "$RAW_BENCH" \
  --role benchmark \
  --dataset-name "$BENCH_DATASET_NAME" \
  --distance-bins distance0_20,distance20_40,distance40_60,distance60_80,distance80_100,distance100_120,distance120_140 \
  --cameras front,rear \
  --maximum-depth-m 150 \
  --expected-cleaned-width 1548 \
  --overwrite

python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.validate_assetto_dataset \
  --dataset-dir "$BENCH_DATASET" \
  --split test \
  --output "$BENCH_DATASET/validation_report.json"

python3 -m json.tool "$BENCH_DATASET/evaluation_metadata.json"
python3 -m json.tool "$BENCH_DATASET/validation_report.json"
```

The preparer records rows without a valid joined instance ID under `skipped`
instead of assigning a potentially incorrect mask.



need to redo this step:
```bash
export POSE_BENCH="$PWD/gigaPose_datasets/datasets/Assettocorsa_new_dataset_benchmark_distance_bin_pose_regenerated"
export POSE_DATASET_NAME="rgb_recovery_benchmark_assetto_20260803_pose_regenerated"
export POSE_DATASET="$PWD/gigaPose_datasets/datasets/$POSE_DATASET_NAME"

python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.prepare_assetto_dataset \
  --assetto-export-root "$POSE_BENCH" \
  --role benchmark \
  --dataset-name "$POSE_DATASET_NAME" \
  --distance-bins distance0_20,distance20_40,distance40_60,distance60_80,distance80_100,distance100_120,distance120_140 \
  --cameras front,rear \
  --maximum-depth-m 150 \
  --expected-cleaned-width 1548 \
  --overwrite

python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.validate_assetto_dataset \
  --dataset-dir "$POSE_DATASET" \
  --split test \
  --output "$POSE_DATASET/validation_report.json"

python3 -m json.tool "$POSE_DATASET/validation_report.json"

```


## 3. Run GigaPose

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.run_gigapose \
  --dataset-name "$BENCH_DATASET_NAME" \
  --run-name gigapose_rgb_recovery_benchmark_assetto_20260803 \
  --batch-size 16 \
  --num-workers 2 \
  --devices 0

export GP_CSV="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["predictions"])' "$GP_RESULT/gru_gigapose_manifest.json")"
test -f "$GP_CSV" && echo "$GP_CSV"
```


```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.run_gigapose \
  --dataset-name "$POSE_DATASET_NAME" \
  --run-name gigapose_rgb_recovery_benchmark_pose_regenerated_20260803 \
  --batch-size 16 \
  --num-workers 2 \
  --devices 0

export GP_REGENERATED_RESULT="$PWD/gigaPose_datasets/results/gigapose_rgb_recovery_benchmark_pose_regenerated_20260803"

export GP_REGENERATED_CSV="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["predictions"])' "$GP_REGENERATED_RESULT/gru_gigapose_manifest.json")"
test -f "$GP_REGENERATED_CSV" && echo "$GP_REGENERATED_CSV"
```


## 4. Run standalone RGB self-recovery models

This runs the original CNN recovery and the DINO matching U-Net used to create
the GRU selector's visual candidates. Both use sequence-safe five-frame state
and soft orientation gating.

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.assetto_validation_eval.run_suite \
  --dataset-dir "$BENCH_DATASET" \
  --predictions "$GP_CSV" \
  --models-root "$MODELS" \
  --models cnn_multicheckpoint,dino_matching_unet \
  --output-root "$BENCH_ROOT/rgb_recovery" \
  --window-size 5 \
  --sequence-aware \
  --sequence-max-frame-gap 1 \
  --sequence-max-time-gap-s 0.5 \
  --orientation-gate-mode soft \
  --device cuda \
  --overwrite


python3 -m tracking.rgb_self_recovery.sliding_window.assetto_validation_eval.run_suite \
  --dataset-dir "$POSE_DATASET" \
  --predictions "$GP_CSV" \
  --models-root "$MODELS" \
  --models cnn_multicheckpoint,dino_matching_unet \
  --output-root "$BENCH_ROOT/rgb_recovery" \
  --window-size 5 \
  --sequence-aware \
  --sequence-max-frame-gap 1 \
  --sequence-max-time-gap-s 0.5 \
  --orientation-gate-mode soft \
  --device cuda \
  --overwrite


```

## 5. Export the exact RGB/CAD candidates expected by the trained GRU

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.export_candidates \
  --dataset-dir "$BENCH_DATASET" \
  --split test \
  --predictions "$GP_CSV" \
  --checkpoint "$MODELS/dino_matching_unet/best_pose.ckpt" \
  --mesh "$BENCH_DATASET/models/obj_000001.ply" \
  --output-dir "$BENCH_ROOT/gru_candidates" \
  --prediction-translation-unit mm \
  --saved-candidates 16 \
  --top-k-gigapose 5 \
  --max-candidates 48 \
  --refinement-iterations 2 \
  --allow-flip-hypotheses \
  --rotation-offsets-deg 20,45 \
  --yaw-offsets-deg 30,90 \
  --log-depth-offsets=-0.35,0.35 \
  --center-offsets-px=-48,48 \
  --broad-seeds 4 \
  --quality-weight 1.0 \
  --confidence-weight 0.5 \
  --silhouette-weight 1.5 \
  --measurement-weight 0.05 \
  --history-weight 0.0 \
  --translation-label-scale-m 1.0 \
  --rotation-label-scale-deg 20.0 \
  --sequence-max-frame-gap 1 \
  --sequence-max-time-gap-s 0.5 \
  --device cuda \
  --overwrite

# -----------------------new bench marck---------------
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.export_candidates \
  --dataset-dir "$POSE_DATASET" \
  --split test \
  --predictions "$GP_CSV" \
  --checkpoint "$MODELS/dino_matching_unet/best_pose.ckpt" \
  --mesh "$POSE_DATASET/models/obj_000001.ply" \
  --output-dir "$BENCH_ROOT/gru_candidates" \
  --prediction-translation-unit mm \
  --saved-candidates 16 \
  --top-k-gigapose 5 \
  --max-candidates 48 \
  --refinement-iterations 2 \
  --allow-flip-hypotheses \
  --rotation-offsets-deg 20,45 \
  --yaw-offsets-deg 30,90 \
  --log-depth-offsets=-0.35,0.35 \
  --center-offsets-px=-48,48 \
  --broad-seeds 4 \
  --quality-weight 1.0 \
  --confidence-weight 0.5 \
  --silhouette-weight 1.5 \
  --measurement-weight 0.05 \
  --history-weight 0.0 \
  --translation-label-scale-m 1.0 \
  --rotation-label-scale-deg 20.0 \
  --sequence-max-frame-gap 1 \
  --sequence-max-time-gap-s 0.5 \
  --device cuda \
  --overwrite


```

## 6. Run the GRU selector with adaptive orientation and fixed lag

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.select \
  --data "$BENCH_ROOT/gru_candidates" \
  --checkpoint "$SELECTOR_ROOT/models/dino_matching_gru/best.ckpt" \
  --output-dir "$BENCH_ROOT/gru_selector_fixed_lag" \
  --minimum-recovery-probability 0.0 \
  --minimum-recovery-margin-over-gigapose 0.0 \
  --temporal-orientation \
  --orientation-fixed-lag \
  --orientation-soft-start-deg 45 \
  --orientation-hard-limit-deg 90 \
  --flip-min-angle-deg 135 \
  --rotation-penalty-weight 1.0 \
  --flip-confirmation-frames 2 \
  --stable-orientation-frames 4 \
  --stable-flip-confirmation-frames 5 \
  --flip-min-probability 0.15 \
  --flip-min-margin 0.03 \
  --flip-consistency-deg 45 \
  --maximum-angular-prediction-deg 30 \
  --orientation-velocity-window 5 \
  --device cuda \
  --overwrite

# -----------------------new bench marck---------------
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.select \
  --data "$BENCH_ROOT/gru_candidates" \
  --checkpoint "$SELECTOR_ROOT/models/dino_matching_gru/best.ckpt" \
  --output-dir "$BENCH_ROOT/gru_selector_fixed_lag" \
  --minimum-recovery-probability 0.0 \
  --minimum-recovery-margin-over-gigapose 0.0 \
  --temporal-orientation \
  --orientation-fixed-lag \
  --orientation-soft-start-deg 45 \
  --orientation-hard-limit-deg 90 \
  --flip-min-angle-deg 135 \
  --rotation-penalty-weight 1.0 \
  --flip-confirmation-frames 2 \
  --stable-orientation-frames 4 \
  --stable-flip-confirmation-frames 5 \
  --flip-min-probability 0.15 \
  --flip-min-margin 0.03 \
  --flip-consistency-deg 45 \
  --maximum-angular-prediction-deg 30 \
  --orientation-velocity-window 5 \
  --device cuda \
  --overwrite

```

## 7. Run GRU-Kalman directly on raw GigaPose

This is the Kalman-alone branch; it intentionally has no selector or adaptive
orientation stage.

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.export_measurements \
  --dataset-dir "$BENCH_DATASET" \
  --split test \
  --predictions "$GP_CSV" \
  --prediction-translation-unit mm \
  --sequence-max-frame-gap 1 \
  --sequence-max-time-gap-s 0.5 \
  --output-dir "$BENCH_ROOT/measurements_raw_gigapose" \
  --overwrite

python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.filter \
  --data "$BENCH_ROOT/measurements_raw_gigapose" \
  --checkpoint "$RAW_KALMAN_ROOT/model/best.ckpt" \
  --output-dir "$BENCH_ROOT/gru_kalman_alone" \
  --max-correction-translation-m 5 \
  --max-correction-rotation-deg 60 \
  --device cuda \
  --overwrite

# -----------------------new bench marck---------------
python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.export_measurements \
  --dataset-dir "$POSE_DATASET" \
  --split test \
  --predictions "$GP_CSV" \
  --prediction-translation-unit mm \
  --sequence-max-frame-gap 1 \
  --sequence-max-time-gap-s 0.5 \
  --output-dir "$BENCH_ROOT/measurements_raw_gigapose" \
  --overwrite


  python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.filter \
  --data "$BENCH_ROOT/measurements_raw_gigapose" \
  --checkpoint "$RAW_KALMAN_ROOT/model/best.ckpt" \
  --output-dir "$BENCH_ROOT/gru_kalman_alone" \
  --max-correction-translation-m 5 \
  --max-correction-rotation-deg 60 \
  --device cuda \
  --overwrite





```

## 8. Adapt the fixed-lag selector output for the two cascade filters

The selector output already contains adaptive-orientation and fixed-lag
decisions. This measurement export preserves those decisions as the cascade
input.

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.export_measurements \
  --dataset-dir "$BENCH_DATASET" \
  --split test \
  --predictions "$BENCH_ROOT/gru_selector_fixed_lag/tracked_predictions.csv" \
  --prediction-translation-unit mm \
  --sequence-max-frame-gap 1 \
  --sequence-max-time-gap-s 0.5 \
  --output-dir "$BENCH_ROOT/measurements_fixed_lag_selector" \
  --overwrite

# -----------------------new bench marck---------------


```

## 9. Run translation-only and full-pose cascade filters

The translation-only model preserves the fixed-lag selector rotation exactly.
The full-pose model starts from that rotation but may smooth/correct it, with a
45-degree safety fallback relative to the selector measurement.

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.filter \
  --data "$BENCH_ROOT/measurements_fixed_lag_selector" \
  --checkpoint "$CASCADE_MODEL_ROOT/model_translation_only/best.ckpt" \
  --output-dir "$BENCH_ROOT/cascade_translation_only" \
  --max-correction-translation-m 100 \
  --max-correction-rotation-deg 180 \
  --device cuda \
  --overwrite

python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.filter \
  --data "$BENCH_ROOT/measurements_fixed_lag_selector" \
  --checkpoint "$CASCADE_MODEL_ROOT/model_full_pose/best.ckpt" \
  --output-dir "$BENCH_ROOT/cascade_full_pose" \
  --max-correction-translation-m 100 \
  --max-correction-rotation-deg 45 \
  --device cuda \
  --overwrite
```

## 10. Full paired evaluation, distance plots, and overlays

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.evaluate_backbones \
  --dataset-dir "$BENCH_DATASET" \
  --split test \
  --model gigapose="$GP_CSV" \
  --model rgb_self_recovery="$BENCH_ROOT/rgb_recovery/cnn_multicheckpoint" \
  --model dino_rgb_recovery="$BENCH_ROOT/rgb_recovery/dino_matching_unet" \
  --model gru_selector_fixed_lag="$BENCH_ROOT/gru_selector_fixed_lag" \
  --model gru_kalman_alone="$BENCH_ROOT/gru_kalman_alone" \
  --model cascade_translation_only="$BENCH_ROOT/cascade_translation_only" \
  --model cascade_full_pose="$BENCH_ROOT/cascade_full_pose" \
  --baseline gigapose \
  --prediction-translation-unit mm \
  --mesh "$BENCH_DATASET/models/obj_000001.ply" \
  --distance-bins-m 0,20,40,60,80,100,120,140 \
  --max-overlays 400 \
  --output-dir "$BENCH_ROOT/comparison/full_benchmark" \
  --overwrite

python3 -m json.tool "$BENCH_ROOT/comparison/full_benchmark/summary.json"
```

The main outputs are:

- `summary.json`: overall and distance-binned metrics plus paired improvements.
- `per_frame_metrics.csv`: every model's error on every available frame.
- `translation_error_by_distance.png` and `rotation_error_by_distance.png`.
- `overlays/`: common-frame CAD overlays for visual comparison.
- Each selector/filter directory also contains its own diagnostics and run report.
