# GRU-Kalman SE(3) filter

This isolated model learns to filter raw GigaPose pose measurements over time.
It does not use an RGB-recovery checkpoint and does not modify existing code.
It is inspired by KalmanNet/GRUTrack, but is adapted to one-car camera-frame
SE(3) pose tracking rather than nuScenes multi-object 3D boxes.

At each frame it:

1. predicts translation and rotation with learned linear/angular velocity;
2. computes translation and SO(3) rotation innovations against raw GigaPose;
3. uses a causal GRU to estimate diagonal pose and velocity gains;
4. updates the filtered pose and motion state;
5. resets at every run, camera, missing-frame, or timestamp discontinuity.

Ground truth supervises training but is never a model input.

The optional component fallback heads separately estimate whether the filtered
translation and filtered rotation are safer than the original GigaPose pose.
Low translation trust restores only GigaPose translation; low rotation trust
restores only GigaPose rotation. Ground truth creates these binary targets only
during training and validation. It is not required during deployment.

Correction-size guards are separate from learned fallback. An extreme
correction restores the corresponding selector measurement component, while a
low trust probability restores the corresponding original GigaPose component.

## 1. Export measurements after GigaPose finishes

```bash
cd /home/anahita/gigapose
conda activate gigapose

export KF_ROOT="$PWD/gigaPose_datasets/results/rgb_self_recovery_gru_kalman"
export TRAIN_DATASET="$PWD/gigaPose_datasets/datasets/rgb_recovery_gru_train_assetto"
export VAL_DATASET="$PWD/gigaPose_datasets/datasets/rgb_recovery_gru_validation_laguna_putnam"
export TRAIN_GP="$PWD/gigaPose_datasets/results/gigapose_rgb_recovery_gru_train_assetto"
export VAL_GP="$PWD/gigaPose_datasets/results/gigapose_rgb_recovery_gru_validation_laguna_putnam"

python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.export_measurements \
  --dataset-dir "$TRAIN_DATASET" --predictions "$TRAIN_GP" \
  --output-dir "$KF_ROOT/data/train" --overwrite

python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.export_measurements \
  --dataset-dir "$VAL_DATASET" --predictions "$VAL_GP" \
  --output-dir "$KF_ROOT/data/validation" --overwrite
```

## 2. Train

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.train \
  --data "$KF_ROOT/data/train" \
  --validation-data "$KF_ROOT/data/validation" \
  --output-dir "$KF_ROOT/models/filter_v1" \
  --clip-length 16 --clip-stride 4 --validation-stride 16 \
  --epochs 100 --batch-size 32 --learning-rate 2e-4 \
  --patience 10 --min-delta 1e-4 --device cuda --overwrite

python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.plot_history \
  --run-dir "$KF_ROOT/models/filter_v1"
```

## 3. Filter the balanced validation set

First run without restrictive fallback limits to measure the learned filter:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.filter \
  --data "$KF_ROOT/data/validation" \
  --checkpoint "$KF_ROOT/models/filter_v1/best.ckpt" \
  --max-correction-translation-m 100 \
  --max-correction-rotation-deg 180 \
  --output-dir "$KF_ROOT/runs/validation_raw" \
  --device cuda --overwrite
```

Then test a guarded deployment version:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.filter \
  --data "$KF_ROOT/data/validation" \
  --checkpoint "$KF_ROOT/models/filter_v1/best.ckpt" \
  --max-correction-translation-m 5 \
  --max-correction-rotation-deg 60 \
  --output-dir "$KF_ROOT/runs/validation_guarded" \
  --device cuda --overwrite
```

Evaluate and generate overlays with the existing evaluator:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.evaluate_backbones \
  --dataset-dir "$VAL_DATASET" --split test \
  --model gigapose="$VAL_GP" \
  --model gru_kalman="$KF_ROOT/runs/validation_raw" \
  --model gru_kalman_guarded="$KF_ROOT/runs/validation_guarded" \
  --baseline gigapose \
  --mesh "$VAL_DATASET/models/obj_000001.ply" \
  --distance-bins-m 0,20,40,60,80,100,120 \
  --max-overlays 328 \
  --output-dir "$KF_ROOT/comparison/validation" --overwrite
```

The balanced Laguna/Putnam set is validation data. A new physical run is still
required for an unbiased final evaluation.

## Independent translation and rotation fallback heads

Exporting a cascade measurement bundle requires both the selector/cascade
measurement and the original GigaPose prediction. The latter is the fallback
baseline:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.export_measurements \
  --dataset-dir "$TRAIN_DATASET" \
  --ground-truth-dataset-dir "$POSE_REGENERATED_TRAIN_DATASET" \
  --predictions "$SELECTOR_TRAIN_CSV" \
  --fallback-baseline-predictions "$GIGAPOSE_TRAIN_CSV" \
  --output-dir "$KF_ROOT/data/fallback_train" --overwrite
```

Warm-start an existing cascade and train only the two heads first. This keeps
the pose filter fixed, making the fallback experiment directly comparable with
the old cascade:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.train \
  --data "$KF_ROOT/data/fallback_train" \
  --validation-data "$KF_ROOT/data/fallback_validation" \
  --output-dir "$KF_ROOT/models/fallback_heads" \
  --initialize-from-checkpoint "$KF_ROOT/models/full_pose/best.ckpt" \
  --use-fallback-heads --freeze-filter-backbone \
  --translation-fallback-margin-m 1.0 \
  --rotation-fallback-margin-deg 5.0 \
  --clip-length 16 --clip-stride 4 --validation-stride 16 \
  --epochs 100 --batch-size 32 --learning-rate 2e-4 \
  --patience 10 --min-delta 1e-4 --device cuda --overwrite
```

Run validation once at permissive trust thresholds so the diagnostics contain
every candidate, baseline, error, and trust probability. Then calibrate the two
thresholds independently. The conservative settings below limit learned
fallback to 2% of validation frames and require at least 70% fallback precision:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.filter \
  --data "$KF_ROOT/data/fallback_validation" \
  --checkpoint "$KF_ROOT/models/fallback_heads/best.ckpt" \
  --max-correction-translation-m 100 --max-correction-rotation-deg 45 \
  --minimum-translation-trust 0.5 --minimum-rotation-trust 0.5 \
  --output-dir "$KF_ROOT/calibration/fallback_validation" \
  --device cuda --overwrite

python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.calibrate_fallback \
  --diagnostics "$KF_ROOT/calibration/fallback_validation/filter_diagnostics.csv" \
  --output "$KF_ROOT/calibration/fallback_validation/thresholds.json" \
  --maximum-fallback-fraction 0.03 \
  --minimum-fallback-precision 0.60 \
  --minimum-mean-improvement 0.01 \
  --max-correction-translation-m 100 --max-correction-rotation-deg 45
```

Use the two thresholds written to `thresholds.json` on untouched test data.
A threshold of `0` disables that learned component fallback while retaining its
diagnostics and the correction-size safety guard.

## Adaptive selector followed by a translation-only GRU-Kalman filter

This cascade preserves the adaptive selector's rotations exactly and retrains
the filter on selector-output translations. Do not reuse the filter trained on
raw GigaPose: its measurement distribution is different.

```text
RGB/mask/CAD recovery candidates
                |
                v
one-layer GRU candidate selector
                |
                v
adaptive temporal orientation + flip hysteresis
                |
                v
translation-only GRU-Kalman filter
                |
                v
final pose (filtered translation, selector rotation)
```

Set paths:

```bash
export SELECTOR_ROOT="$PWD/gigaPose_datasets/results/rgb_self_recovery_gru_dino_matching"
export CASCADE_ROOT="$PWD/gigaPose_datasets/results/rgb_self_recovery_adaptive_selector_kalman"
export TRAIN_DATASET="$PWD/gigaPose_datasets/datasets/rgb_recovery_gru_train_assetto"
export VAL_DATASET="$PWD/gigaPose_datasets/datasets/rgb_recovery_gru_validation_laguna_putnam"
```

Replay the same trained one-layer selector with adaptive orientation on both
candidate bundles. Ground truth is not used by selection:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.select \
  --data "$SELECTOR_ROOT/data/train" \
  --checkpoint "$SELECTOR_ROOT/models/dino_matching_gru/best.ckpt" \
  --temporal-orientation \
  --flip-confirmation-frames 2 \
  --stable-orientation-frames 4 \
  --stable-flip-confirmation-frames 5 \
  --minimum-recovery-probability 0 \
  --minimum-recovery-margin-over-gigapose 0 \
  --output-dir "$CASCADE_ROOT/selector/train" \
  --device cuda --overwrite

python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.select \
  --data "$SELECTOR_ROOT/data/validation" \
  --checkpoint "$SELECTOR_ROOT/models/dino_matching_gru/best.ckpt" \
  --temporal-orientation \
  --flip-confirmation-frames 2 \
  --stable-orientation-frames 4 \
  --stable-flip-confirmation-frames 5 \
  --minimum-recovery-probability 0 \
  --minimum-recovery-margin-over-gigapose 0 \
  --output-dir "$CASCADE_ROOT/selector/validation" \
  --device cuda --overwrite
```

Export those selected poses as the second-stage measurement bundles. Pass the
CSV file itself, not its parent result directory:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.export_measurements \
  --dataset-dir "$TRAIN_DATASET" \
  --predictions "$CASCADE_ROOT/selector/train/tracked_predictions.csv" \
  --prediction-translation-unit mm \
  --output-dir "$CASCADE_ROOT/data/train" --overwrite

python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.export_measurements \
  --dataset-dir "$VAL_DATASET" \
  --predictions "$CASCADE_ROOT/selector/validation/tracked_predictions.csv" \
  --prediction-translation-unit mm \
  --output-dir "$CASCADE_ROOT/data/validation" --overwrite
```

Train a new translation-only filter. Rotation error remains in the logged
metrics, but it is excluded from the optimization objective because output
rotation is copied from the adaptive selector:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.train \
  --data "$CASCADE_ROOT/data/train" \
  --validation-data "$CASCADE_ROOT/data/validation" \
  --output-dir "$CASCADE_ROOT/model" \
  --preserve-measurement-rotation \
  --measurement-preservation-weight 0.05 \
  --clip-length 16 --clip-stride 4 --validation-stride 16 \
  --epochs 100 --batch-size 32 --learning-rate 2e-4 \
  --patience 10 --min-delta 1e-4 \
  --device cuda --overwrite
```

First evaluate the learned translation without a restrictive correction guard:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.filter \
  --data "$CASCADE_ROOT/data/validation" \
  --checkpoint "$CASCADE_ROOT/model/best.ckpt" \
  --max-correction-translation-m 100 \
  --max-correction-rotation-deg 180 \
  --output-dir "$CASCADE_ROOT/validation_raw" \
  --device cuda --overwrite
```

Compare GigaPose, the adaptive selector measurement, and the cascade. The
GigaPose model must point to its actual MultiHypothesis CSV:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.evaluate_backbones \
  --dataset-dir "$VAL_DATASET" --split test \
  --model gigapose="$VAL_GP_CSV" \
  --model adaptive_selector="$CASCADE_ROOT/selector/validation" \
  --model adaptive_selector_kalman="$CASCADE_ROOT/validation_raw" \
  --baseline adaptive_selector \
  --prediction-translation-unit mm \
  --mesh "$VAL_DATASET/models/obj_000001.ply" \
  --distance-bins-m 0,20,40,60,80,100,120 \
  --max-overlays 328 \
  --output-dir "$CASCADE_ROOT/comparison/validation" --overwrite
```

The acceptance criterion is strict: adaptive-selector rotation metrics must be
identical, while translation median should not materially regress and the
60--80 m translation p90 should improve. Tune translation preservation or a
translation-only fallback threshold only after inspecting the unguarded run.
