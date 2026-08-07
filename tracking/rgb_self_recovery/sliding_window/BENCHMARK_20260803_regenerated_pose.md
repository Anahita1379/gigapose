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
python3 -m json.tool "$POSE_DATASET/evaluation_metadata.json"

```
The preparer records rows without a valid joined instance ID under `skipped`
instead of assigning a potentially incorrect mask.

## 3. Run GigaPose

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.run_gigapose \
  --dataset-name "$POSE_DATASET_NAME" \
  --run-name gigapose_rgb_recovery_benchmark_pose_regenerated_20260803 \
  --batch-size 16 \
  --num-workers 2 \
  --devices 0


export GP_CSV="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["predictions"])' "$GP_RESULT/gru_gigapose_manifest.json")"
test -f "$GP_CSV" && echo "$GP_CSV"
```


## 4. Run standalone RGB self-recovery models

This runs the original CNN recovery and the DINO matching U-Net used to create
the GRU selector's visual candidates. Both use sequence-safe five-frame state
and soft orientation gating.

```bash
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
running this one right now
The selector output already contains adaptive-orientation and fixed-lag
decisions. This measurement export preserves those decisions as the cascade
input.

```bash
# -----------------------new bench marck---------------

python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_filter.export_measurements \
  --dataset-dir "$POSE_DATASET" \
  --split test \
  --predictions "$BENCH_ROOT/gru_selector_fixed_lag/tracked_predictions.csv" \
  --prediction-translation-unit mm \
  --sequence-max-frame-gap 1 \
  --sequence-max-time-gap-s 0.5 \
  --output-dir "$BENCH_ROOT/measurements_fixed_lag_selector" \
  --overwrite
```

## 9. Run translation-only and full-pose cascade filters
running this one right now
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
  --dataset-dir "$POSE_DATASET" \
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
  --mesh "$POSE_DATASET/models/obj_000001.ply" \
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







new Run:
1. Reproduce the full benchmark
```bash 
cd /home/anahita/gigapose
conda activate gigapose

export ROOT="$PWD/gigaPose_datasets/results/rgb_self_recovery_benchmark_pose_regenerated_20260803"
export DATA="$ROOT/fallback_data/full"
export DATASET="$PWD/gigaPose_datasets/datasets/rgb_recovery_benchmark_assetto_20260803_pose_regenerated"
export KF_CHECKPOINT="$PWD/gigaPose_datasets/results/rgb_self_recovery_fixed_lag_kalman/model_full_pose/best.ckpt"

export GP_CSV="$PWD/gigaPose_datasets/results/gigapose_rgb_recovery_benchmark_pose_regenerated_20260803/predictions/large-pbrreal-rgb-mmodel_rgb_recovery_benchmark_assetto_20260803_pose_regenerated-test_gigapose_rgb_recovery_benchmark_pose_regenerated_20260803MultiHypothesis.csv"

export FIXED_LAG_RUN="$ROOT/cascade_gru_kalman_fixed_lag_translation"
export REPLAY_RUN="$ROOT/cascade_gru_kalman_fixed_lag_translation_replay"
export COMPARISON="$ROOT/comparison/all_existing_plus_fixed_lag_replay"

```

Run causal GRU + fixed-lag translation
```bash

python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_fixed_lag_translation.run \
  --data "$DATA" \
  --checkpoint "$KF_CHECKPOINT" \
  --output-dir "$FIXED_LAG_RUN" \
  --window-size 5 \
  --lag 2 \
  --minimum-gate-m 1.5 \
  --depth-gate-fraction 0.02 \
  --minimum-candidate-improvement-m 1.0 \
  --motion-blend 0.10 \
  --device cuda \
  --overwrite

# This creates:
causal_predictions.csv
tracked_predictions.csv              # fixed-lag translation
fixed_lag_diagnostics.csv
replay_inputs.npz
run_report.json
  ```
Run GRU state replay
```bash 
python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_fixed_lag_translation.replay \
  --data "$DATA" \
  --checkpoint "$KF_CHECKPOINT" \
  --fixed-lag-run "$FIXED_LAG_RUN" \
  --output-dir "$REPLAY_RUN" \
  --device cuda \
  --overwrite

  # This creates:
replayed_translation_only_predictions.csv   # recommended replay
replayed_full_pose_predictions.csv          # diagnostic
tracked_predictions.csv                     # alias of translation-only replay
replay_diagnostics.csv
run_report.json
```
Translation-only replay preserves the causal GRU rotation. Full-pose replay allows the rebuilt recurrent state to change rotation too.


Evaluate everything together
  --model rgb_self_recovery="$ROOT/rgb_recovery/cnn_multicheckpoint/tracked_predictions.csv" \
  --model dino_rgb_recovery="$ROOT/rgb_recovery/dino_matching_unet/tracked_predictions.csv" \
  --model gru_selector_fixed_lag="$ROOT/gru_selector_fixed_lag/tracked_predictions.csv" \
  --model gru_kalman_alone="$ROOT/gru_kalman_alone/tracked_predictions.csv" \
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.evaluate_backbones \
  --dataset-dir "$DATASET" \
  --split test \
  --model gigapose="$GP_CSV" \
  --model cascade_translation_only="$ROOT/cascade_translation_only/tracked_predictions.csv" \
  --model cascade_full_pose="$ROOT/cascade_full_pose/tracked_predictions.csv" \
  --model causal_gru="$FIXED_LAG_RUN/causal_predictions.csv" \
  --model fixed_lag_translation="$FIXED_LAG_RUN/tracked_predictions.csv" \
  --model replay_translation_only="$REPLAY_RUN/replayed_translation_only_predictions.csv" \
  --model replay_full_pose="$REPLAY_RUN/replayed_full_pose_predictions.csv" \
  --baseline cascade_full_pose \
  --mesh "$DATASET/models/obj_000001.ply" \
  --prediction-translation-unit mm \
  --distance-bins-m 0,20,40,60,80,100,120,140 \
  --max-overlays 0 \
  --output-dir "$COMPARISON" \
  --overwrite
```
To generate readable overlays, I recommend comparing only four outputs:
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.evaluate_backbones \
  --dataset-dir "$DATASET" \
  --split test \
  --model gigapose="$GP_CSV" \
  --model causal_gru="$FIXED_LAG_RUN/causal_predictions.csv" \
  --model fixed_lag_translation="$FIXED_LAG_RUN/tracked_predictions.csv" \
  --model replay_translation_only="$REPLAY_RUN/replayed_translation_only_predictions.csv" \
  --baseline causal_gru \
  --mesh "$DATASET/models/obj_000001.ply" \
  --prediction-translation-unit mm \
  --distance-bins-m 0,20,40,60,80,100,120,140 \
  --max-overlays 120 \
  --output-dir "$ROOT/comparison/fixed_lag_replay_overlays" \
  --overwrite

```



--------------------------------------------
# new run with factor Graphs: 
----------------------------------------------------------
## Implemented modes

```text
Mode A: RGB/CAD/GigaPose candidate poses for complete sequence
                         ↓
             full-sequence factor graph
                         ↓
                optimized trajectory

This is the clean replacement for:
GRU selector → GRU-Kalman → fixed-lag translation
```
Candidate generation remains necessary because it supplies visual pose measurements, but no temporal GRU output is used.

```text
Mode B: prior-assisted graph
existing trajectory + RGB/CAD candidates
                         ↓
             full-sequence factor graph
                         ↓
                optimized trajectory


```
The external trajectory is optional. It can be:
Existing full-pose cascade.
Numeric transformer output.
Frozen visual-fusion transformer output.
Another tracker.
It serves as:
A good nonlinear optimization initialization.
A soft pose factor that the graph may move away from.

## Factors
Each continuous sequence is optimized independently using:
Robust RGB/CAD candidate-pose factors.
Alternating candidate assignment and continuous optimization.
Translation acceleration factors.
Translation jerk factors.
Angular-acceleration factors on SO(3).
Optional causal-GRU pose factors in Mode B.
Huber robust loss by default.
Candidate assignment combines:
RGB/CAD visual cost.
Consistency with the current graph pose.
Neighbor-interpolated motion consistency.
The neighbor term prevents a visually preferred outlier from trapping the graph at a bad initialization.

## Run Mode A: graph-only replacement
```bash
cd /home/anahita/gigapose

export ROOT="$PWD/gigaPose_datasets/results/rgb_self_recovery_benchmark_pose_regenerated_20260803"
export CANDIDATES="$ROOT/gru_candidates"
export GRAPH="$ROOT/sequence_factor_graph"

python3 -m tracking.rgb_self_recovery.sliding_window.sequence_factor_graph.run \
  --data "$CANDIDATES" \
  --mode candidates \
  --output-dir "$GRAPH/candidates_only" \
  --outer-iterations 3 \
  --maximum-nfev 100 \
  --overwrite
```

## Why an optional prior can help
Full-pose graph optimization is nonlinear and has discrete candidate ambiguity. Examples include:
Front/rear flips.
Several plausible depth candidates.
Visually similar candidates.
Weak evidence at sequence endpoints.
A whole local section initialized around the wrong candidate.
A strong prior can keep the optimizer in the correct basin. But it is not required, and Mode A is the correct experiment if you want a complete GRU replacement.

## Updated Mode B command
Use the actual existing full-pose cascade:
```bash

cd /home/anahita/gigapose

export ROOT="$PWD/gigaPose_datasets/results/rgb_self_recovery_benchmark_pose_regenerated_20260803"
export CANDIDATES="$ROOT/gru_candidates"
export CASCADE_FULL="$ROOT/cascade_full_pose/tracked_predictions.csv"
export GRAPH="$ROOT/sequence_factor_graph"

python3 -m tracking.rgb_self_recovery.sliding_window.sequence_factor_graph.run \
  --data "$CANDIDATES" \
  --mode prior \
  --prior-predictions "$CASCADE_FULL" \
  --prior-translation-unit mm \
  --output-dir "$GRAPH/cascade_prior_plus_candidates" \
  --outer-iterations 3 \
  --maximum-nfev 100 \
  --overwrite
```

## Outputs
Each mode writes:
tracked_predictions.csv: optimized full-pose trajectory.
initial_predictions.csv: input trajectory before optimization.
diagnostics.jsonl: selected candidate and pose change for each frame.
run_report.json: settings, solver status and before/after metrics.
The report distinguishes:
optimizer_success: whether SciPy formally converged before its evaluation limit.
solution_accepted: whether the result was finite and did not regress the graph objective.
If a solution is invalid or materially worsens the objective, that segment automatically falls back to its initial trajectory

## Compare modes
```bash
export DATASET="$PWD/gigaPose_datasets/datasets/rgb_recovery_benchmark_assetto_20260803_pose_regenerated"
export GP_CSV="$PWD/gigaPose_datasets/results/gigapose_rgb_recovery_benchmark_pose_regenerated_20260803/predictions/large-pbrreal-rgb-mmodel_rgb_recovery_benchmark_assetto_20260803_pose_regenerated-test_gigapose_rgb_recovery_benchmark_pose_regenerated_20260803MultiHypothesis.csv"

python3 -m tracking.rgb_self_recovery.sliding_window.evaluate_backbones \
  --dataset-dir "$DATASET" \
  --split test \
  --model gigapose="$GP_CSV" \
  --model causal_gru="$CAUSAL" \
  --model graph_candidates="$GRAPH/candidates_only" \
  --model graph_causal="$GRAPH/causal_plus_candidates" \
  --baseline causal_gru \
  --prediction-translation-unit mm \
  --mesh "$DATASET/models/obj_000001.ply" \
  --distance-bins-m 0,20,40,60,80,100,120,140 \
  --max-overlays 160 \
  --output-dir "$GRAPH/comparison" \
  --overwrite
```


Mode A answers:
Can the factor graph replace temporal GRU/Kalman processing entirely?

Mode B answers a different question:
Can sequence optimization improve an already strong cascade or transformer result?


Transformer comparison later
Use either transformer’s tracking CSV as the generic prior:
```bash
export TRANSFORMER_PREDICTIONS="$RUNS/frozen_visual/tracked_predictions.csv"

python3 -m tracking.rgb_self_recovery.sliding_window.sequence_factor_graph.run \
  --data "$CANDIDATES" \
  --mode prior \
  --prior-predictions "$TRANSFORMER_PREDICTIONS" \
  --prior-translation-unit mm \
  --output-dir "$GRAPH/frozen_transformer_prior_plus_candidates" \
  --overwrite

```
Then compare:
numeric transformer
frozen visual transformer
graph-only
graph + numeric transformer prior
graph + visual transformer prior
existing full-pose cascade
