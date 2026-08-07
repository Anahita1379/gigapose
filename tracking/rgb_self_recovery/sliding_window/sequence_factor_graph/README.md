# Sequence factor-graph full-pose optimizer

This isolated package replaces temporal GRU/Kalman filtering with sparse batch
SE(3) optimization. It does not modify candidate generation, the GRU selector,
GRU-Kalman, fixed-lag smoother, transformers, or their saved outputs.

Two experiments use the same factors and differ only in initialization/prior:

- **Mode A — `candidates`:** RGB/CAD candidates only. Initialization is the
  lowest visual-cost candidate at each frame.
- **Mode B — `prior`:** any existing trajectory initializes the graph and is
  retained as a soft factor together with RGB/CAD candidates. The prior can be
  the existing full-pose cascade now, or a numeric/visual transformer later.

Mode A is the clean temporal-model replacement. Mode B is an ablation that
tests whether a strong external initialization/prior helps the nonlinear graph;
the graph still jointly optimizes the entire sequence using past and future.

Each exported contiguous `segment_id` is optimized independently. The graph
contains robust candidate-pose factors, translation acceleration and jerk
factors, and angular-acceleration factors. Candidate selection is a hard
max-mixture approximation: alternate RGB/CAD/pose-consistency assignment and
continuous sparse least-squares optimization. Ground truth is never passed to
the optimizer; when present in the bundle it is used only for `run_report.json`.

## Run both modes

```bash
cd /home/anahita/gigapose

export ROOT="$PWD/gigaPose_datasets/results/rgb_self_recovery_benchmark_pose_regenerated_20260803"
export CANDIDATES="$ROOT/gru_candidates"
export CASCADE_FULL="$ROOT/cascade_full_pose/tracked_predictions.csv"
export GRAPH="$ROOT/sequence_factor_graph"

# Mode A: candidates only
python3 -m tracking.rgb_self_recovery.sliding_window.sequence_factor_graph.run \
  --data "$CANDIDATES" \
  --mode candidates \
  --output-dir "$GRAPH/candidates_only" \
  --outer-iterations 3 \
  --maximum-nfev 100 \
  --overwrite

# Mode B: existing full-pose cascade soft prior + candidates
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

Both output directories contain:

- `tracked_predictions.csv`: optimized full-pose trajectory;
- `initial_predictions.csv`: visual initialization or external prior input;
- `diagnostics.jsonl`: selected candidate and per-frame pose changes;
- `run_report.json`: factor settings, segment solver status, and optional GT
  before/after metrics.

## Compare against existing benchmark outputs

```bash
export DATASET="$PWD/gigaPose_datasets/datasets/rgb_recovery_benchmark_assetto_20260803_pose_regenerated"
export GP_CSV="$PWD/gigaPose_datasets/results/gigapose_rgb_recovery_benchmark_pose_regenerated_20260803/predictions/large-pbrreal-rgb-mmodel_rgb_recovery_benchmark_assetto_20260803_pose_regenerated-test_gigapose_rgb_recovery_benchmark_pose_regenerated_20260803MultiHypothesis.csv"

python3 -m tracking.rgb_self_recovery.sliding_window.evaluate_backbones \
  --dataset-dir "$DATASET" --split test \
  --model gigapose="$GP_CSV" \
  --model cascade_translation_only="$ROOT/cascade_translation_only" \
  --model cascade_full_pose="$ROOT/cascade_full_pose" \
  --model graph_candidates="$GRAPH/candidates_only" \
  --model graph_cascade_prior="$GRAPH/cascade_prior_plus_candidates" \
  --baseline cascade_full_pose \
  --prediction-translation-unit mm \
  --mesh "$DATASET/models/obj_000001.ply" \
  --distance-bins-m 0,20,40,60,80,100,120,140 \
  --max-overlays 160 \
  --output-dir "$GRAPH/comparison" \
  --overwrite
```

The defaults are deliberately soft because these poses are camera-relative.
When reliable `T_map_camera` is available, a later map-frame graph can use much
tighter physical acceleration and track factors.

## Use a transformer as Mode B prior later

The graph accepts any one-pose-per-frame tracking CSV:

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

No fixed-lag smoother is required after this offline graph. Fixed lag is useful
only for online deployment, where the complete future sequence is unavailable.
