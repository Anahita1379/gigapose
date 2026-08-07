# GRU-Kalman fixed-lag translation smoother

This is a separate package. It does not modify the GRU-Kalman model, selector,
fallback-head package, or their existing outputs.

The package performs:

```text
selector measurement + original GigaPose
                    |
                    v
          existing causal GRU-Kalman
                    |
                    v
       centered five-frame motion regression
                    |
                    v
 trajectory-gated candidate replacement and local translation refinement
                    |
                    v
       corrected translation + unchanged GRU rotation
```

For target frame `t`, the default five-frame window is `t-2 ... t+2`.
Consequently, online deployment has a two-frame delay. The motion regression
fits position, velocity, and acceleration to the other four causal estimates.
It computes an adaptive gate from fit residuals, median absolute deviation, and
object depth. A suspicious causal translation is replaced only when the
selector or original GigaPose candidate is materially more consistent with the
local trajectory.

Ground truth is used only for metrics when it exists in the measurement bundle.
It is never read by the detector, candidate selection, or reconstruction.

## Run

Use the original full-pose GRU-Kalman checkpoint, not a fallback-head
checkpoint:

```bash
cd /home/anahita/gigapose
conda activate gigapose

export ROOT="$PWD/gigaPose_datasets/results/rgb_self_recovery_benchmark_pose_regenerated_20260803"
export DATA="$ROOT/fallback_data/full"
export KF_CHECKPOINT="$PWD/gigaPose_datasets/results/rgb_self_recovery_fixed_lag_kalman/model_full_pose/best.ckpt"

python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_fixed_lag_translation.run \
  --data "$DATA" \
  --checkpoint "$KF_CHECKPOINT" \
  --output-dir "$ROOT/cascade_gru_kalman_fixed_lag_translation" \
  --window-size 5 --lag 2 \
  --minimum-gate-m 1.5 \
  --depth-gate-fraction 0.02 \
  --minimum-candidate-improvement-m 1.0 \
  --motion-blend 0.10 \
  --device cuda --overwrite
```

The output contains:

- `tracked_predictions.csv`: final pose predictions;
- `causal_predictions.csv`: the unchanged first-pass GRU output for paired
  before/after evaluation;
- `fixed_lag_diagnostics.csv`: every gate, residual, candidate source, velocity,
  acceleration, and repair decision;
- `run_report.json`: configuration and before/after metrics;
- `replay_inputs.npz`: corrected measurements and replay start indices reserved
  for the future state-replay implementation.

## Important behavior

- Rotation is copied exactly from the causal GRU-Kalman output.
- A frame is not changed merely because GigaPose disagrees with the GRU.
- The local motion fit must be reliable.
- An alternate candidate must improve trajectory residual by the configured
  minimum.
- Repairs are capped by `--maximum-repair-m`.
- `--allow-motion-only-repair` is off by default. Therefore the optimizer does
  not invent a translation when neither GigaPose nor the selector supports it.

The first command prepares state replay but does not perform it automatically.
The separate replay command consumes the prepared inputs and reruns each whole
segment. Whole-segment replay is mathematically equivalent to restoring the
state immediately before the earliest repair, but avoids changing the existing
GRU API to expose hidden state:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_kalman_fixed_lag_translation.replay \
  --data "$DATA" \
  --checkpoint "$KF_CHECKPOINT" \
  --fixed-lag-run "$ROOT/cascade_gru_kalman_fixed_lag_translation" \
  --output-dir "$ROOT/cascade_gru_kalman_fixed_lag_translation_replay" \
  --device cuda --overwrite
```

Replay writes two versions:

- `replayed_translation_only_predictions.csv` is recommended. It uses replayed
  translation and preserves causal GRU rotation.
- `replayed_full_pose_predictions.csv` is diagnostic. It permits the replayed
  recurrent state to alter both translation and rotation.
- `tracked_predictions.csv` is an alias of the recommended translation-only
  replay.

Ground truth is not used during replay. `replay_diagnostics.csv` records how a
repaired measurement propagates to subsequent recurrent states.

## Tests

```bash
python3 -m unittest \
  tracking.rgb_self_recovery.sliding_window.gru_kalman_fixed_lag_translation.test_smoother
```
