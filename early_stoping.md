# Early stopping for GigaPose fine-tuning

Early stopping is available, but disabled by default, in all five fine-tuning
entry points:

- `python -m fine_tuning.train`
- `python -m fine_tuning.train_val`
- `python -m fine_tuning.pose_aware_training.train`
- `python -m fine_tuning.ot_training.train`
- `python -m fine_tuning.residual_pose_training.train`

Add `--early-stopping` to enable it. `--max-steps` remains a hard upper bound,
so training ends at whichever happens first: the early-stopping condition or
`--max-steps`.

At startup, the trainer prints `EARLY STOPPING CONFIGURED` with the selected
metric, mode, delta, patience, and start step. At the first completed normal
validation at or after the start step, rank zero prints `EARLY STOPPING ACTIVE`.
For example, with a start step of 2000 and validation every 250 steps, the
activation message is printed at step 2000. Sanity validation never activates
the callback or consumes patience.

After every active validation, rank zero also prints the patience transition:

- `EARLY STOPPING BEST INITIALIZED` for the first eligible metric.
- `EARLY STOPPING PATIENCE` when a stale validation increments the counter.
- `EARLY STOPPING PATIENCE RESET` when a sufficient improvement resets a
  nonzero counter.
- `EARLY STOPPING IMPROVED` for consecutive sufficient improvements when the
  counter was already zero.
- `EARLY STOPPING PATIENCE EXHAUSTED` when the counter reaches its limit.
- `EARLY STOPPING STOP CONDITION REACHED` for a NaN, divergence threshold, or
  explicit target threshold rather than patience exhaustion.

## Pose score

Pose-aware and residual-pose training use a joint metric instead of stopping
on translation alone:

\[
S_{\mathrm{pose}}
=
\frac{E_t}{1000}
+
\frac{E_R}{10},
\]

where `E_t` is mean translation error in millimeters and `E_R` is mean rotation
error in degrees. Lower is better. A score change of 1.0 corresponds to either
1000 mm of translation error or 10 degrees of rotation error. Because the
formula is linear, the batch-size-weighted full-validation mean has the same
meaning as calculating the score from the full-validation mean errors.

The logged names are:

- Pose-aware IST: `val/monitor_pose_score`
- Residual pose: `val/monitor_refined_pose_score`
- Residual baseline, for comparison only: `val/monitor_baseline_pose_score`

The residual score always evaluates the refined pose. If the rotation residual
head is disabled, the refined rotation is the baseline IST rotation, so a
translation improvement cannot hide a rotation regression.

## Default stopping metric by training regime

| Entry point | Default monitored metric | Mode | `min_delta` | Patience | Start step |
| --- | --- | --- | ---: | ---: | ---: |
| `fine_tuning.train` | `val/loss`; `val/matching` for AE-only | min | 0.0001 | 16 | 2000 |
| `fine_tuning.train_val` | `val/loss`; `val/matching` for AE-only | min | 0.0001 | 16 | 2000 |
| `fine_tuning.pose_aware_training.train` | `val/monitor_pose_score` | min | 0.01 | 16 | 2000 |
| `fine_tuning.ot_training.train` | `val/monitor_ot_gt_top1_accuracy` | max | 0.001 | 12 | 2000 |
| `fine_tuning.residual_pose_training.train` | `val/monitor_refined_pose_score` | min | 0.01 | 16 | 2000 |

These are defaults, not restrictions. Any epoch-aggregated validation metric
can be selected with `--early-stopping-monitor`, together with the correct
`--early-stopping-mode min` or `max`.

Pose-aware training still saves its scale-ranked checkpoints. Early stopping
uses the joint pose score because choosing only the best scale can retain a
checkpoint whose rotation or metric translation is worse. OT continues to use
GT top-1 correspondence accuracy because it does not produce a metric 6D pose.
Legacy IST does not reconstruct the full metric pose inside its lightweight
validation, so its task loss is used there. In legacy `all` mode, `val/loss`
contains both the IST regression losses and AE InfoNCE; it no longer silently
ignores the AE part of the combined optimization.

## Training-mode coverage

- Legacy `--nets-to-train ist`: `val/loss` is IST scale plus in-plane loss.
- Legacy `--nets-to-train ae`: the default is `val/matching`; AE InfoNCE is also
  logged in `val/loss` and `val/infoNCE`.
- Legacy `--nets-to-train all`: `val/loss` is IST scale/in-plane plus AE
  InfoNCE, so the combined validation objective includes both trained networks.
- Pose-aware `--nets-to-train ist`: the default is the reconstructed joint pose
  score.
- Pose-aware `--nets-to-train all`: the default remains the joint pose score,
  because final pose quality is the downstream target. Use
  `--early-stopping-monitor val/loss` if you deliberately want the combined
  IST plus retrieval objective to determine stopping instead.
- OT training: AE/DINO descriptors are the only optimized network and GT OT
  top-1 accuracy is the default stopping metric.
- Residual `--nets-to-train ist --train-ist`: IST and the enabled residual heads
  are optimized together.
- Residual `--nets-to-train ist --no-train-ist`: only the enabled residual heads
  are optimized.
- Residual `--nets-to-train all --train-ist`: AE, IST, and the enabled residual
  heads are optimized.
- Residual `--nets-to-train all --no-train-ist`: AE and the enabled residual
  heads are optimized while IST remains frozen.

Every residual combination monitors `val/monitor_refined_pose_score`. With
`--no-rotation-residual`, the score still contains the unchanged baseline
rotation error; with `--rotation-residual`, it contains the refined rotation
error. As with pose-aware `all` mode, residual `all` mode may explicitly use
`val/loss` if combined objective loss is preferred over final refined pose.

## Options

- `--early-stopping` enables the feature; `--no-early-stopping` disables it.
- `--early-stopping-monitor NAME` selects the validation metric.
- `--early-stopping-mode {min,max}` says which direction is better.
- `--early-stopping-min-delta VALUE` is the minimum absolute improvement that
  resets patience. Smaller changes count as stale.
- `--early-stopping-patience N` allows `N` completed validation checks without
  sufficient improvement before stopping.
- `--early-stopping-start-step N` prevents sanity validation and early noisy
  training from updating the best score, ranked checkpoints, or patience before
  step `N`.
- `--early-stopping-stopping-threshold VALUE` optionally stops immediately once
  a desired target is crossed. In `min` mode this means below the target; in
  `max` mode it means above it.
- `--early-stopping-divergence-threshold VALUE` optionally stops immediately
  when the metric becomes unacceptably bad. In `min` mode this means above the
  threshold; in `max` mode it means below it.
- `--early-stopping-check-finite` stops on NaN or infinity. This is enabled by
  default; use `--no-early-stopping-check-finite` only for deliberate debugging.
- `--early-stopping-best-checkpoints N` retains the top `N` monitored
  checkpoints when that metric does not already have a specialized top-k
  checkpoint callback.

Patience is counted in validation checks, not optimizer steps or epochs. With
`--validation-interval 250` and `--early-stopping-patience 16`, the stale window
is approximately `250 * 16 = 4000` optimizer steps. The first eligible
validation establishes the initial best value; subsequent insufficient
improvements consume patience.

## Checkpoints

All checkpoint files are under:

`gigaPose_datasets/results/<run-name>/checkpoints/`

While early stopping is enabled:

- Existing periodic checkpoints continue to be written at
  `--checkpoint-interval`.
- If the stopping metric already has a specialized callback, that callback is
  reused, such as `best-ot-step*.ckpt`.
- Otherwise, the top monitored models are written as
  `best-early-step*.ckpt`.
- `last.ckpt` is refreshed after every completed validation, including the
  validation that triggers stopping.

For inference or a new fine-tuning run, normally select the best checkpoint for
the stopping metric, not `last.ckpt`. The latter is useful for inspecting or
resuming the final validation state. These training entry points load
`--checkpoint` as model weights only; they intentionally start a fresh optimizer,
global step, and early-stopping patience counter.

## Examples

Pose-aware IST with the recommended joint pose score:

```bash
python -m fine_tuning.pose_aware_training.train \
  ...your existing arguments... \
  --early-stopping \
  --early-stopping-patience 16 \
  --early-stopping-min-delta 0.01 \
  --early-stopping-start-step 2000
```

Residual translation/rotation training uses the refined joint score by default:

```bash
python -m fine_tuning.residual_pose_training.train \
  ...your existing arguments... \
  --rotation-residual \
  --early-stopping \
  --early-stopping-patience 16 \
  --early-stopping-min-delta 0.01
```

OT training, where higher correspondence accuracy is better:

```bash
python -m fine_tuning.ot_training.train \
  ...your existing arguments... \
  --early-stopping \
  --early-stopping-patience 12 \
  --early-stopping-min-delta 0.001
```

Legacy IST/light-heavy validation:

```bash
python -m fine_tuning.train_val \
  ...your existing arguments... \
  --early-stopping \
  --early-stopping-monitor val/loss \
  --early-stopping-mode min
```

Heavy CAD overlays are diagnostic and are not used for stopping. The stopping
decision always comes from the normal full validation loader.

## Full-validation aggregation and DDP

Validation metrics used by these callbacks are logged with:

- `on_step=False`
- `on_epoch=True`
- the actual batch size
- distributed synchronization enabled

Therefore a single unusually good or bad validation batch cannot trigger or
reset early stopping. The metric is batch-size weighted across the completed
validation loader and synchronized across DDP ranks before checkpoint ranking
and the stopping decision.
