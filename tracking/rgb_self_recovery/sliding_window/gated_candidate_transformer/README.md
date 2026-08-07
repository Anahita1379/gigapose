# Gated Candidate Transformer

The colored architecture and exact default parameter breakdown are available
in [`transformer_architecture.svg`](transformer_architecture.svg) (with a PNG
copy beside it).

This is an isolated experimental replacement for the GRU-selector plus GRU–Kalman cascade. It does not modify the existing recovery, GRU, Kalman, or evaluation code.

## Model

For an eight-frame window with 16 RGB/CAD pose candidates per frame:

```text
candidate features ── gated candidate cross-attention ──┐
                                                        ├─ candidate score
frame evidence ───── gated temporal cross-attention ────┼─ orientation-mode score
                                                        ├─ SE(3) residual
                                                        ├─ translation/rotation uncertainty
                                                        ├─ translation/rotation trust
                                                        └─ abstention probability
                                                                  │
                                 learned fixed-lag translation refinement
                              (position + velocity + acceleration + jerk losses)
                                                                  │
                                          adaptive fixed-lag orientation guard
                                                                  │
                                                              final pose
```

This uses cross-attention, not global self-attention. A frame query first attends
to its candidates. Separate temporal queries then attend to frame evidence with
a lag-aware mask:

\[
M_{ij}=0\ \text{if}\ j\le i+\ell,\qquad M_{ij}=-\infty\ \text{otherwise}.
\]

The default \(\ell=4\) preserves four-frame fixed-lag evidence. In an
eight-frame window, the deployed target is position \(8-4-1=3\), so it sees
positions 0 through 7, while earlier queries cannot leak frames beyond their
own four-frame future horizon. Invalid/padded frames remain independently
masked. A learned sigmoid gate decides how much local candidate evidence and
temporal evidence to use. Candidate permutation therefore does not act like a
fake temporal ordering.

The network learns candidate selection and continuous pose correction jointly.
It also has a second lag-masked temporal branch that predicts a gated XYZ
correction from frame-local pose proposals, timestamps, and fused evidence.
The final translation—not merely the initial per-frame residual—is supervised
against ground truth. Sequence losses additionally compare ground-truth and
predicted velocity, acceleration, and jerk, so abrupt translation shifts are
learned as part of the model. The translation branch uses the same future-lag
mask and cannot indirectly leak later frames through already-contextualized
key/value tokens.

Two supervised trust heads independently predict whether the refined
translation and refined rotation beat candidate-zero GigaPose. For candidate
\(i\), their targets are

\[
y^t_i=\mathbb 1[e^t_{i,\mathrm{refined}}+m_t<e^t_{\mathrm{GP}}],
\qquad
y^R_i=\mathbb 1[e^R_{i,\mathrm{refined}}+m_R<e^R_{\mathrm{GP}}].
\]

Translation error is Euclidean distance and rotation error is the geodesic
SO(3) angle. Ground truth creates these labels only during training. At
They are retained as optional ablations but have zero loss weight and zero
inference threshold by default. If enabled, each learned trust probability controls its own component: low
translation trust restores GigaPose translation, while low rotation trust
restores GigaPose rotation. The two decisions can differ. Adaptive orientation
and fixed-lag flip handling run after this component-wise fallback. The
deterministic decoder also keeps correction limits and uncertainty-based
abstention.

DINO remains upstream: the candidate exporter uses the trained DINO matching U-Net to create RGB/CAD candidates and features. This transformer consumes that exported bundle rather than raw images.

## Train

```bash
export DATA_ROOT="$PWD/gigaPose_datasets/results/rgb_self_recovery_gru_dino_matching/data"
export TRANSFORMER_ROOT="$PWD/gigaPose_datasets/results/gated_candidate_transformer"

python3 -m tracking.rgb_self_recovery.sliding_window.gated_candidate_transformer.train \
  --data "$DATA_ROOT/train" \
  --validation-data "$DATA_ROOT/validation" \
  --output-dir "$TRANSFORMER_ROOT/model_window8" \
  --window-length 8 \
  --model-dim 128 \
  --heads 4 \
  --cross-layers 2 \
  --attention-future-lag 4 \
  --translation-fixed-lag-layers 1 \
  --velocity-weight 0.1 \
  --acceleration-weight 0.05 \
  --jerk-weight 0.02 \
  --epochs 120 \
  --batch-size 24 \
  --learning-rate 2e-4 \
  --patience 12 \
  --device cuda
```

Training writes `best_pose.ckpt`, selected by the normalized RMSE of the
actually selected, fixed-lag-refined validation pose. It also writes
`best_loss.ckpt`, the backward-compatible `best.ckpt` loss alias, `last.ckpt`,
`history.csv`, and `run_report.json`. Start end-to-end evaluation with
`best_pose.ckpt`, but compare it with `best_loss.ckpt` on the untouched test set.

## Infer

The input must be a candidate bundle made with the same candidate exporter and feature schema used for training.

```bash
export BENCH_ROOT="$PWD/gigaPose_datasets/results/rgb_self_recovery_benchmark_20260803"
export TRANSFORMER_ROOT="$PWD/gigaPose_datasets/results/gated_candidate_transformer"

python3 -m tracking.rgb_self_recovery.sliding_window.gated_candidate_transformer.infer \
  --data "$BENCH_ROOT/gru_candidates" \
  --checkpoint "$TRANSFORMER_ROOT/model_window8/best_pose.ckpt" \
  --output-dir "$TRANSFORMER_ROOT/benchmark_window8_fixed_lag" \
  --fixed-lag 4 \
  --device cuda \
  --overwrite
```

The output directory contains `tracked_predictions.csv`, `diagnostics.jsonl`, and `report.json`, and can be passed to the existing `evaluate_backbones` command as another model.

Version-5 checkpoints contain the learned translation fixed-lag branch and
record the training-time temporal attention lag. Earlier checkpoints must be retrained; silently
initializing a missing trust head or changing attention visibility would be
unsafe. Inference requires `--fixed-lag` to equal the checkpoint lag. This
prevents a model trained with future context from silently running with a
different attention horizon. Use `--attention-future-lag 0` during training and
`--fixed-lag 0` during inference for a separate causal, zero-lookahead model.

## Dataset policy

The existing 8,698-frame training bundle is enough for a first controlled experiment, but a 128-dimensional transformer will benefit from more independent runs. Reusing the current 5,165-frame benchmark for training is valid only after creating a new, untouched benchmark. Once used for training, the old benchmark must not be reported as test performance.

After making a replacement benchmark, append the old benchmark with another `--data` argument:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gated_candidate_transformer.train \
  --data "$DATA_ROOT/train" \
  --data "$PWD/gigaPose_datasets/results/rgb_self_recovery_benchmark_20260803/gru_candidates" \
  --validation-data "$DATA_ROOT/validation" \
  --output-dir "$TRANSFORMER_ROOT/model_window8_expanded" \
  --window-length 8 --epochs 120 --batch-size 24 --device cuda
```

The loader rejects overlapping physical `source_run` identifiers across training and validation by default. A new benchmark should contain unseen physical runs, both cameras, all distance bins, and difficult conditions such as long range, weak masks, and orientation flips.
