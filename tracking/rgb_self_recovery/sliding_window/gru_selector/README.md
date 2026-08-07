# Causal GRU candidate selector

This is an isolated experiment. It does not modify the CNN, DINO, matching
U-Net, teacher/student, or existing sliding-window code.

The GRU does **not** regress an unconstrained absolute pose. At every frame it
chooses among a fixed candidate set:

1. candidate 0 is the original, unmodified rank-0 GigaPose pose;
2. the remaining candidates come from the existing RGB/mask/CAD recovery
   network and its broad translation, depth, rotation, and optional 180-degree
   flip hypotheses;
3. a causal GRU uses prior frames plus current candidate evidence to select one
   candidate;
4. inference can abstain and return candidate 0 when recovery probability or
   its margin over GigaPose is too small.

Ground truth is used only while exporting training labels and reporting test
errors. It is not an input feature to the GRU.

## Sequence guarantees

Sequences are identified by `source_run + camera_id`, sorted by
`source_frame`/timestamp, and split at camera/run changes, scene changes,
nonmonotonic ordering, missing frames, or excessive timestamp gaps. Training
clips never cross a segment. The hidden state is cleared at every segment
boundary during replay.

Train and validation bundles must use different complete `source_run` values.
Training rejects run overlap by default. The previously sampled 519-per-bin
data can only be used if its `frame_map.json` retains correct source metadata;
each omitted frame becomes a segment boundary, so it cannot falsely form a
continuous clip.

## Assetto sequence dataset used here

The GRU experiment uses the already shade-cleaned distance-bin export:

```bash
export ASSETTO_SOURCE="/mnt/ssd2tb/.local_share_backup/Steam/steamapps/common/assettocorsa/apps/lua/multi_cam_obs/frames/Assettocorsa_new_dataset_distance_bin_cleaned"
```

Only the 0--120 m bins are included. The split is by complete physical run:

- train: 13 runs, currently 8,698 valid frames in 14 run/camera sequences;
- validation: 164 Laguna fog/front plus 164 Putnam rain/rear frames, giving
  equal front/rear frame counts;
- final evaluation: not assigned yet; use a new complete physical run.

The preparer decodes the exact packed `instance_id`, requires at least 64 mask
pixels, rejects invalid camera depth, and requires the already-cleaned width of
1548 pixels. It never crops these images a second time.

Prepare all three datasets:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.prepare_assetto_dataset \
  --assetto-export-root "$ASSETTO_SOURCE" --role train --overwrite

python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.prepare_assetto_dataset \
  --assetto-export-root "$ASSETTO_SOURCE" --role validation --overwrite

```

Validate them:

```bash
for name in \
  rgb_recovery_gru_train_assetto \
  rgb_recovery_gru_validation_laguna_putnam
do
  python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.validate_assetto_dataset \
    --dataset-dir "$PWD/gigaPose_datasets/datasets/$name" \
    --output "$PWD/gigaPose_datasets/datasets/$name/validation_report.json"
done
```

Run the existing Assetto residual GigaPose checkpoint on each dataset:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.run_gigapose \
  --dataset-name rgb_recovery_gru_train_assetto \
  --devices 0 --batch-size 16 --num-workers 2

python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.run_gigapose \
  --dataset-name rgb_recovery_gru_validation_laguna_putnam \
  --devices 0 --batch-size 16 --num-workers 2
```

These commands write the GigaPose results below
`gigaPose_datasets/results/gigapose_rgb_recovery_gru_*`.

## 1. Choose the fixed per-frame candidate model

Use one already-trained RGB self-recovery checkpoint. For example:

```bash
export CANDIDATE_CKPT="$PWD/gigaPose_datasets/results/rgb_self_recovery_backbone_comparison/dino_matching_unet/best_pose.ckpt"
```

Pooled DINO, matching U-Net, DINO matching U-Net, and depth-privileged student
checkpoints are also accepted. Use the **same checkpoint and export arguments**
for train, validation, and test bundles. GRU training does not update this
candidate model.

## 2. Export train and validation candidate bundles

First run GigaPose on each full sequence split. Then set these paths:

```bash
export GRU_ROOT="$PWD/gigaPose_datasets/results/rgb_self_recovery_gru"
export TRAIN_DATASET="$PWD/gigaPose_datasets/datasets/rgb_recovery_gru_train_assetto"
export VAL_DATASET="$PWD/gigaPose_datasets/datasets/rgb_recovery_gru_validation_laguna_putnam"
export TRAIN_GP="$PWD/gigaPose_datasets/results/gigapose_rgb_recovery_gru_train_assetto"
export VAL_GP="$PWD/gigaPose_datasets/results/gigapose_rgb_recovery_gru_validation_laguna_putnam"
```

Export candidates and oracle labels:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.export_candidates \
  --dataset-dir "$TRAIN_DATASET" \
  --split test \
  --predictions "$TRAIN_GP" \
  --checkpoint "$CANDIDATE_CKPT" \
  --mesh "$TRAIN_DATASET/models/obj_000001.ply" \
  --saved-candidates 16 \
  --sequence-max-frame-gap 1 \
  --sequence-max-time-gap-s 0.5 \
  --output-dir "$GRU_ROOT/data/train" \
  --device cuda \
  --overwrite

python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.export_candidates \
  --dataset-dir "$VAL_DATASET" \
  --split test \
  --predictions "$VAL_GP" \
  --checkpoint "$CANDIDATE_CKPT" \
  --mesh "$VAL_DATASET/models/obj_000001.ply" \
  --saved-candidates 16 \
  --sequence-max-frame-gap 1 \
  --sequence-max-time-gap-s 0.5 \
  --output-dir "$GRU_ROOT/data/validation" \
  --device cuda \
  --overwrite
```

Inspect each `manifest.json`. In particular,
`oracle_nonbaseline_fraction` tells you how often the exported candidate pool
contains a better option than raw GigaPose. If it is almost zero, a GRU cannot
produce useful recovery no matter how well it trains.

## 3. Train the GRU

An eight-frame training clip gives the GRU more context than the five-frame
selector while remaining causal. This can be changed with `--clip-length 5`.

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.train \
  --data "$GRU_ROOT/data/train" \
  --validation-data "$GRU_ROOT/data/validation" \
  --output-dir "$GRU_ROOT/models/cnn_gru" \
  --clip-length 8 \
  --clip-stride 1 \
  --validation-stride 8 \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 2e-4 \
  --patience 10 \
  --min-delta 1e-4 \
  --device cuda \
  --overwrite
```

`best.ckpt` is selected by validation objective. `last.ckpt`, `history.csv`,
and `run_report.json` are also saved. Plot progress with:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.plot_history \
  --run-dir "$GRU_ROOT/models/cnn_gru"
```

## 4. Export and run a held-out test sequence

Export the test candidate bundle exactly as above, changing the dataset,
GigaPose CSV, and output to `$GRU_ROOT/data/test`. Then replay the causal GRU:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.select \
  --data "$GRU_ROOT/data/test" \
  --checkpoint "$GRU_ROOT/models/cnn_gru/best.ckpt" \
  --minimum-recovery-probability 0.45 \
  --minimum-recovery-margin-over-gigapose 0.05 \
  --output-dir "$GRU_ROOT/runs/test_cnn_gru" \
  --device cuda \
  --overwrite
```

The output directory contains a BOP-compatible `tracked_predictions.csv`, a
per-frame `selection_diagnostics.csv`, and `run_report.json`. Candidate 0 is an
explicit GigaPose fallback; the two thresholds above control abstention.

Use the existing evaluator and overlays by supplying that output directory as
a model:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.evaluate_backbones \
  --dataset-dir "$TEST_DATASET" \
  --split test \
  --model gigapose="$TEST_GP" \
  --model gru="$GRU_ROOT/runs/test_cnn_gru" \
  --baseline gigapose \
  --mesh "$TEST_DATASET/models/obj_000001.ply" \
  --distance-bins-m 0,20,40,60,80,100,120 \
  --max-overlays 164 \
  --output-dir "$GRU_ROOT/comparison/test_cnn_gru" \
  --overwrite
```

This first implementation intentionally uses an offline two-stage workflow:
candidate extraction, then GRU selection. It makes training reproducible and
allows repeated GRU experiments without rerunning CAD rendering. The selector
itself is causal and resets correctly, so the same model can later be fused
with online candidate generation without changing its learned inputs.

## Temporal orientation and flip hysteresis

Long-distance front/rear ambiguity is handled without an auxiliary visual
orientation head. Enable the causal temporal-orientation layer when replaying
an existing selector checkpoint:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.select \
  --data "$GRU_ROOT/data/validation" \
  --checkpoint "$GRU_ROOT/models/dino_matching_gru/best.ckpt" \
  --temporal-orientation \
  --orientation-soft-start-deg 45 \
  --orientation-hard-limit-deg 90 \
  --flip-min-angle-deg 135 \
  --flip-confirmation-frames 2 \
  --stable-orientation-frames 4 \
  --stable-flip-confirmation-frames 5 \
  --orientation-fixed-lag \
  --flip-min-probability 0.15 \
  --flip-min-margin 0.03 \
  --maximum-angular-prediction-deg 30 \
  --minimum-recovery-probability 0 \
  --minimum-recovery-margin-over-gigapose 0 \
  --output-dir "$GRU_ROOT/runs/putnam_laguna_temporal_orientation_gru" \
  --device cuda \
  --overwrite
```

The layer uses only prior accepted predictions and timestamps. It never uses
ground truth. It maintains previous and constant-angular-velocity orientations,
constructs temporal offsets at plus/minus 10 and 20 degrees plus an explicit
opposite-mode hypothesis, softly penalizes candidates beyond 45 degrees, and
holds candidates beyond 90 degrees. A front/rear transition beyond 135 degrees
must remain visually preferred and orientation-consistent before the persistent
flip state changes. A provisional orientation uses two-frame confirmation. Once
four ordinary accepted frames establish a stable orientation state, a flip
requires five consistent frames. Ordinary candidates in the established mode
cancel a pending transition. After a confirmed transition, stability resets and
is rebuilt. While a transition is pending, translation is retained from the
GRU-selected recovery candidate and orientation falls back to the
constant-angular-velocity prediction.

`--orientation-fixed-lag` resolves the beginning of a genuine transition as
well as its confirming frame. Pending rows are buffered until hysteresis makes
a decision. If the transition is confirmed, every earlier pending row is
backfilled with that row's mutually consistent visual-candidate orientation;
its GRU-selected translation is unchanged. If ordinary same-mode evidence
cancels the transition, the pending rows keep their safe temporal-fallback
orientations. In the stable state this adds at most four frames of latency for
the default five-frame confirmation. Sequence resets and inconsistent flip
hypotheses discard the unresolved buffer, so rows are never revised across a
run, camera, or temporal discontinuity. The replay command is offline, so its
CSV is written only after all such decisions have been resolved.

Legacy behavior remains available by omitting `--temporal-orientation`. No
candidate re-export or GRU retraining is needed because this is a causal
post-selection orientation layer. `selection_diagnostics.csv` records temporal
reranking, fallback, pending/confirmed transitions, flip state, and generated
temporal candidate names for every frame. With fixed lag enabled it also records
`fixed_lag_backfilled` and `resolved_flip_state`; `run_report.json` reports the
number of revised frames.
