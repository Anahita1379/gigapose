# Frozen visual-fusion candidate transformer

This experiment compares two lag-masked transformers with identical candidate
sets, windows, losses, and decoding:

- **Numeric:** 23 candidate features + 8 frame features.
- **Visual fusion:** the same numeric features plus cached, frozen evidence from
  the trained DINO matching U-Net.

The cached frame vector pools the observed RGB/mask CNN representation and
frozen DINO representation. The candidate vector is the exact pre-head
RGB/mask/CAD matching representation used by the DINO matching U-Net. The
transformer learns only small projection/gating layers and the existing
transformer; neither DINO nor the matching U-Net is updated.

With the default width-24 DINO matching U-Net and 128-dimensional transformer,
the cached vectors are 384-dimensional per frame and 432-dimensional per
candidate. With the end-to-end translation fixed-lag branch, the numeric
transformer has 697,169 trainable parameters; visual projection/gating raises
the visual-fusion transformer to 901,713 trainable
parameters. The frozen DINO/matching-U-Net parameters are not duplicated in the
transformer checkpoint because their outputs are cached in the bundle.

Both models use the same lag-aware translation refinement trained with final
position, velocity, acceleration, and jerk objectives. They also retain the
adaptive fixed-lag orientation guard. Independent trust/fallback heads remain
available for ablations but are disabled by default.

## 1. Variables

```bash
cd /home/anahita/gigapose

export RAW_TRAIN="$PWD/gigaPose_datasets/datasets/Assettocorsa_new_dataset_distance_bin_pose_regenerated"
export RAW_TEST="$PWD/gigaPose_datasets/datasets/Assettocorsa_new_dataset_benchmark_distance_bin_pose_regenerated"
export TRAIN_NAME="rgb_recovery_transformer_train_pose_regenerated"
export VALIDATION_NAME="rgb_recovery_transformer_validation_pose_regenerated"
export TEST_NAME="rgb_recovery_transformer_test_pose_regenerated"
export TRAIN_DATASET="$PWD/gigaPose_datasets/datasets/$TRAIN_NAME"
export VALIDATION_DATASET="$PWD/gigaPose_datasets/datasets/$VALIDATION_NAME"
export TEST_DATASET="$PWD/gigaPose_datasets/datasets/$TEST_NAME"
export DINO_MATCH_CKPT="$PWD/gigaPose_datasets/results/rgb_self_recovery_backbone_comparison/dino_matching_unet/best_pose.ckpt"
export BUNDLES="$PWD/gigaPose_datasets/results/transformer_visual_fusion_bundles"
export MODELS="$PWD/gigaPose_datasets/results/transformer_visual_fusion_models"
export RUNS="$PWD/gigaPose_datasets/results/transformer_visual_fusion_test"
```

Use physically separate training and validation runs. Do not split neighboring
frames from one run across train and validation.

## 2. Prepare corrected-pose train, validation, and test datasets

The preparation command writes a BOP/WebDataset `test` split for each role;
`train` and `validation` below describe experimental roles, not directory names.

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.prepare_assetto_dataset \
  --assetto-export-root "$RAW_TRAIN" \
  --role train \
  --include-runs all \
  --dataset-name "$TRAIN_NAME" \
  --distance-bins distance0_20,distance20_40,distance40_60,distance60_80,distance80_100,distance100_120 \
  --cameras front,rear \
  --maximum-depth-m 120 \
  --expected-cleaned-width 1548 \
  --overwrite

python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.prepare_assetto_dataset \
  --assetto-export-root "$RAW_TEST" \
  --role validation \
  --include-runs 20260803_laguna2026_fog_1opp_3laps_front,20260803_laguna2026_clear_1opp_3laps_rear \
  --dataset-name "$VALIDATION_NAME" \
  --distance-bins distance0_20,distance20_40,distance40_60,distance60_80,distance80_100,distance100_120 \
  --cameras front,rear \
  --maximum-depth-m 120 \
  --validation-frames-per-run 1000 \
  --expected-cleaned-width 1548 \
  --overwrite


# need to update this one later, right now it is rear light
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.prepare_assetto_dataset \
  --assetto-export-root "$RAW_TEST" \
  --role benchmark \
  --include-runs 20260803_laguna2026_clear_1opp_3laps_farFront,20260803_putnam_clear_1opp_2laps_farRear,20260803_laguna2026_fog_1opp_3laps_front,20260803_laguna2026_clear_1opp_3laps_rear  \
  --dataset-name "$TEST_NAME" \
  --distance-bins distance0_20,distance20_40,distance40_60,distance60_80,distance80_100,distance100_120,distance120_140 \
  --cameras front,rear \
  --maximum-depth-m 140 \
  --expected-cleaned-width 1548 \
  --overwrite
```

This revised split uses all 15 July runs for training, including the two runs
previously used for validation. Validation uses 1,000 frames from one
independent August front run and 1,000 frames from one independent August rear
run. Selection favors the largest consecutive segments: front is one continuous
block, while rear uses four long continuous blocks because no single 1,000-frame
rear block exists. The remaining far-front and Putnam far-rear runs stay
untouched for final testing, including the 120–140 m OOD range. No physical run
appears in more than one split.

Run GigaPose on all three:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.run_gigapose \
  --dataset-name "$TRAIN_NAME" --run-name gigapose_transformer_train_pose_regenerated \
  --batch-size 16 --num-workers 2 --devices 0

python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.run_gigapose \
  --dataset-name "$VALIDATION_NAME" --run-name gigapose_transformer_validation_pose_regenerated \
  --batch-size 16 --num-workers 2 --devices 0

python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.run_gigapose \
  --dataset-name "$TEST_NAME" --run-name gigapose_transformer_test_pose_regenerated \
  --batch-size 16 --num-workers 2 --devices 0

export GP_TRAIN="$PWD/gigaPose_datasets/results/gigapose_transformer_train_pose_regenerated/predictions/large-pbrreal-rgb-mmodel_${TRAIN_NAME}-test_gigapose_transformer_train_pose_regeneratedMultiHypothesis.csv"
export GP_VALIDATION="$PWD/gigaPose_datasets/results/gigapose_transformer_validation_pose_regenerated/predictions/large-pbrreal-rgb-mmodel_${VALIDATION_NAME}-test_gigapose_transformer_validation_pose_regeneratedMultiHypothesis.csv"
export GP_TEST="$PWD/gigaPose_datasets/results/gigapose_transformer_test_pose_regenerated/predictions/large-pbrreal-rgb-mmodel_${TEST_NAME}-test_gigapose_transformer_test_pose_regeneratedMultiHypothesis.csv"
```

## 3. Export identical candidate bundles

Run once for each split. These commands use the frozen DINO matching U-Net to
produce the candidate poses and the same 23/8 numeric features used by both
transformers.

```bash
for ROLE in train validation test; do
  case "$ROLE" in
    train) DATASET="$TRAIN_DATASET"; PREDICTIONS="$GP_TRAIN" ;;
    validation) DATASET="$VALIDATION_DATASET"; PREDICTIONS="$GP_VALIDATION" ;;
    test) DATASET="$TEST_DATASET"; PREDICTIONS="$GP_TEST" ;;
  esac

  python3 -m tracking.rgb_self_recovery.sliding_window.gru_selector.export_candidates \
    --dataset-dir "$DATASET" \
    --split test \
    --predictions "$PREDICTIONS" \
    --checkpoint "$DINO_MATCH_CKPT" \
    --output-dir "$BUNDLES/$ROLE" \
    --saved-candidates 16 \
    --top-k-gigapose 5 \
    --device cuda \
    --overwrite
done
```


## 4. Cache frozen visual embeddings

```bash
for ROLE in train validation test; do
  case "$ROLE" in
    train) DATASET="$TRAIN_DATASET" ;;
    validation) DATASET="$VALIDATION_DATASET" ;;
    test) DATASET="$TEST_DATASET" ;;
  esac
  python3 -m tracking.rgb_self_recovery.sliding_window.frozen_visual_fusion_transformer.prepare_visual_features \
    --data "$BUNDLES/$ROLE" \
    --dataset-dir "$DATASET" \
    --split test \
    --checkpoint "$DINO_MATCH_CKPT" \
    --device cuda \
    --overwrite
done
```


for ROLE in test; do
  case "$ROLE" in
    test) DATASET="$TEST_DATASET" ;;
  esac
  python3 -m tracking.rgb_self_recovery.sliding_window.frozen_visual_fusion_transformer.prepare_visual_features \
    --data "$BUNDLES/$ROLE" \
    --dataset-dir "$DATASET" \
    --split test \
    --checkpoint "$DINO_MATCH_CKPT" \
    --device cuda \
    --overwrite
done




Each bundle now contains `visual_features.npz` and `visual_manifest.json`.
Float16 is used only for cached storage; training converts it to float32.

## 5. Train the numeric transformer

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gated_candidate_transformer.train \
  --data "$BUNDLES/train" \
  --validation-data "$BUNDLES/validation" \
  --output-dir "$MODELS/numeric" \
  --window-length 8 \
  --attention-future-lag 4 \
  --model-dim 128 \
  --cross-layers 2 \
  --translation-fixed-lag-layers 1 \
  --velocity-weight 0.1 \
  --acceleration-weight 0.05 \
  --jerk-weight 0.02 \
  --epochs 120 \
  --patience 12 \
  --batch-size 24 \
  --device cuda \
  --overwrite
```

## 6. Train the frozen visual-fusion transformer

Use the same options and seed:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.frozen_visual_fusion_transformer.train \
  --data "$BUNDLES/train" \
  --validation-data "$BUNDLES/validation" \
  --output-dir "$MODELS/frozen_visual" \
  --window-length 8 \
  --attention-future-lag 4 \
  --model-dim 128 \
  --cross-layers 2 \
  --translation-fixed-lag-layers 1 \
  --velocity-weight 0.1 \
  --acceleration-weight 0.05 \
  --jerk-weight 0.02 \
  --epochs 120 \
  --patience 12 \
  --batch-size 24 \
  --device cuda \
  --overwrite
```

## 7. Run both on the untouched test bundle

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.gated_candidate_transformer.infer \
  --data "$BUNDLES/test" \
  --checkpoint "$MODELS/numeric/best_pose.ckpt" \
  --output-dir "$RUNS/numeric" \
  --fixed-lag 4 \
  --device cuda \
  --overwrite

python3 -m tracking.rgb_self_recovery.sliding_window.frozen_visual_fusion_transformer.infer \
  --data "$BUNDLES/test" \
  --checkpoint "$MODELS/frozen_visual/best_pose.ckpt" \
  --output-dir "$RUNS/frozen_visual" \
  --fixed-lag 4 \
  --device cuda \
  --overwrite
```

## 8. Evaluate

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.frozen_visual_fusion_transformer.evaluate \
  --dataset-dir "$TEST_DATASET" \
  --split test \
  --data "$BUNDLES/test" \
  --gigapose-predictions "$GP_TEST" \
  --numeric-output "$RUNS/numeric" \
  --visual-output "$RUNS/frozen_visual" \
  --output-dir "$RUNS/comparison" \
  --distance-bins-m 0,20,40,60,80,100,120,140 \
  --max-overlays 120 \
  --overwrite
```

`comparison/pose_metrics/summary.json` contains count, mean, RMSE, median, and
p90 for translation, rotation, and ADD, along with improvements over GigaPose
and distance bins. It also contains the CAD-overlay count and paired results.

`comparison/diagnostic_summary.json` contains translation/rotation trust Brier
score, ECE, fallback precision/recall/F1, front/rear-flip recovery, weather
groups, and true observed-mask-area groups. `trust_calibration.png` plots
predicted trust against observed held-out improvement.

Ground truth is used only by training targets and this held-out evaluator. It is
not read by transformer inference or fallback decisions.



```bash 
cd /home/anahita/gigapose

export TEST_DATASET="$PWD/gigaPose_datasets/datasets/rgb_recovery_transformer_test_pose_regenerated"
export GP_TEST="$PWD/gigaPose_datasets/results/gigapose_transformer_test_pose_regenerated/predictions/large-pbrreal-rgb-mmodel_rgb_recovery_transformer_test_pose_regenerated-test_gigapose_transformer_test_pose_regeneratedMultiHypothesis.csv"

export BUNDLES="$PWD/gigaPose_datasets/results/transformer_visual_fusion_bundles"
export RUNS="$PWD/gigaPose_datasets/results/transformer_visual_fusion_test"

export BENCH_ROOT="$PWD/gigaPose_datasets/results/rgb_self_recovery_benchmark_pose_regenerated_20260803"
export SCIPY_GRAPH="$BENCH_ROOT/sequence_factor_graph"

export COMPARISON="$RUNS/all_models_comparison_with_gtsam"
export GTSAM_GRAPH="$BENCH_ROOT/sequence_factor_graph_gtsam"
```

Then run:
```bash
python3 -m tracking.rgb_self_recovery.sliding_window.evaluate_backbones \
  --dataset-dir "$TEST_DATASET" \
  --split test \
  --model gigapose="$GP_TEST" \
  --model cascade_translation_only="$BENCH_ROOT/cascade_translation_only" \
  --model cascade_full_pose="$BENCH_ROOT/cascade_full_pose" \
  --model numeric_transformer="$RUNS/numeric" \
  --model frozen_visual_transformer="$RUNS/frozen_visual" \
  --model graph_full_cascade_prior="$SCIPY_GRAPH/cascade_prior_plus_candidates" \
  --model graph_translation_cascade_prior="$SCIPY_GRAPH/translation_cascade_prior_outer5" \
  --model gtsam_translation_prior="$GTSAM_GRAPH/translation_prior" \
  --model gtsam_full_pose_prior="$GTSAM_GRAPH/full_pose_prior" \
  --baseline cascade_full_pose \
  --prediction-translation-unit mm \
  --mesh "$TEST_DATASET/models/obj_000001.ply" \
  --distance-bins-m 0,20,40,60,80,100,120,140 \
  --max-overlays 160 \
  --output-dir "$COMPARISON" \
  --overwrite
```
| Model | Translation mean | Translation RMSE | Rotation mean | Rotation RMSE | ADD mean | ADD RMSE |
|---|---:|---:|---:|---:|---:|---:|
| GigaPose | 9.703 m | 16.914 m | 43.68° | 71.18° | 9.805 m | 16.942 m |
| Translation cascade | 2.423 m | 4.406 m | 14.11° | 33.01° | 2.470 m | 4.429 m |
| Full-pose cascade | 2.720 m | 5.796 m | 12.75° | 32.20° | 2.761 m | 5.814 m |
| Numeric transformer | 3.994 m | 9.704 m | 13.44° | 31.83° | 4.033 m | 9.724 m |
| Frozen visual transformer | 3.508 m | 9.544 m | **10.83°** | **27.48°** | 3.538 m | 9.560 m |
| Full-cascade + graph | 2.528 m | 4.819 m | 12.01° | 30.02° | 2.564 m | 4.837 m |
| Translation-cascade + graph | **2.240 m** | **3.674 m** | 11.45° | 29.76° | **2.276 m** | **3.695 m** |



## 1. Visual transformer + SciPy graph
```bash
cd /home/anahita/gigapose

export BUNDLES="$PWD/gigaPose_datasets/results/transformer_visual_fusion_bundles"
export RUNS="$PWD/gigaPose_datasets/results/transformer_visual_fusion_test"

export VISUAL_TRANSFORMER="$RUNS/frozen_visual/tracked_predictions.csv"
export VISUAL_GRAPH="$RUNS/frozen_visual_plus_graph"

python3 -m tracking.rgb_self_recovery.sliding_window.sequence_factor_graph.run \
  --data "$BUNDLES/test" \
  --mode prior \
  --prior-predictions "$VISUAL_TRANSFORMER" \
  --prior-translation-unit mm \
  --output-dir "$VISUAL_GRAPH/scipy" \
  --outer-iterations 5 \
  --maximum-nfev 100 \
  --overwrite

```
## 2. Visual transformer + GTSAM graph
```bash

python3 -m tracking.rgb_self_recovery.sliding_window.sequence_factor_graph_gtsam.run \
  --data "$BUNDLES/test" \
  --mode prior \
  --prior-predictions "$VISUAL_TRANSFORMER" \
  --prior-translation-unit mm \
  --prior-components full \
  --output-dir "$VISUAL_GRAPH/gtsam_full_prior" \
  --outer-iterations 5 \
  --maximum-iterations 100 \
  --overwrite

  ```

  I would also run this ablation:
  ```bash
  python3 -m tracking.rgb_self_recovery.sliding_window.sequence_factor_graph_gtsam.run \
  --data "$BUNDLES/test" \
  --mode prior \
  --prior-predictions "$VISUAL_TRANSFORMER" \
  --prior-translation-unit mm \
  --prior-components translation \
  --output-dir "$VISUAL_GRAPH/gtsam_translation_prior" \
  --outer-iterations 5 \
  --maximum-iterations 100 \
  --overwrite

```
That will tell us whether preserving the transformer’s rotation as a graph factor helps, or whether it is better used only as initialization.

## 3. Evaluate all three visual versions
```bash 
export TEST_DATASET="$PWD/gigaPose_datasets/datasets/rgb_recovery_transformer_test_pose_regenerated"
export GP_TEST="$PWD/gigaPose_datasets/results/gigapose_transformer_test_pose_regenerated/predictions/large-pbrreal-rgb-mmodel_rgb_recovery_transformer_test_pose_regenerated-test_gigapose_transformer_test_pose_regeneratedMultiHypothesis.csv"

python3 -m tracking.rgb_self_recovery.sliding_window.evaluate_backbones \
  --dataset-dir "$TEST_DATASET" \
  --split test \
  --model gigapose="$GP_TEST" \
  --model frozen_visual_transformer="$RUNS/frozen_visual" \
  --model visual_plus_scipy_graph="$VISUAL_GRAPH/scipy" \
  --model visual_plus_gtsam_full="$VISUAL_GRAPH/gtsam_full_prior" \
  --model visual_plus_gtsam_translation="$VISUAL_GRAPH/gtsam_translation_prior" \
  --baseline frozen_visual_transformer \
  --prediction-translation-unit mm \
  --mesh "$TEST_DATASET/models/obj_000001.ply" \
  --distance-bins-m 0,20,40,60,80,100,120,140 \
  --max-overlays 160 \
  --output-dir "$VISUAL_GRAPH/comparison" \
  --overwrite

```

Using the visual transformer as the baseline makes the paired statistics directly answer:
Did the graph reduce its catastrophic translation errors?
How often did the graph improve/worsen individual frames?
Did it preserve the transformer’s strong rotation?
Did long-range RMSE improve?
Did it damage the excellent 0–60 m performance?
The ideal outcome would be:
Approximately unchanged visual-transformer performance below 60 m
Translation error closer to the translation-cascade graph beyond 60 m
Rotation remaining near the visual transformer’s 10.83° overall mean
A substantial reduction from its 9.54 m translation RMSE