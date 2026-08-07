# Five-frame RGB self-recovery without retraining

This package is isolated from the existing `tracking.rgb_self_recovery` code.
It loads the same checkpoint and uses the same RGB/mask-versus-CAD verifier,
but replaces frame-local winner selection with a causal five-frame candidate
path search followed by a small continuous pose optimization.

No existing tracker file is modified. Retraining the verifier is optional;
the window logic has no learned parameters.

## Window objective

For candidate index sequence \(c_{t-W+1:t}\), the selector minimizes

$$
E =
\lambda_u\sum_k E_{\mathrm{RGB}}(c_k)
+ E_{\mathrm{speed}}
+ E_{\mathrm{acceleration}}
+ E_{\mathrm{rotation}}
+ E_{\mathrm{angular\ acceleration}}.
$$

The default window has \(W=5\) and retains four neural candidates per frame,
so exhaustive path selection evaluates at most \(4^5=1024\) short paths.
Large residuals use a smooth robust penalty. After discrete path selection,
translations and SO(3) increments are jointly optimized while staying close
to the selected RGB/CAD poses.

This is causal: it uses the current frame and up to four previous frames. It
does not use future images and does not revise rows already written to the
output CSV.

Frames are treated as one physical sequence only when `source_run` and
`camera_id` match. They are ordered by `source_frame` (with timestamp and
scene/image IDs as deterministic fallbacks). By default the tracker clears its
tracks, candidate beam, optical-flow history, and window whenever source frames
are nonconsecutive or timestamps are more than 0.5 seconds apart. The next
frame therefore starts with broad same-frame recovery instead of inheriting an
invalid motion prior. Configure this with `--sequence-max-frame-gap` and
`--sequence-max-time-gap-s`; use `--no-sequence-aware` only for an ablation.

Set `--window-size 1` to remove only the added window optimizer while retaining
the original track, optical-flow, and candidate-beam history. This is the clean
original-sequential-tracker versus sliding-window ablation. For a genuinely
independent baseline, add `--per-frame`; this clears all temporal state before
every image, starts from fresh GigaPose, and runs broad same-frame recovery.

## Masks

Pass a direct mask directory with:

```bash
--mask-dir /path/to/masks \
--mask-pattern '{scene_id:06d}_{im_id:06d}.png'
```

A frame-level image may be binary for one car or integer-valued for multiple
instances. For one binary file per instance, include `detection_id` or
`external_id` in the pattern, for example:

```bash
--mask-pattern '{scene_id:06d}_{im_id:06d}_{detection_id:02d}.png'
```

Fallback choices are:

- `dataset` (default): retain the existing frame-map/RLE/bounding-box mask;
- `bbox`: explicitly use the detection rectangle;
- `render`: use the fresh rank-0 GigaPose CAD silhouette when no direct mask
  exists.

Rendered fallback is circular—the pose proposes its own observed mask—so use
it only when direct segmentation is unavailable. Direct instance masks are
preferred for training and real inference.

## Generate data for retraining

The generator first tries `--mask-dir`, then the dataset RLE mask, and finally
the GT CAD render when `--mask-fallback render` is selected:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.generate_dataset \
  --dataset-dir gigaPose_datasets/datasets/NEW_ASSETTO_CORSA \
  --split train_pbr_web_gsam_clean \
  --mesh gigaPose_datasets/datasets/NEW_ASSETTO_CORSA/models/obj_000001.ply \
  --mask-dir /path/to/train/masks \
  --mask-pattern '{key}_i{instance_index:02d}.png' \
  --mask-fallback render \
  --output-dir gigaPose_datasets/results/rgb_window_data/train \
  --candidates-per-instance 12 \
  --overwrite
```

Generate validation from a separate split and output directory. Train the
unchanged verifier through the local wrapper:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.train \
  --data gigaPose_datasets/results/rgb_window_data/train \
  --validation-data gigaPose_datasets/results/rgb_window_data/val \
  --output-dir gigaPose_datasets/results/rgb_window_model \
  --epochs 200 \
  --batch-size 8 \
  --device cuda
```

## Run

Start with a short one-car smoke test:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.run \
  --predictions PATH/GigaPoseMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/NEW_ASSETTO_CORSA \
  --split test \
  --checkpoint gigaPose_datasets/results/rgb_window_model/best.ckpt \
  --mesh gigaPose_datasets/datasets/NEW_ASSETTO_CORSA/models/obj_000001.ply \
  --mask-dir /path/to/test/masks \
  --mask-pattern '{scene_id:06d}_{im_id:06d}.png' \
  --mask-fallback dataset \
  --output-dir gigaPose_datasets/results/rgb_window_smoke \
  --window-size 5 \
  --sequence-aware \
  --sequence-max-frame-gap 1 \
  --sequence-max-time-gap-s 0.5 \
  --window-candidates 4 \
  --orientation-gate-mode soft \
  --state-iou-confidence-weight 0.5 \
  --max-frames 100 \
  --save-overlays \
  --overlay-every 1 \
  --device cuda \
  --overwrite
```

Outputs retain the original names: `tracked_predictions.csv`,
`candidate_diagnostics.csv`, `overlays/`, and `run_report.json`. Diagnostics
add the selected window rank and unary, speed, acceleration, rotation, and
angular-acceleration costs. Soft orientation mode keeps every candidate and
adds a robust orientation penalty to candidate ranking. The IoU confidence
floor affects only the normal/uncertain/lost tracker state; diagnostics retain
the unmodified network score as `verifier_confidence`. The IoU floor is disabled
when verifier confidence is below `0.05`.

The runner also scores unmodified rank-0 GigaPose as a safety baseline. A
recovery with verifier confidence below `0.05` that moves more than `1 m` or
`45 deg` from that baseline is rejected unless an existing trajectory supports
the change. The fallback pose is retained with mode `lost` and source suffix
`recovery_abstention_rank0`; it can seed later recovery without being reported
as a successful correction.

Defaults are starting scales, not learned physical noise. Tune them on a
separate validation sequence, particularly FPS, acceleration, and rotation
scales.

## Direct training from the distance-binned Assetto export

`generate_dataset` can read `Assettocorsa_new_dataset_copy` directly. It joins
`source_manifest.csv`, `camera_frames.csv`, `transforms.csv`, and
`bboxes_3d.csv`, so the 900 MB export does not need to be copied into another
WebDataset first. The adapter uses the packed `instance_id` masks and the
project's validated Assetto-to-OpenCV pose conversion.
If a later export has no saved masks, add `--mask-fallback render`; the adapter
will derive the training mask by rendering the ground-truth CAD pose.

Keep complete recording runs—not random neighboring frames—held out for
validation. Pass the identical held-out run list to both commands:

```bash
AC_EXPORT=/mnt/ssd2tb/.local_share_backup/Steam/steamapps/common/assettocorsa/apps/lua/multi_cam_obs/frames/Assettocorsa_new_dataset_copy
CAD=gigaPose_datasets/datasets/racecar/models/obj_000001.ply
RECOVERY_DATA=gigaPose_datasets/results/rgb_self_recovery_assetto_window_data

python3 -m tracking.rgb_self_recovery.sliding_window.generate_dataset \
  --assetto-export-root "$AC_EXPORT" \
  --assetto-split train \
  --assetto-validation-run 20260730_putnam_rain_1opp_2lap_farRear \
  --assetto-validation-run 20260730_laguna2026_fog_1opp_2laps_mostlyFront \
  --assetto-cameras front,rear \
  --assetto-distance-bins distance0_20,distance20_40,distance40_60,distance60_80,distance80_100,distance100_120 \
  --mesh "$CAD" \
  --cad-axis-convention x-forward-z-up \
  --fit-aabb nonuniform \
  --assetto-max-depth-m 150 \
  --assetto-frames-per-bin 529 \
  --assetto-short-bin-policy all \
  --candidates-per-instance 12 \
  --output-dir "$RECOVERY_DATA/train" \
  --overwrite

python3 -m tracking.rgb_self_recovery.sliding_window.generate_dataset \
  --assetto-export-root "$AC_EXPORT" \
  --assetto-split validation \
  --assetto-validation-run 20260730_putnam_rain_1opp_2lap_farRear \
  --assetto-validation-run 20260730_laguna2026_fog_1opp_2laps_mostlyFront \
  --assetto-cameras front,rear \
  --assetto-distance-bins distance0_20,distance20_40,distance40_60,distance60_80,distance80_100,distance100_120 \
  --mesh "$CAD" \
  --cad-axis-convention x-forward-z-up \
  --fit-aabb nonuniform \
  --assetto-max-depth-m 150 \
  --assetto-frames-per-bin 529 \
  --assetto-short-bin-policy all \
  --candidates-per-instance 12 \
  --output-dir "$RECOVERY_DATA/validation" \
  --overwrite

python3 -m tracking.rgb_self_recovery.sliding_window.train \
  --data "$RECOVERY_DATA/train" \
  --validation-data "$RECOVERY_DATA/validation" \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_assetto_window_model \
  --epochs 200 \
  --batch-size 8 \
  --anti-flip-training \
  --device cuda
```

For a quick pipeline check, add `--max-frames 50` to both generation commands
and train for two epochs. Remove that limit for the real run.

The network is still trained per frame: it learns the RGB/mask/CAD evidence and
pose correction. The five-frame optimizer itself is intentionally not a
learned network; it combines the retrained verifier's candidate scores with
velocity, acceleration, and rotation smoothness at inference time.

The balanced selector allocates each bin's target as evenly as available data
allows across recording runs, then takes one consecutive block per run/camera.
It never duplicates images. `--assetto-short-bin-policy error` requires exactly
the requested count and stops on a short bin; `all` keeps every unique image in
short bins.

Shaded-side removal is disabled by default. Enable the known 258-pixel strips
only for a dataset version that still contains them:

```bash
--assetto-crop-shaded-sides --assetto-shaded-side-pixels 258
```

The RGB verifier's training shards do not require GigaPose predictions because
the candidate failures are synthesized around simulator ground truth. Running
the trained sliding-window pipeline does require a GigaPose multi-hypothesis
CSV; its hypotheses seed the per-frame candidate pool that the verifier scores
and the temporal window reranks.

## Compare CNN and DINOv2 image backbones

All three variants require a training run on the same train/validation shards.
The baseline trains the original CNN. Frozen DINO trains the CNN, CAD encoder,
DINO projection/fusion, and output heads while keeping every DINO parameter
fixed. Last-block mode additionally trains DINO's final transformer block and
normalization with a lower learning rate.

```bash
DATA=gigaPose_datasets/results/rgb_self_recovery_assetto_window_data
MODELS=gigaPose_datasets/results/rgb_self_recovery_backbone_comparison

# 1. Existing CNN baseline (original implementation remains unchanged).
python3 -m tracking.rgb_self_recovery.sliding_window.train \
  --data "$DATA/train" \
  --validation-data "$DATA/validation" \
  --output-dir "$MODELS/cnn" \
  --epochs 200 --batch-size 8 --anti-flip-training --device cuda

# 2. Frozen DINOv2-S/14 plus the current CNN and CAD branches.
python3 -m tracking.rgb_self_recovery.sliding_window.train_dino \
  --dino-mode frozen \
  --dino-model dinov2_vits14 \
  --data "$DATA/train" \
  --validation-data "$DATA/validation" \
  --output-dir "$MODELS/dino_frozen" \
  --epochs 200 --batch-size 4 --anti-flip-training --device cuda

# 3. Continue from the best frozen pose checkpoint and tune DINO's final block.
python3 -m tracking.rgb_self_recovery.sliding_window.train_dino \
  --dino-mode last_block \
  --dino-model dinov2_vits14 \
  --initialize-from "$MODELS/dino_frozen/best_pose.ckpt" \
  --data "$DATA/train" \
  --validation-data "$DATA/validation" \
  --output-dir "$MODELS/dino_last_block" \
  --learning-rate 5e-5 \
  --dino-learning-rate 1e-5 \
  --epochs 60 --batch-size 4 --anti-flip-training --device cuda
```

Both isolated trainers save `best_loss.ckpt`, `best_rotation.ckpt`,
`best_center.ckpt`, `best_depth.ckpt`, and `best_pose.ckpt`, plus the backward-compatible
`best.ckpt` loss alias and `last.ckpt`. `best_pose.ckpt` minimizes an
equal-weight average of center, log-depth, and rotation errors after normalizing
by their configured correction ranges. Early stopping remains based on validation loss.
Use `best_pose.ckpt` as the first end-to-end candidate, but choose the deployed
checkpoint using held-out GigaPose translation/rotation results and overlays.

The sliding-window runner recognizes both original CNN checkpoints and the new
DINO checkpoint format through the same `--checkpoint` argument. DINO RGB
features are computed once per frame crop and reused across all candidate poses
and iterative recovery passes.

## Evaluate the three trained backbones

First run the sliding-window inference command three times on the same prepared
held-out sequence, changing only `--checkpoint` and `--output-dir`. Then compare
the resulting `tracked_predictions.csv` files against simulator ground truth:

```bash
python3 -m tracking.rgb_self_recovery.sliding_window.evaluate_backbones \
  --dataset-dir gigaPose_datasets/datasets/PREPARED_HELD_OUT_SEQUENCE \
  --split test \
  --model cnn=gigaPose_datasets/results/rgb_window_eval/cnn \
  --model dino_frozen=gigaPose_datasets/results/rgb_window_eval/dino_frozen \
  --model dino_last_block=gigaPose_datasets/results/rgb_window_eval/dino_last_block \
  --baseline cnn \
  --mesh gigaPose_datasets/datasets/PREPARED_HELD_OUT_SEQUENCE/models/obj_000001.ply \
  --distance-bins-m 0,20,40,60,80,100,120 \
  --max-overlays 120 \
  --output-dir gigaPose_datasets/results/rgb_window_eval/backbone_comparison \
  --overwrite
```

The evaluator writes `summary.json`, paired CNN-versus-DINO improvement and win
fractions, `per_frame_metrics.csv`, translation/rotation distance-bin plots,
and RGB/CAD overlays. Ground truth is green; each model receives a distinct
color and its frame-level translation and rotation errors are printed in the
overlay legend. The current protocol expects one ground-truth car per frame.
