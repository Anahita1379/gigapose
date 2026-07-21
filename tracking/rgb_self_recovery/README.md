# RGB-only self-recovering 6D pose tracker

This is an isolated experimental replacement for the scalar recovery path in
the existing tracker. Everything in this implementation lives under
`tracking/rgb_self_recovery/`; running it does not alter `tracking.run_tracking`
or its checkpoints.

The purpose is to recover when an initial GigaPose pose is wrong, rather than
only smooth that pose. It uses real image evidence, CAD geometry, and a broad
pose search. It never reads sensor depth. CAD-rendered depth is used internally
to encode the candidate's own geometry and is available in real deployment
because it comes from the mesh, not from a depth camera.

## What is implemented

The package contains four independent stages:

1. `generate_dataset.py` creates grouped training examples from synthetic
   Assetto Corsa ground truth. Each group contains the observed RGB/instance
   mask and several candidate CAD poses, including moderate errors, large
   errors, and a 180-degree flip.
2. `train.py` trains a render-and-compare verifier/refiner. It predicts a 2D
   center correction, log-depth correction, full SO(3) correction, candidate
   confidence, and continuous candidate quality.
3. `inference.py` performs batched iterative correction and scoring for a set
   of hypotheses.
4. `run.py` is a standalone sequential tracker. It combines fresh GigaPose
   top-K poses, a temporal pose beam, robust constant-velocity propagation,
   optical flow, 180-degree flips, rotation grids, center offsets, and depth
   offsets. Low-confidence states trigger broad same-frame recovery. Broad
   checks also run at initialization and periodically, so the system does not
   have to wait for a later frame before attempting recovery.

For multiple cars, Hungarian association and appearance/mask evidence preserve
track identity. CAD scoring is occlusion-aware: pixels occupied by other
detected instances are removed from the candidate render before comparison.

## Model inputs and outputs

For each target detection, the image branch receives:

- an RGB crop;
- the observed target-instance mask.

For each candidate pose, the render branch receives five channels:

- candidate silhouette;
- three camera-space CAD-normal channels;
- one normalized relative CAD-depth channel.

The network predicts

\[
(\Delta u,\Delta v),\quad
\Delta\log z,\quad
\Delta\boldsymbol\omega\in\mathfrak{so}(3),\quad
p_{\mathrm{valid}},\quad q.
\]

The translation update is parameterized by the projected center and metric
depth:

\[
z' = z\exp(\Delta\log z),\qquad
\mathbf t'=z'K^{-1}[u+\Delta u,v+\Delta v,1]^T.
\]

The rotation update is a left camera-frame correction:

\[
R'=\operatorname{Exp}([\Delta\boldsymbol\omega]_\times)R.
\]

The exponential is the SO(3) exponential map, not an arbitrary scalar
exponential. It converts a three-component axis-angle tangent vector into a
valid orthonormal rotation matrix with determinant +1.

Residual regression is optimized only for candidates inside the configured
correction basin. All candidates—including very bad or flipped ones—train the
confidence, quality, and within-instance ranking losses. Consequently a large
error can be handled in either of two ways: refine a nearby candidate, or reject
the bad candidate and select a better hypothesis from the broad pool.

The loss is

\[
\mathcal L =
\lambda_c\mathcal L_{\mathrm{center}}+
\lambda_z\mathcal L_{\log z}+
\lambda_R\mathcal L_{R}+
\lambda_p\mathcal L_{\mathrm{BCE}}+
\lambda_q\mathcal L_{\mathrm{quality}}+
\lambda_{\mathrm{rank}}\mathcal L_{\mathrm{rank}}.
\]

Rotation error is evaluated with a stable `atan2` SO(3) geodesic angle, so a
near-180-degree disagreement does not collapse to a small error.

## 1. Generate training and validation data

Training data:

```bash
python -m tracking.rgb_self_recovery.generate_dataset \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_new_dataset \
  --split train_pbr_web_gsam_clean \
  --mesh gigaPose_datasets/datasets/assettocorsa_new_dataset/models/obj_000001.ply \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_data/train \
  --max-frames 10000 \
  --candidates-per-instance 12 \
  --overwrite
```

Validation data must be generated separately from the validation split:

```bash
python -m tracking.rgb_self_recovery.generate_dataset \
  --dataset-dir gigaPose_datasets/datasets/assettocorsa_new_dataset \
  --split val_pbr_web_gsam_clean \
  --mesh gigaPose_datasets/datasets/assettocorsa_new_dataset/models/obj_000001.ply \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_data/val \
  --max-frames 2000 \
  --candidates-per-instance 12 \
  --overwrite
```

Generation is CPU plus CAD rendering. It prints progress every 25 source
frames. Both manifests explicitly record `uses_observed_depth: false`.

## 2. Train the verifier/refiner

```bash
python -m tracking.rgb_self_recovery.train \
  --data gigaPose_datasets/results/rgb_self_recovery_data/train \
  --validation-data gigaPose_datasets/results/rgb_self_recovery_data/val \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_model \
  --epochs 200 \
  --batch-size 8 \
  --num-workers 4 \
  --learning-rate 2e-4 \
  --patience 20 \
  --device cuda \
  --logger wandb \
  --run-name assettocorsa_rgb_self_recovery \
  --wandb-project gigapose
```

`batch-size` counts car-instance groups. Each group contains all of its pose
candidates, so GPU memory is approximately proportional to
`batch_size * candidates_per_instance`. Start with 8; reduce it if memory is
tight. Increasing data-loader workers helps only until loading/augmentation is
no longer the bottleneck.

Outputs:

- `best.ckpt`: lowest full validation loss;
- `last.ckpt`: latest completed epoch;
- `run_report.json`: complete epoch history and configuration;
- W&B train/validation loss components, center error, log-depth error,
  rotation error, selected-candidate quality, and oracle quality.

## 3. Run RGB-only self-recovery on real sequential data

Start with a short smoke run and overlays:

```bash
python -m tracking.rgb_self_recovery.run \
  --predictions PATH/TO/GigaPoseMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/real_20260505v1_front_gsam_v4 \
  --split test \
  --checkpoint gigaPose_datasets/results/rgb_self_recovery_model/best.ckpt \
  --mesh gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
  --association-config tracking/configs/improved.json \
  --output-dir gigaPose_datasets/results/rgb_self_recovery_smoke \
  --device cuda \
  --max-frames 100 \
  --save-overlays \
  --overlay-every 1 \
  --overwrite
```

If the dataset contains its own matching model, omit `--mesh`; the script first
tries `<dataset-dir>/models/obj_000001.ply`.

For the full sequence, remove `--max-frames` and normally increase
`--overlay-every` to 10 or disable overlays. Important search controls are:

- `--top-k-gigapose 5`: fresh global hypotheses retained per detection;
- `--beam-size 4`: corrected temporal alternatives retained per track;
- `--global-interval 5`: periodic fresh/broad global safety check;
- `--max-candidates 48`: cap on the broad pool;
- `--refinement-iterations 2`: neural correction iterations per candidate;
- `--broad-recovery-confidence 0.55`: low-confidence broad-search trigger.

The default broad pool includes 180-degree flips, camera-axis rotation offsets,
yaw offsets, center offsets, and log-depth offsets. Raising the pool or the
iteration count improves search coverage but increases CAD rendering and neural
inference time nearly linearly.

Outputs:

- `tracked_predictions.csv`: BOP/GigaPose-compatible final poses in millimetres;
- `candidate_diagnostics.csv`: verifier confidence, quality, silhouette IoU,
  correction size, candidate source, and whether broad recovery was used;
- `overlays/`: red original GigaPose render, green recovered render, cyan target
  mask edge, and yellow detection box;
- `run_report.json`: data/checkpoint paths, state counts, broad-recovery count,
  elapsed time, and full arguments.

## Real-world use and limitations

This is the correct architecture for independent RGB correction, but training a
network does not guarantee that every wrong pose will be fixed. Recovery still
requires at least one candidate to enter the learned correction basin, which is
why broad search and a pose beam are essential. Severe occlusion, an incorrect
segmentation mask, car symmetry, a mismatched CAD, or a synthetic-to-real
appearance gap can still produce a confident wrong solution.

Before trusting output poses:

1. Compare red/green overlays on several real sequences.
2. Inspect recovery separately for front/rear symmetry, large translation
   errors, crossings, and occlusion.
3. Compare temporal jitter and disagreement with the original GigaPose output.
4. If trusted real camera-frame labels are available, prepare them in the same
   BOP/WebDataset format and fine-tune this verifier on them at a lower learning
   rate. Do not train it on unverified pseudo-labels, because the verifier will
   learn their systematic pose bias.

The small image encoder is deliberately self-contained. If synthetic-to-real
transfer remains the limiting factor, the next controlled extension should be
a frozen pretrained visual backbone plus this same geometry/search pipeline,
not another scalar-only residual head.
