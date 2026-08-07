# Optional map-aware five-frame RGB self-recovery

This isolated package extends `sliding_window` with track-map diagnostics and
soft candidate reranking. It can run today without map information; in that
case it behaves as the five-frame RGB window tracker and reports map status as
disabled or missing.

The neural verifier remains unchanged. Map constraints are runtime costs, so
retraining is not required to enable them later.

## Candidate transformation

When transforms are available, every camera-frame candidate is converted by

$$
T_{\mathrm{map,object}} =
T_{\mathrm{map,lidar}}
T_{\mathrm{lidar,camera}}
T_{\mathrm{camera,object}}.
$$

The candidate center is projected onto `track_map.npz` to obtain Frenet
coordinates \((s,d,h)\), local widths, and the centerline tangent.
If the centered CAD origin should sit above the mapped surface, set
`--map-expected-center-height-m` instead of forcing the expected height to zero.

The map-augmented unary cost is

$$
E_{\mathrm{unary}} = E_{\mathrm{RGB}}
+ \lambda_{\mathrm{map}}
  \left(E_{\mathrm{boundary}}+E_{\mathrm{heading}}+E_{\mathrm{height}}\right).
$$

With half car width \(w/2\), the zero-penalty interval is reduced so the body,
not merely its center, remains inside

$$
-w_{\mathrm{left}}(s) \le d \le w_{\mathrm{right}}(s).
$$

The five-frame map path additionally penalizes backward progress, longitudinal
acceleration, and lateral velocity. All map costs are soft by default.

## Safety-first phases

1. **No map available:** omit `--track-map`; run exactly like sliding-window.
2. **Diagnostics only:** pass the map and transforms but leave
   `--map-weight 0 --map-motion-weight 0`.
3. Inspect `track_s_m`, `track_d_m`, height, boundary violation, and heading
   error in `candidate_diagnostics.csv`.
4. **Soft reranking:** introduce small nonzero weights.
5. Only after validation, optionally set a finite
   `--map-hard-boundary-violation-m`.

Do not use a hard projection initially. A centerline, pit-lane branch,
extrinsic, map registration, or CAD-origin error can otherwise force a good RGB
pose to a wrong map position.

## Current Assetto Corsa data without transforms

Run without map arguments:

```bash
python3 -m tracking.rgb_self_recovery.map_window_recovery.run \
  --predictions PATH/GigaPoseMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/NEW_ASSETTO_CORSA \
  --checkpoint gigaPose_datasets/results/rgb_window_model/best.ckpt \
  --mesh gigaPose_datasets/datasets/NEW_ASSETTO_CORSA/models/obj_000001.ply \
  --mask-dir /path/to/masks \
  --mask-fallback dataset \
  --output-dir gigaPose_datasets/results/map_window_no_map \
  --window-size 5 \
  --device cuda \
  --overwrite
```

## Later: diagnostics with map transforms

The per-frame `frame_map.json` row must contain `T_map_lidar`/`t_map_lidar`, or
point to a YAML through `sample_metadata_path`/`metadata_path`. The YAML must
contain `t_map_lidar`. The LiDAR-camera transform can come from that YAML as
`t_lidar_camera_prior`, or from an optimized JSON passed explicitly.

```bash
python3 -m tracking.rgb_self_recovery.map_window_recovery.run \
  --predictions PATH/GigaPoseMultiHypothesis.csv \
  --dataset-dir gigaPose_datasets/datasets/REAL_SEQUENCE \
  --checkpoint gigaPose_datasets/results/rgb_window_model/best.ckpt \
  --track-map PATH/track_map.npz \
  --map-extrinsics PATH/extrinsic_iteration_1.json \
  --map-metadata-root PATH/metadata \
  --map-transform-unit m \
  --map-weight 0 \
  --map-motion-weight 0 \
  --output-dir gigaPose_datasets/results/map_window_diagnostics \
  --window-size 5 \
  --device cuda \
  --overwrite
```

After confirming alignment, a conservative first reranking test is:

```bash
--map-weight 0.25 --map-motion-weight 0.10
```

## Masks and retraining

This package exposes the same mask-directory and fallback flags as
`sliding_window`. Generate data and train through local entry points:

```bash
python3 -m tracking.rgb_self_recovery.map_window_recovery.generate_dataset --help
python3 -m tracking.rgb_self_recovery.map_window_recovery.train --help
```

Map values are not network inputs, so the same retrained checkpoint can be
used by the ordinary, sliding-window, and map-window runners.
