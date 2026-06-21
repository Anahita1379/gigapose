# GigaPose Racecar Runbook

This note summarizes how this repo is currently being used for the custom
`racecar` object: what to run first, what gets saved where, and which options
you can change when preparing/running a subset of images.

## 1. Big Picture

GigaPose does not start from only a raw image. For each test image it expects:

- an RGB image
- camera intrinsics
- a CAD model for the object
- rendered CAD templates
- a target list saying which object should be estimated in which image
- an initial detection: bounding box + segmentation mask

For the custom racecar dataset, `src/scripts/prepare_racecar_dataset.py` builds
the BOP/WebDataset-style test data and creates placeholder detections. These
placeholder masks are enough to run the pipeline, but pose quality depends on
replacing them with better car detections/masks.

## 2. Environment Setup

From the repo root:

```bash
conda env create -f environment.yml
conda activate gigapose
bash src/scripts/install_env.sh
pip install -e .
pip install git+https://github.com/thodan/bop_toolkit.git
```

The project expects the main data root to be:

```text
gigaPose_datasets/
```

The default root is configured in:

```text
configs/user/default.yaml
```

Current default:

```yaml
local_root_dir: ./gigaPose_datasets
```

## 3. Download Pretrained Models

Run:

```bash
python3 -m src.scripts.download_gigapose
python3 -m src.scripts.download_megapose
```

Expected checkpoint location:

```text
gigaPose_datasets/pretrained/gigaPose_v1.ckpt
gigaPose_datasets/pretrained/megapose-models/
```

The GigaPose checkpoint path is configured in:

```text
configs/model/large.yaml
```

Current value:

```yaml
checkpoint_path: ${machine.root_dir}/pretrained/gigaPose_v1.ckpt
```

## 4. Prepare the Racecar Dataset

Main script:

```text
src/scripts/prepare_racecar_dataset.py
```

Default source dataset:

```text
/home/anahita/Dataset/rosbag_extracted_300m
```

Expected source layout:

```text
/home/anahita/Dataset/rosbag_extracted_300m/
  images/
  depth/
  intrinsics/
  CAD_car/
  val_image.txt
```

Run with defaults:

```bash
python3 -m src.scripts.prepare_racecar_dataset
```

Useful options:

```bash
python3 -m src.scripts.prepare_racecar_dataset \
  --source-root /home/anahita/Dataset/rosbag_extracted_300m \
  --output-root gigaPose_datasets/datasets \
  --dataset-name racecar \
  --split-file val_image.txt \
  --max-images 100 \
  --mask-file /home/anahita/Dataset/rosbag_extracted_300m/mask/mask_005020.png
```

`--mask-mode full` uses the whole image as the object mask.

`--mask-mode depth` builds a rough mask from valid depth pixels. This can be
better than full-image masks if the depth image mostly contains the object, but
it can also include background if depth is noisy.

`--mask-file` uses a real bright-foreground mask instead of a placeholder. The
script resizes it to the cropped middle RGB image and uses it for the CNOS-style
detection bbox/segmentation.

## 5. Use Only Selected Images

The clean way to use only some images is to choose them while building the
dataset. Do not only edit `test_targets_bop19.json`, because the dataloader
still reads every sample inside the WebDataset tar file.

Use comma-separated source frame IDs:

```bash
python3 -m src.scripts.prepare_racecar_dataset \
  --frame-ids 005020,005025,005030 \
  --mask-file /home/anahita/Dataset/rosbag_extracted_300m/mask/mask_005020.png
```

Or use a text file:

```text
005020
005025
005030
```

Then run:

```bash
python3 -m src.scripts.prepare_racecar_dataset \
  --frame-ids-file my_frames.txt \
  --mask-mode full
```

These IDs are the original source frame IDs from the image filenames. Inside the
generated GigaPose dataset, they are renumbered as:

```text
im_id 0, 1, 2, ...
```

For example, if you select:

```text
005020, 005025, 005030
```

then GigaPose sees them as:

```text
000001_000000
000001_000001
000001_000002
```

If an explicit frame ID is not listed in the selected `--split-file`, the script
will still accept it as long as these files exist:

```text
<source-root>/images/<FRAME_ID>.png
<source-root>/depth/<FRAME_ID>.png
<source-root>/intrinsics/<FRAME_ID>.npy
```

For example, frames `005020`, `005025`, and `005030` are in `train_image.txt`,
not `val_image.txt`, but they can still be selected directly with `--frame-ids`
because the source image/depth/intrinsics files exist.

## 6. What Gets Saved by Dataset Preparation

The dataset preparation script writes:

```text
gigaPose_datasets/datasets/racecar/models/
```

Files:

```text
obj_000001.ply
obj_000001.obj
models_info.json
```

Meaning:

- `obj_000001.ply`: copied CAD mesh
- `obj_000001.obj`: OBJ export used by the template renderer
- `models_info.json`: object diameter and bounding box metadata

It also writes:

```text
gigaPose_datasets/datasets/racecar/test/
```

Files:

```text
shard-000000.tar
key_to_shard.json
```

Meaning:

- `shard-000000.tar`: RGB/depth/camera samples consumed by GigaPose
- `key_to_shard.json`: maps image keys like `000001_000025` to shard `0`

It writes the target list:

```text
gigaPose_datasets/datasets/racecar/test_targets_bop19.json
```

Example entry:

```json
{
  "scene_id": 1,
  "im_id": 0,
  "obj_id": 1,
  "inst_count": 1
}
```

It writes the detection/mask file:

```text
gigaPose_datasets/datasets/cnos-fastsam/cnos-fastsam_racecar-test.json
```

Example entry:

```json
{
  "scene_id": 1,
  "image_id": 0,
  "category_id": 1,
  "score": 1.0,
  "bbox": [0, 0, 768, 320],
  "segmentation": {"counts": [0, 245760], "size": [320, 768]},
  "time": 0.0
}
```

This file is selected by:

```text
src/utils/dataset.py
```

Current racecar mapping:

```python
"racecar": "cnos-fastsam_racecar-test.json"
```

## 7. Render Racecar Templates

GigaPose needs rendered views of the CAD model before testing.

Run:

```bash
python3 -m src.scripts.render_custom_templates custom_dataset_name=racecar
```

Expected output:

```text
gigaPose_datasets/datasets/templates/racecar/
```

Important files/folders:

```text
gigaPose_datasets/datasets/templates/racecar/000001/
gigaPose_datasets/datasets/templates/racecar/object_poses/000001.npy
```

The template config used at test time is in:

```text
configs/data/bop.yaml
```

Current template directory pattern:

```yaml
template_config:
  dir: ${machine.root_dir}/datasets/templates/
```

For `racecar`, GigaPose expands that to:

```text
gigaPose_datasets/datasets/templates/racecar
```

## 8. Run Pose Estimation

Run the coarse GigaPose estimator:

```bash
python3 test.py test_dataset_name=racecar run_id=racecar_test
```

Main config:

```text
configs/test.yaml
```

Important values:

```yaml
save_dir: ${machine.root_dir}/results/${name_exp}
name_exp: ${model.model_name}_${run_id}
test_dataset_name:
run_id:
max_num_dets_per_forward: 4
```

With:

```bash
test_dataset_name=racecar run_id=racecar_test
```

the result directory is:

```text
gigaPose_datasets/results/large_racecar_test/
```

Per-batch `.npz` predictions are saved in:

```text
gigaPose_datasets/results/large_racecar_test/predictions/
```

Final CSV predictions are saved in the same folder:

```text
gigaPose_datasets/results/large_racecar_test/predictions/
  large-pbrreal-rgb-mmodel_racecar-test_racecar_test.csv
  large-pbrreal-rgb-mmodel_racecar-test_racecar_testMultiHypothesis.csv
```

The normal CSV keeps the top pose per detection. The `MultiHypothesis` CSV keeps
multiple pose hypotheses.

For tiny subsets, such as 3 images, `test.py` clamps the internal visualization
log interval to at least 1. Without this, `len(test_dataloader) // 30` becomes
0 and the test loop can crash. The script also clears stale per-batch `.npz`
files from the current `predictions/` directory before a new run, so rerunning a
3-frame subset with a previously used `run_id` does not mix old 100-frame batch
files into the final CSV.

## 9. Run Refinement

After coarse prediction:

```bash
python3 refine.py test_dataset_name=racecar run_id=racecar_test
python3 refine.py test_dataset_name=assettocorsa run_id=ac_front
```

Refined outputs are saved under the same experiment result directory, typically
in one of:

```text
gigaPose_datasets/results/large_racecar_test/refined_predictions/
gigaPose_datasets/results/large_racecar_test/refined_multiple_predictions/
```

The setting:

```yaml
use_multiple: true
```

in `configs/test.yaml` controls whether multiple hypotheses are refined.

## 10. Visualize Racecar Predictions

Script:

```text
src/scripts/visualize_racecar_predictions.py
```

Run with defaults:

```bash
python3 -m src.scripts.visualize_racecar_predictions
```

The visualizer uses:

```text
gigaPose_datasets/datasets/racecar/frame_map.json
```

to map internal GigaPose ids such as `im_id=0` back to the original source frame
ids such as `005020`. If you are visualizing predictions from an older prepared
dataset that does not have this file, pass the same IDs used during preparation:

```bash
python3 -m src.scripts.visualize_racecar_predictions \
  --frame-ids 005020,005025,005030
```

Or specify paths:

```bash
python3 -m src.scripts.visualize_racecar_predictions \
  --predictions gigaPose_datasets/results/large_racecar_test/predictions/large-pbrreal-rgb-mmodel_racecar-test_racecar_test.csv \
  --source-root /home/anahita/Dataset/rosbag_extracted_300m \
  --mesh gigaPose_datasets/datasets/racecar/models/obj_000001.ply \
  --output-dir gigaPose_datasets/results/large_racecar_test/overlays \
  --split-file val_image.txt \
  --max-images 20
```

For the new dataset, run : 
```bash
python3 -m src.scripts.visualize_racecar_predictions \
  --prediction-file gigaPose_datasets/results/large_arcl_racecar_test/refined_multiple_predictions/large-pbrreal-rgb-mmodel_racecar-test_testMultiHypothesis.csv \
  --frame-map gigaPose_datasets/datasets/racecar/frame_map.json \
  --output-dir gigaPose_datasets/results/large_arcl_racecar_test/overlays \
  --max-images 23
```
```bash
python3 -m src.scripts.visualize_racecar_predictions \
  --prediction-file  /home/anahita/gigapose/gigaPose_datasets/results/large_ac_front/refined_multiple_predictions/large-pbrreal-rgb-mmodel_assettocorsa-test_frontMultiHypothesis.csv \
  --frame-map /home/anahita/gigapose/gigaPose_datasets/datasets/assettocorsa/frame_map.json \
  --output-dir gigaPose_datasets/results/large_AC_test/overlays \
  --max-images 100
```



Expected output:

```text
gigaPose_datasets/results/large_racecar_test/overlays/
```

The overlay draws:

- projected CAD bounding box
- object coordinate axes
- predicted object origin

If the pose is good, the projected box should roughly sit on the car.

## 11. Important Input Options You Can Change

Dataset preparation:

```bash
--source-root
```

Where the raw rosbag-extracted dataset lives.

```bash
--output-root
```

Where the generated GigaPose dataset is written. Usually keep this as:

```text
gigaPose_datasets/datasets
```

```bash
--dataset-name
```

Dataset name used by configs and output paths. Current value:

```text
racecar
```

If you change this, you must also add a detection-file mapping in:

```text
src/utils/dataset.py
```

```bash
--split-file
```

Which source split file to read, for example:

```text
val_image.txt
```

```bash
--max-images
```

How many frames to use from the split file. Ignored when `--frame-ids` or
`--frame-ids-file` is provided.

```bash
--frame-ids
--frame-ids-file
```

Use an explicit subset of original source frame IDs.

```bash
--cad-name
```

CAD file inside:

```text
<source-root>/CAD_car/
```

```bash
--mask-mode
```

Use `full` or `depth` placeholder masks.

Testing:

```bash
test_dataset_name=racecar
```

Selects the custom racecar dataset.

```bash
run_id=racecar_test
```

Controls the result folder name.

```bash
max_num_dets_per_forward=4
```

Controls how many detections are processed per forward pass. Lower this if GPU
memory is a problem.

## 12. If There Is No Car in an Image

GigaPose estimates poses for detections it is given. It is not the first-stage
"is there a car?" detector.

Best behavior:

- if no car is present, the detector should output no detection
- do not include that image as a positive target in `test_targets_bop19.json`
- do not create a fake car mask for that image

If a false detection/mask is provided, GigaPose may still output a pose, but it
will likely be meaningless.

## 13. Main Workflow Checklist

Use this order for the racecar pipeline:

1. Install environment.
2. Download pretrained checkpoints.
3. Prepare the racecar dataset.
4. Render custom racecar templates.
5. Run `test.py`.
6. Optionally run `refine.py`.
7. Visualize predictions with `visualize_racecar_predictions.py`.

Commands:

```bash
conda activate gigapose

python3 -m src.scripts.prepare_racecar_dataset \
  --frame-ids 005020,005025,005030 \
  --mask-file /home/anahita/Dataset/rosbag_extracted_300m/mask/mask_005020.png

python3 -m src.scripts.render_custom_templates custom_dataset_name=racecar

python3 test.py test_dataset_name=racecar run_id=racecar_test

python3 -m src.scripts.visualize_racecar_predictions
```

## 14. Note About `gigapose_iac_loader.py`

`gigapose_iac_loader.py` writes an older BOP-folder-style dataset layout under:

```text
/mnt/ssd2tb/gigapose/datasets/racecar
```

The active GigaPose test path in this repo is using the WebDataset layout made
by:

```text
src/scripts/prepare_racecar_dataset.py
```

So for the current workflow, prefer `prepare_racecar_dataset.py`.
