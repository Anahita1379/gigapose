import os
import numpy as np
from tqdm import tqdm
import time
from functools import partial
import multiprocessing
import hydra
from omegaconf import OmegaConf
import glob
from pathlib import Path
import shutil
import subprocess
import sys

from src.lib3d.template_transform import get_obj_poses_from_template_level
from src.utils.logging import get_logger

logger = get_logger(__name__)

"""
Render template images for a custom BOP-style dataset.

GigaPose needs a bank of synthetic CAD views before it can estimate pose. For
each object mesh in <root>/datasets/<dataset>/models, this script renders 162
template RGBA images and 162 depth images into:

  <root>/datasets/templates/<dataset>/<OBJECT_ID>/

It also saves the camera/object poses used for those renders in:

  <root>/datasets/templates/<dataset>/object_poses/<OBJECT_ID>.npy

The main custom-dataset gotcha is path handling: rendering happens in a child
Python process, and Panda3D failed to resolve relative mesh paths. This version
resolves dataset/model paths before spawning the renderer.
"""


def call_render(
    idx_obj,
    list_cad_path,
    list_output_dir,
    list_obj_pose_path,
    disable_output,
    num_gpus,
    use_blenderProc,
):
    """Render all template views for one CAD object."""
    output_dir = list_output_dir[idx_obj]
    cad_path = list_cad_path[idx_obj]
    obj_pose_path = list_obj_pose_path[idx_obj]

    # Re-render from scratch so stale/incomplete PNGs cannot be mistaken for a
    # successful template bank.
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    gpus_device = idx_obj % num_gpus
    os.makedirs(output_dir, exist_ok=True)
    if use_blenderProc:  # TODO: remove blenderProc
        command = [
            "blenderproc",
            "run",
            "./src/lib3d/blenderproc.py",
            str(cad_path),
            str(obj_pose_path),
            str(output_dir),
            str(gpus_device),
        ]
    else:  # TODO: understand why this is not working for tless and itodd
        # Use the current interpreter instead of plain "python" so the renderer
        # runs inside the same conda environment as the parent command.
        command = [
            sys.executable,
            "-m",
            "src.custom_megapose.call_panda3d",
            str(cad_path),
            str(obj_pose_path),
            str(output_dir),
            str(gpus_device),
        ]

    if disable_output:
        command.append("true")
    else:
        command.append("false")
    command.append("true")  # scale translation to meter
    subprocess.run(command, check=False)

    # The renderer writes one RGBA PNG and one depth PNG for each template pose.
    num_images = len(glob.glob(f"{output_dir}/*.png"))
    if num_images == len(np.load(obj_pose_path)) * 2:
        return True
    else:
        logger.info(f"Found only {num_images} for  {cad_path} {obj_pose_path}")
        return False


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="train",
)
def render(cfg) -> None:
    num_gpus = 4
    disable_output = True

    OmegaConf.set_struct(cfg, False)
    # Resolve the root path before handing paths to the child renderer. Relative
    # paths were the reason Panda3D could find the file on disk but fail to load
    # it as a model.
    root_dir = Path(cfg.data.test.root_dir).resolve()
    root_save_dir = root_dir / "templates"
    template_poses = get_obj_poses_from_template_level(level=1, pose_distribution="all")
    template_poses[:, :3, 3] *= 0.4  # zoom to object
    dataset_name = cfg.custom_dataset_name

    dataset_save_dir = root_save_dir / f"{dataset_name}"
    logger.info(f"Rendering templates for {dataset_name}")
    os.makedirs(dataset_save_dir, exist_ok=True)
    obj_pose_dir = dataset_save_dir / "object_poses"
    os.makedirs(obj_pose_dir, exist_ok=True)

    cad_dir = (root_dir / dataset_name / "models").resolve()

    # Accept either BOP PLY meshes or OBJ meshes. If both exist for the same
    # object id, prefer OBJ because Panda3D's loader is more reliable for it.
    cad_paths_by_stem = {path.stem: path for path in sorted(cad_dir.glob("*.ply"))}
    cad_paths_by_stem.update(
        {path.stem: path for path in sorted(cad_dir.glob("*.obj"))}
    )
    cad_paths = sorted(cad_paths_by_stem.values())
    logger.info(f"Found {len(list(cad_paths))} objects in {cad_dir}")
    logger.info(f"Found {len(list(cad_paths))} objects")

    output_dirs = []
    obj_pose_paths = []
    for cad_path in cad_paths:
        object_id = int(os.path.basename(cad_path).split(".")[0][4:])
        output_dir = dataset_save_dir / f"{object_id:06d}"
        output_dirs.append(output_dir)

        obj_pose_path = os.path.join(obj_pose_dir, f"{object_id:06d}.npy")
        obj_pose_paths.append(obj_pose_path)
        np.save(obj_pose_path, template_poses)

    os.makedirs(dataset_save_dir, exist_ok=True)

    logger.info("Start rendering for {} objects".format(len(cad_paths)))
    start_time = time.time()
    call_render_ = partial(
        call_render,
        list_cad_path=cad_paths,
        list_output_dir=output_dirs,
        # Keep a pose path per object. The original script passed one shared
        # obj_pose_path, which would break as soon as there is more than one CAD.
        list_obj_pose_path=obj_pose_paths,
        disable_output=disable_output,
        num_gpus=num_gpus,
        use_blenderProc=True if dataset_name in ["tless", "itodd"] else False,
    )
    with multiprocessing.Pool(processes=cfg.machine.num_workers) as pool:
        values = list(
            tqdm(
                pool.imap_unordered(call_render_, range(len(cad_paths))),
                total=len(cad_paths),
            )
        )
    correct_values = [val for val in values if val]
    logger.info(f"Finished for {len(correct_values)}/{len(cad_paths)} objects")
    finish_time = time.time()
    logger.info(f"Total time {len(cad_paths)}: {finish_time - start_time}")


if __name__ == "__main__":
    render()
