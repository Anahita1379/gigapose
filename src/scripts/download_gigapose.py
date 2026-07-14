# import os
# from pathlib import Path
# import hydra
# from omegaconf import DictConfig
# from src.utils.logging import get_logger
# from huggingface_hub import hf_hub_download

# logger = get_logger(__name__)


# @hydra.main(
#     version_base=None,
#     config_path="../../configs",
#     config_name="train",
# )
# def download(cfg: DictConfig) -> None:
#     root_dir = Path(cfg.machine.root_dir)
#     source_url = "https://huggingface.co/datasets/nv-nguyen/gigaPose/resolve/main/gigaPose_v1.ckpt"
#     pretrained_dir = root_dir / "pretrained"
#     os.makedirs(pretrained_dir, exist_ok=True)

#     download_cmd = f"wget -O {pretrained_dir}/gigaPose_v1.ckpt {source_url}"
#     logger.info(f"Running {download_cmd}")
#     os.system(download_cmd)


# if __name__ == "__main__":
#     download()


import os
from pathlib import Path

import hydra
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig

from src.utils.logging import get_logger

logger = get_logger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="train",
)
def download(cfg: DictConfig) -> None:
    root_dir = Path(cfg.machine.root_dir)
    pretrained_dir = root_dir / "pretrained"
    os.makedirs(pretrained_dir, exist_ok=True)

    logger.info("Downloading gigaPose_v1.ckpt from Hugging Face...")

    hf_hub_download(
        repo_id="nv-nguyen/gigaPose",
        repo_type="dataset",
        filename="gigaPose_v1.ckpt",
        local_dir=pretrained_dir,
    )


if __name__ == "__main__":
    download()