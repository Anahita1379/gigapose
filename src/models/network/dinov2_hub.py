from pathlib import Path
import shutil

import torch

"""
Compatibility wrapper around torch.hub.load("facebookresearch/dinov2", ...).

The gigapose conda environment currently uses Python 3.9. Newer DINOv2 cache
revisions include annotations like "float | None", which are evaluated at import
time and fail on Python 3.9 unless "from __future__ import annotations" is
present. Instead of editing ~/.cache/torch/hub in place, copy the cached DINOv2
repo into a writable repo-local folder and patch that copy.

We also allow pretrained=False. The GigaPose checkpoint is loaded later by
Lightning, so avoiding DINOv2's own weight download keeps offline runs working.
"""


def _add_future_annotations(path: Path):
    """Add postponed annotation evaluation to a cached DINOv2 source file."""
    if not path.exists():
        return
    text = path.read_text()
    future_import = "from __future__ import annotations"
    if future_import in text:
        return
    path.write_text(f"{future_import}\n{text}")


def patch_dinov2_for_python39(hub_dir: Path):
    """Patch the DINOv2 files that use Python 3.10 union type annotations."""
    for relative_path in (
        "dinov2/layers/attention.py",
        "dinov2/layers/block.py",
    ):
        _add_future_annotations(hub_dir / relative_path)


def load_dinov2_model(repo_or_dir, model, pretrained=True):
    """Hydra target used by configs/model/ae_net/dinov2_l.yaml."""
    cached_dir = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    patched_dir = Path.cwd() / ".torch_hub_patched" / "facebookresearch_dinov2_main"

    if cached_dir.exists():
        # Work from a repo-local copy so restricted/sandboxed environments do
        # not need write access to ~/.cache/torch/hub.
        if not patched_dir.exists():
            shutil.copytree(cached_dir, patched_dir)
        patch_dinov2_for_python39(patched_dir)
        return torch.hub.load(
            str(patched_dir), model, source="local", pretrained=pretrained
        )

    model_ = torch.hub.load(repo_or_dir, model, pretrained=pretrained)
    return model_
