"""Shared path helpers for the Assetto development-evaluation wrappers."""

from __future__ import annotations

from pathlib import Path


def resolve_predictions(value: Path) -> Path:
    """Resolve a CSV or a GigaPose result directory to one top-k CSV."""
    if value.is_file():
        return value
    if not value.is_dir():
        raise FileNotFoundError(value)
    candidates = sorted(value.glob("predictions/*MultiHypothesis.csv"))
    if not candidates:
        candidates = sorted(value.glob("*MultiHypothesis.csv"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one MultiHypothesis.csv below {value}; "
            f"found {len(candidates)}: {candidates}"
        )
    return candidates[0]
