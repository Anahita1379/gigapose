"""Batch full-pose factor-graph refinement for RGB/CAD candidate sequences."""

from .optimizer import FactorGraphConfig, optimize_sequence

__all__ = ["FactorGraphConfig", "optimize_sequence"]
