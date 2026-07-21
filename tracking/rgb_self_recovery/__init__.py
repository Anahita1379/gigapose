"""Isolated RGB/render-aware multi-hypothesis 6D pose recovery.

This package deliberately does not modify the legacy adaptive tracker.  It
uses no observed depth: all geometric input channels are rendered from the CAD
at a candidate pose and compared with an RGB observation and segmentation.
"""

from tracking.rgb_self_recovery.model import RGBRenderRecoveryNet

__all__ = ["RGBRenderRecoveryNet"]
