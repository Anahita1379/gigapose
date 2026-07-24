"""Isolated LightGlue-assisted extension of RGB self-recovery tracking.

The package intentionally imports the existing tracking implementation as a
library without modifying it.  ALIKED and LightGlue are loaded lazily so the
rest of the repository remains usable when their optional dependencies or
weights are unavailable.
"""

from tracking.lightglue_tracking.config import LightGlueTrackingConfig

__all__ = ["LightGlueTrackingConfig"]
