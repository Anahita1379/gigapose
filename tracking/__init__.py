"""Adaptive multi-hypothesis 6D pose tracking for GigaPose outputs.

All poses inside this package are 4x4 OpenCV camera-from-object transforms
whose translations are expressed in metres.
"""

from tracking.config import TrackerConfig
from tracking.tracker import AdaptivePoseTracker

__all__ = ["AdaptivePoseTracker", "TrackerConfig"]
