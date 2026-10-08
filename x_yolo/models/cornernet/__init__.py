"""CornerNet Detecting Objects as Paired Keypoints."""

from .loss import CornerNetLoss
from .model import CornerNet

__all__ = ["CornerNet", "CornerNetLoss"]
