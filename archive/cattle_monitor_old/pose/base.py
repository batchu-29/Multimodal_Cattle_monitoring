"""Pose type and backend interface. Backends are swappable behind `PoseBackend`."""
from __future__ import annotations

from typing import Dict, Optional, Protocol, Tuple

import numpy as np

# bodypart name -> (x, y, confidence), pixel coordinates in the full frame
Pose = Dict[str, Tuple[float, float, float]]

Box = Tuple[float, float, float, float]  # x1, y1, x2, y2


class PoseBackend(Protocol):
    def estimate(self, frame: np.ndarray, box: Optional[Box] = None) -> Pose:
        """Keypoints for the cow in `frame` (inside `box` if given). Empty dict if none found."""
        ...
