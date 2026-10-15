"""Our own keypoint names, and the mapping from each backend's names to them.

Everything downstream (gait, posture) uses only the canonical names, so swapping the pose
backend means writing a new mapping here and nothing else.
"""
from __future__ import annotations

from typing import Dict

# SuperAnimal-Quadruped name -> canonical name. Names not listed are dropped on purpose:
# antler/ear/throat points are unreliable on cattle, and several points duplicate each other
# (tail_base == back_end, body_middle_left == body_middle_right).
SUPERANIMAL_QUADRUPED: Dict[str, str] = {
    "nose": "nose",
    "neck_base": "neck_base",
    "back_base": "withers",
    "back_middle": "back_middle",
    "tail_base": "tail_base",
    "front_left_thai": "front_left_thigh",   # sic: "thai" is the model's spelling of thigh
    "front_left_knee": "front_left_knee",
    "front_left_paw": "front_left_paw",
    "front_right_thai": "front_right_thigh",
    "front_right_knee": "front_right_knee",
    "front_right_paw": "front_right_paw",
    "back_left_thai": "hind_left_thigh",
    "back_left_knee": "hind_left_knee",
    "back_left_paw": "hind_left_paw",
    "back_right_thai": "hind_right_thigh",
    "back_right_knee": "hind_right_knee",
    "back_right_paw": "hind_right_paw",
}
