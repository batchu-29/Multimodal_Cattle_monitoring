"""Cattle pose estimation: frames in -> named keypoints out.

    from pose_estimation import SuperAnimalBackend
    poses = SuperAnimalBackend().estimate_frames([frame1, frame2, ...])   # one Pose per frame

A Pose is {name: (x, y, confidence)} in pixel coordinates. Names are OUR canonical names
(see SUPERANIMAL_QUADRUPED below), so swapping the model later means writing a new mapping
and nothing else changes.

Backend: DeepLabCut SuperAnimal-Quadruped, pretrained, no training. Needs `pip install deeplabcut`.
Speed: about 5 s per frame on a plain CPU, so keep the frame count small (or use a GPU).
Limitation: DeepLabCut's per-frame animal slots are not tracked identities; we keep the slot
with the highest mean confidence, which is right for one cow and wrong for a crowded frame.
"""
from __future__ import annotations

import glob
import os
import shutil
import tempfile
from typing import Dict, List, Tuple

import cv2
import numpy as np

# bodypart name -> (x, y, confidence), pixel coordinates in the full frame
Pose = Dict[str, Tuple[float, float, float]]

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


class SuperAnimalBackend:
    def __init__(self, model_name: str = "hrnet_w32", detector_name: str = "fasterrcnn_resnet50_fpn_v2"):
        self.model_name = model_name
        self.detector_name = detector_name

    def estimate_frames(self, frames: List[np.ndarray]) -> List[Pose]:
        """One Pose per frame, in order (empty Pose if nothing was detected). Frames must be the
        same size. DeepLabCut's API takes videos, so they are written to a tiny temp clip."""
        from deeplabcut.modelzoo.video_inference import video_inference_superanimal  # lazy: heavy

        work = tempfile.mkdtemp(prefix="dlc_")
        try:
            path = os.path.join(work, "frames.mp4")
            h, w = frames[0].shape[:2]
            out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 5, (w, h))
            for f in frames:
                out.write(f)
            out.release()
            video_inference_superanimal(
                [path],
                "superanimal_quadruped",
                model_name=self.model_name,
                detector_name=self.detector_name,
                videotype=".mp4",
                dest_folder=work,
                video_adapt=False,
            )
            h5 = glob.glob(os.path.join(work, "*.h5"))
            if not h5:
                raise RuntimeError("DeepLabCut produced no .h5 output")
            return poses_from_h5(h5[0])
        finally:
            shutil.rmtree(work, ignore_errors=True)


def poses_from_h5(h5_path: str) -> List[Pose]:
    import pandas as pd

    df = pd.read_hdf(h5_path).droplevel("scorer", axis=1)
    individuals = list(df.columns.get_level_values("individuals").unique())

    def mean_conf(ind: str) -> float:
        lik = df[ind].xs("likelihood", level="coords", axis=1).to_numpy(dtype=float)
        return float(np.nanmean(lik)) if np.isfinite(lik).any() else -1.0

    sub = df[max(individuals, key=mean_conf)]
    poses: List[Pose] = []
    for i in range(len(sub)):
        pose: Pose = {}
        for dlc_name, name in SUPERANIMAL_QUADRUPED.items():
            x, y, c = (float(sub.iloc[i][(dlc_name, k)]) for k in ("x", "y", "likelihood"))
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(c):
                pose[name] = (x, y, c)
        poses.append(pose)
    return poses
