"""DeepLabCut SuperAnimal-Quadruped backend (pretrained, no training).

Runs on a whole clip at once (much cheaper than frame-by-frame) and returns one canonical
`Pose` per video frame. Needs `pip install deeplabcut`.

Limitation: DeepLabCut's per-frame animal slots are not tracked identities. We keep the slot
with the highest mean confidence over the clip, which is right for a one-cow side-view clip
and wrong for a crowded one.
"""
from __future__ import annotations

import glob
import os
import shutil
import tempfile
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .base import Pose
from .keypoints import SUPERANIMAL_QUADRUPED


class SuperAnimalBackend:
    def __init__(self, model_name: str = "hrnet_w32", detector_name: str = "fasterrcnn_resnet50_fpn_v2"):
        self.model_name = model_name
        self.detector_name = detector_name

    def estimate_frames(self, frames: List[np.ndarray]) -> List[Pose]:
        """Pose for a handful of still frames (all the same size). One Pose per frame, in order.

        DeepLabCut's SuperAnimal API takes videos, so the frames are written to a tiny temp clip.
        """
        work = tempfile.mkdtemp(prefix="dlc_frames_")
        try:
            path = os.path.join(work, "frames.mp4")
            h, w = frames[0].shape[:2]
            out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 5, (w, h))
            for f in frames:
                out.write(f)
            out.release()
            _, poses = self.estimate_video(path)
            return poses
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def estimate_video(self, video_path: str, sample_fps: Optional[float] = None) -> Tuple[float, List[Pose]]:
        """Return (effective fps, [Pose per frame]). A frame with no detection gets an empty Pose.

        sample_fps thins the clip first (every Nth frame) so the pose model runs on far fewer
        frames; ~10 fps is plenty for a ~1 Hz stride. The returned fps is the thinned rate.
        """
        from deeplabcut.modelzoo.video_inference import video_inference_superanimal  # lazy

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        work = tempfile.mkdtemp(prefix="dlc_")
        try:
            if sample_fps and sample_fps < fps - 0.5:
                step = max(1, int(round(fps / sample_fps)))
                thinned = os.path.join(work, "thinned.mp4")
                _thin(video_path, thinned, step, fps / step)
                video_path, fps = thinned, fps / step
            video_inference_superanimal(
                [video_path],
                "superanimal_quadruped",
                model_name=self.model_name,
                detector_name=self.detector_name,
                videotype=os.path.splitext(video_path)[1] or ".mp4",
                dest_folder=work,
                video_adapt=False,
            )
            h5 = [p for p in glob.glob(os.path.join(work, "*.h5"))]
            if not h5:
                raise RuntimeError("DeepLabCut produced no .h5 output")
            return fps, poses_from_h5(h5[0])
        finally:
            shutil.rmtree(work, ignore_errors=True)


def _thin(src: str, dst: str, step: int, fps: float) -> None:
    cap = cv2.VideoCapture(src)
    out = None
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                if out is None:
                    h, w = frame.shape[:2]
                    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                out.write(frame)
            i += 1
    finally:
        cap.release()
        if out is not None:
            out.release()


def poses_from_h5(h5_path: str) -> List[Pose]:
    import pandas as pd

    df = pd.read_hdf(h5_path)
    df = df.droplevel("scorer", axis=1)
    individuals = list(df.columns.get_level_values("individuals").unique())

    def mean_conf(ind: str) -> float:
        lik = df[ind].xs("likelihood", level="coords", axis=1).to_numpy(dtype=float)
        return float(np.nanmean(lik)) if np.isfinite(lik).any() else -1.0

    best = max(individuals, key=mean_conf)
    sub = df[best]
    poses: List[Pose] = []
    for i in range(len(sub)):
        pose: Pose = {}
        for dlc_name, name in SUPERANIMAL_QUADRUPED.items():
            x, y, c = (float(sub.iloc[i][(dlc_name, k)]) for k in ("x", "y", "likelihood"))
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(c):
                pose[name] = (x, y, c)
        poses.append(pose)
    return poses
