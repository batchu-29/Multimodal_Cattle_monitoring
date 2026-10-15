"""Why does gait scoring fail on a clip? Per-keypoint confidence over a short window.

    python scripts/diagnose_gait.py samples/gait_ref/cow1.mp4 4.0 5.0 6
        clip, start_s, length_s, sample_fps

Cuts the window, runs pose on ~length*fps frames, and reports which of the keypoints the gait
metrics need are confident, how many frames survive, and how far the cow travels.
"""
import json
import os
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cattle_monitor.gait import REQUIRED, GaitConfig, compute_gait  # noqa: E402
from cattle_monitor.pose.superanimal import SuperAnimalBackend  # noqa: E402

clip, start, length, fps_s = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
cap = cv2.VideoCapture(clip)
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(3)), int(cap.get(4))
cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
tmp = os.path.join(tempfile.mkdtemp(prefix="diag_"), "window.mp4")
out = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
for _ in range(int(length * fps)):
    ok, f = cap.read()
    if not ok:
        break
    out.write(f)
out.release()

eff_fps, poses = SuperAnimalBackend().estimate_video(tmp, sample_fps=fps_s)
n = len(poses)
print(f"\n{n} frames at {eff_fps:.1f} fps  (window {start}s-{start + length}s of {os.path.basename(clip)})\n")
cfg = GaitConfig()
print(f"{'keypoint':20s} {'frames conf>=0.3':>18s} {'median conf':>12s}")
for name in REQUIRED:
    confs = np.array([p[name][2] if name in p else 0.0 for p in poses])
    print(f"{name:20s} {100 * (confs >= cfg.min_conf).mean():17.0f}% {np.median(confs):12.2f}")

m = compute_gait(poses, eff_fps, cfg)
print(f"\nframes with ALL required keypoints confident: {m.n_valid}/{n} (need {cfg.min_valid_frames})")
print(m)

# how far does she travel, in trunk lengths, and where does the pose bunch up?
xs = [np.mean([p[k][0] for k in ("front_left_thigh", "hind_left_thigh") if k in p]) for p in poses
      if "front_left_thigh" in p and "hind_left_thigh" in p]
if xs:
    print(f"\nthigh midpoint x: first {xs[0]:.0f}px -> last {xs[-1]:.0f}px (frame width {w}px)")
os.makedirs(os.path.join(ROOT, "out", "debug"), exist_ok=True)
with open(os.path.join(ROOT, "out", "debug", "diagnose_poses.json"), "w") as fh:
    json.dump({"fps": eff_fps, "poses": poses}, fh)
