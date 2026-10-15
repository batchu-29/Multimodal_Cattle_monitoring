"""Run pose + gait metrics on the five reference cow clips and cache the poses.

    python scripts/run_gait_ref.py        # cow 1 only, from the project root
    python scripts/run_gait_ref.py 2 3    # chosen cows

Poses are cached as JSON so thresholds can be tuned without rerunning the (slow) pose model.
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cattle_monitor.gait import compute_gait  # noqa: E402
from cattle_monitor.pose.superanimal import SuperAnimalBackend  # noqa: E402

CACHE = os.environ.get("GAIT_REF_CACHE", r"C:\Users\pbach\gait_ref_poses")
os.makedirs(CACHE, exist_ok=True)
backend = SuperAnimalBackend()
cows = [int(a) for a in sys.argv[1:]] or [1]   # default: cow 1 only (pose on a full clip is slow on CPU)
for n in cows:
    t0 = time.time()
    clip = os.path.join(ROOT, "samples", "gait_ref", f"cow{n}.mp4")
    fps, poses = backend.estimate_video(clip, sample_fps=8)
    with open(os.path.join(CACHE, f"cow{n}.json"), "w") as fh:
        json.dump({"fps": fps, "poses": poses}, fh)
    m = compute_gait(poses, fps)
    print(f"cow{n}: {len(poses)} frames @ {fps:.1f}fps in {time.time() - t0:.0f}s -> {m}", flush=True)
