"""Show which frames the extractor picks for a clip and how confident the pose model is on them.

    python scripts/debug_frames.py samples/gait_ref/cow1.mp4 [n_frames]
Writes out/debug/<clip>_<i>.jpg with keypoints drawn.
"""
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cattle_monitor.daily import default_frames_picker  # noqa: E402
from cattle_monitor.gait import STATIC_REQUIRED  # noqa: E402
from cattle_monitor.pose.superanimal import SuperAnimalBackend  # noqa: E402

clip = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
picked = default_frames_picker(n)(clip)
print("picked frames at (s):", [round(t, 2) for _, t in picked])
poses = SuperAnimalBackend().estimate_frames([f for f, _ in picked])
out_dir = os.path.join(ROOT, "out", "debug")
os.makedirs(out_dir, exist_ok=True)
stem = os.path.splitext(os.path.basename(clip))[0]
for i, ((frame, t), pose) in enumerate(zip(picked, poses)):
    print(f"-- frame {i} t={t:.2f}s, {len(pose)} keypoints")
    for name in STATIC_REQUIRED:
        x, y, c = pose.get(name, (float("nan"), float("nan"), 0.0))
        print(f"   {name:20s} conf={c:.2f}")
    img = frame.copy()
    for name, (x, y, c) in pose.items():
        cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0) if c >= 0.3 else (0, 0, 255), -1)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_{i}.jpg"), img)
