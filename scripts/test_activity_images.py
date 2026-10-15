"""Run activity_monitoring on the labelled sample photos and print expected vs predicted.

    python scripts/test_activity_images.py          # the 6 chosen photos (2 per class)
    python scripts/test_activity_images.py --all     # every image under samples/activity/

The folder name is the expected label. Annotated copies go to out/activity/.
"""
import glob
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import activity_monitoring as am  # noqa: E402

CHOSEN = {
    "standing": ("holstein_pd.jpg", "moyer_1909.jpg"),
    "lying": ("vacas_006548.jpg", "tierpark_187.jpg"),
    "sitting": ("fence_barbed.jpg", "other_side.jpg"),
}
if "--all" in sys.argv:
    paths = sorted(glob.glob(os.path.join(ROOT, "samples", "activity", "*", "*.jpg")))
else:
    paths = [os.path.join(ROOT, "samples", "activity", lab, f) for lab, fs in CHOSEN.items() for f in fs]

out_dir = os.path.join(ROOT, "out", "activity")
os.makedirs(out_dir, exist_ok=True)
rows, correct = [], 0
for p in paths:
    expected = os.path.basename(os.path.dirname(p))
    frame = cv2.imread(p)
    pose = am.pose_from_frame(frame)
    r = am.classify_pose(pose)
    am._draw(frame, pose, r, os.path.join(out_dir, f"{expected}_{os.path.basename(p)}"))
    ok = r.label == expected
    correct += ok
    f = "  n/a" if r.front_extension is None else f"{r.front_extension:5.2f}"
    h = "  n/a" if r.hind_extension is None else f"{r.hind_extension:5.2f}"
    rows.append(f"{'OK ' if ok else 'BAD'} expected={expected:9s} got={r.label:9s} conf={r.confidence:.2f} "
                f"front={f} hind={h}  {os.path.basename(p)}")
print("\n".join(rows))
print(f"\n{correct}/{len(paths)} correct")
