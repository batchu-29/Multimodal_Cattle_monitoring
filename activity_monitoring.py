"""Activity monitoring: one cow frame in -> standing / sitting / lying out.

    python activity_monitoring.py frame.jpg                 # prints the label
    python activity_monitoring.py frame.jpg --json          # full detail
    python activity_monitoring.py frame.jpg --draw out.jpg  # also saves an annotated image

As a library:
    from activity_monitoring import classify_frame, classify_pose
    result = classify_frame(cv2.imread("frame.jpg"))
    result.label        # "standing" | "sitting" | "lying" | "unknown"

How it decides (rules on pose keypoints, no extra model):
  The pose model finds the thigh and paw of each leg. A standing leg is straight, so the paw
  sits well below the thigh; a folded leg (lying) puts the paw back up near the thigh.
  We measure that drop, divided by the cow's trunk length (front-thigh to hind-thigh) so
  camera distance does not matter, separately for the front legs and the hind legs:

      front extended + hind extended  -> standing
      front extended + hind folded    -> sitting   (rear down, front up)
      front folded   + hind folded    -> lying
      front folded   + hind extended  -> sitting   (front down first: kneeling, getting up/down)

  Sitting is the in-between posture, so it also catches the two transitions. If you define
  sitting differently, change SITTING_RULE in one place below.

Honest limits: the threshold was set from geometry and checked on only ~11 photos (see
scripts/test_activity_images.py). On clean side-on photos it separated the classes, but with
thin margins (standing 0.45-0.63, lying 0.10-0.32, threshold 0.35). It does NOT check that the
cow is side-on: a cow seen from behind can come out "standing" with high confidence.
Calibrate on labelled frames (that is where the planned logistic-regression classifier
takes over).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# so sibling-module imports work however the script is launched (including `python -I`)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

Pose = Dict[str, Tuple[float, float, float]]  # name -> (x, y, confidence)

STANDING, SITTING, LYING, UNKNOWN = "standing", "sitting", "lying", "unknown"

FRONT_THIGHS = ("front_left_thigh", "front_right_thigh")
FRONT_PAWS = ("front_left_paw", "front_right_paw")
HIND_THIGHS = ("hind_left_thigh", "hind_right_thigh")
HIND_PAWS = ("hind_left_paw", "hind_right_paw")

# (front extended?, hind extended?) -> label
SITTING_RULE: Dict[Tuple[bool, bool], str] = {
    (True, True): STANDING,
    (True, False): SITTING,
    (False, True): SITTING,
    (False, False): LYING,
}


@dataclass
class Config:
    min_kp_conf: float = 0.3      # a keypoint below this is ignored
    extended_at: float = 0.35     # leg drop / trunk length at or above this = extended leg
    min_trunk_px: float = 20.0    # smaller than this the cow is too small to judge
    margin_scale: float = 0.25    # how far from the threshold counts as fully sure


@dataclass
class PostureResult:
    label: str
    confidence: float             # 0-1: how clear-cut the geometry is, times keypoint confidence
    front_extension: Optional[float]
    hind_extension: Optional[float]
    trunk_px: Optional[float]
    reason: str


def _usable(pose: Pose, name: str, min_conf: float) -> Optional[np.ndarray]:
    p = pose.get(name)
    return np.array(p[:2], dtype=float) if p is not None and p[2] >= min_conf else None


def _extension(pose: Pose, thighs: Sequence[str], paws: Sequence[str], trunk: float, cfg: Config
               ) -> Tuple[Optional[float], float]:
    """Mean (paw_y - thigh_y)/trunk over the legs on one end of the body, and the mean
    keypoint confidence behind it. Image y grows downward, so a standing leg is positive."""
    drops, confs = [], []
    for thigh, paw in zip(thighs, paws):
        t, p = _usable(pose, thigh, cfg.min_kp_conf), _usable(pose, paw, cfg.min_kp_conf)
        if t is not None and p is not None:
            drops.append((p[1] - t[1]) / trunk)
            confs.append(min(pose[thigh][2], pose[paw][2]))
    return (float(np.mean(drops)), float(np.mean(confs))) if drops else (None, 0.0)


def classify_pose(pose: Pose, cfg: Optional[Config] = None) -> PostureResult:
    cfg = cfg or Config()

    def unknown(reason: str, **kw) -> PostureResult:
        return PostureResult(UNKNOWN, 0.0, kw.get("front"), kw.get("hind"), kw.get("trunk"), reason)

    f_th = [_usable(pose, n, cfg.min_kp_conf) for n in FRONT_THIGHS]
    h_th = [_usable(pose, n, cfg.min_kp_conf) for n in HIND_THIGHS]
    f_th, h_th = [p for p in f_th if p is not None], [p for p in h_th if p is not None]
    if not f_th or not h_th:
        return unknown("need at least one confident front thigh and one hind thigh to size the body")

    trunk = float(np.linalg.norm(np.mean(f_th, axis=0) - np.mean(h_th, axis=0)))
    if trunk < cfg.min_trunk_px:
        return unknown(f"cow too small or end-on (trunk {trunk:.0f}px)", trunk=trunk)

    front, f_conf = _extension(pose, FRONT_THIGHS, FRONT_PAWS, trunk, cfg)
    hind, h_conf = _extension(pose, HIND_THIGHS, HIND_PAWS, trunk, cfg)
    if front is None or hind is None:
        return unknown("front or hind paws not visible with enough confidence", front=front, hind=hind, trunk=trunk)

    f_ext, h_ext = front >= cfg.extended_at, hind >= cfg.extended_at
    label = SITTING_RULE[(f_ext, h_ext)]

    # sure when both ends are far from the threshold and the keypoints are confident
    clarity = min(abs(front - cfg.extended_at), abs(hind - cfg.extended_at)) / cfg.margin_scale
    confidence = float(min(1.0, clarity) * min(f_conf, h_conf))
    reason = (f"front legs {'extended' if f_ext else 'folded'} ({front:.2f}), "
              f"hind legs {'extended' if h_ext else 'folded'} ({hind:.2f}), threshold {cfg.extended_at}")
    return PostureResult(label, confidence, front, hind, trunk, reason)


_backend = None


def pose_from_frame(frame: np.ndarray) -> Pose:
    """Keypoints from the pretrained SuperAnimal-Quadruped model (needs `deeplabcut`)."""
    global _backend
    if _backend is None:
        from pose_estimation import SuperAnimalBackend  # lazy: heavy import

        _backend = SuperAnimalBackend()
    # one-frame videos are fragile in the pose API, so give it the frame twice and keep the first
    return _backend.estimate_frames([frame, frame])[0]


def classify_frame(
    frame: np.ndarray,
    pose_fn: Callable[[np.ndarray], Pose] = pose_from_frame,
    cfg: Optional[Config] = None,
) -> PostureResult:
    return classify_pose(pose_fn(frame), cfg)


def _draw(frame: np.ndarray, pose: Pose, result: PostureResult, path: str) -> None:
    import cv2

    img = frame.copy()
    for name, (x, y, c) in pose.items():
        cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0) if c >= 0.3 else (0, 0, 255), -1)
    text = f"{result.label} ({result.confidence:.2f})"
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imwrite(path, img)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Classify a cow frame as standing / sitting / lying.")
    p.add_argument("image")
    p.add_argument("--json", action="store_true", help="print the full result as JSON")
    p.add_argument("--draw", metavar="OUT.jpg", help="save an annotated copy of the image")
    args = p.parse_args(argv)

    import cv2

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"cannot read image: {args.image}", file=sys.stderr)
        return 2
    pose = pose_from_frame(frame)
    result = classify_pose(pose)
    if args.draw:
        _draw(frame, pose, result, args.draw)
    print(json.dumps(asdict(result), indent=2) if args.json else f"{result.label} ({result.confidence:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
