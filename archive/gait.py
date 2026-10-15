"""Gait numbers from a sequence of canonical poses (one 5 s side-view clip).

All distances are divided by the cow's trunk length (front-thigh midpoint to hind-thigh
midpoint), so camera distance does not matter. Direction of travel is detected from the
motion, so left-to-right and right-to-left clips give the same numbers.

  head_bob        vertical swing of the nose relative to the body, detrended      (higher = worse)
  hind_asymmetry  |stride_left - stride_right| / mean stride of the hind paws     (higher = worse)
  back_arch       how far the mid-back sits above the neck-to-tail line           (higher = worse)

Caveat: the far-side hind leg can be partly hidden, which inflates asymmetry. Back arch
depends on the weakest keypoints (neck_base, back_middle), so treat it as the least reliable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .pose.base import Pose

REQUIRED = (
    "nose", "neck_base", "back_middle", "tail_base",
    "front_left_thigh", "front_right_thigh", "hind_left_thigh", "hind_right_thigh",
    "hind_left_paw", "hind_right_paw",
)
THIGHS = ("front_left_thigh", "front_right_thigh", "hind_left_thigh", "hind_right_thigh")


@dataclass
class GaitConfig:
    min_conf: float = 0.3            # keypoint confidence needed for a frame to count
    min_valid_frames: int = 15       # fewer usable frames than this -> not enough data
    min_travel_trunks: float = 0.5   # the cow must move at least this many trunk lengths


@dataclass
class GaitMetrics:
    n_frames: int
    n_valid: int
    walking: bool
    reliable: bool
    reason: str                       # why not reliable ("" when reliable)
    head_bob: Optional[float] = None
    hind_asymmetry: Optional[float] = None
    back_arch: Optional[float] = None


def _pt(p: Pose, name: str) -> np.ndarray:
    return np.array(p[name][:2], dtype=float)


def _valid(p: Pose, min_conf: float) -> bool:
    return all(n in p and p[n][2] >= min_conf for n in REQUIRED)


def _amp(values: np.ndarray) -> float:
    return float(np.percentile(values, 95) - np.percentile(values, 5))


STATIC_REQUIRED = ("neck_base", "back_middle", "tail_base") + THIGHS


def compute_static(poses: Sequence[Pose], cfg: Optional[GaitConfig] = None, min_frames: int = 2) -> GaitMetrics:
    """Few-frame mode: only what a still frame can show, i.e. back arch.

    head_bob and hind_asymmetry need motion over time, so they stay None here; a few frames
    cannot measure them (a stride looks different in every frame). Use compute_gait on a clip
    when those matter.
    """
    cfg = cfg or GaitConfig()
    good = [p for p in poses if all(n in p and p[n][2] >= cfg.min_conf for n in STATIC_REQUIRED)]
    n_frames, n_valid = len(poses), len(good)
    if n_valid < min_frames:
        return GaitMetrics(n_frames, n_valid, True, False, f"only {n_valid} usable frames (need {min_frames})")

    arches = []
    for p in good:
        front = (_pt(p, "front_left_thigh") + _pt(p, "front_right_thigh")) / 2
        hind = (_pt(p, "hind_left_thigh") + _pt(p, "hind_right_thigh")) / 2
        trunk = float(np.linalg.norm(front - hind))
        neck, tail, mid = _pt(p, "neck_base"), _pt(p, "tail_base"), _pt(p, "back_middle")
        if trunk < 1e-6 or abs(tail[0] - neck[0]) < 1e-6:
            continue
        line_y = neck[1] + (tail[1] - neck[1]) * (mid[0] - neck[0]) / (tail[0] - neck[0])
        arches.append((line_y - mid[1]) / trunk)
    if not arches:
        return GaitMetrics(n_frames, n_valid, True, False, "degenerate body geometry")
    return GaitMetrics(n_frames, n_valid, True, True, "", None, None, float(np.median(arches)))


def compute_gait(poses: Sequence[Pose], fps: float, cfg: Optional[GaitConfig] = None) -> GaitMetrics:
    cfg = cfg or GaitConfig()
    valid = [(i, p) for i, p in enumerate(poses) if _valid(p, cfg.min_conf)]
    n_frames, n_valid = len(poses), len(valid)

    def fail(reason: str, walking: bool = False) -> GaitMetrics:
        return GaitMetrics(n_frames, n_valid, walking, False, reason)

    if n_valid < cfg.min_valid_frames:
        return fail(f"only {n_valid} usable frames (need {cfg.min_valid_frames})")

    t = np.array([i for i, _ in valid], dtype=float) / fps
    ref = np.array([np.mean([_pt(p, n) for n in THIGHS], axis=0) for _, p in valid])
    front = np.array([(_pt(p, "front_left_thigh") + _pt(p, "front_right_thigh")) / 2 for _, p in valid])
    hind = np.array([(_pt(p, "hind_left_thigh") + _pt(p, "hind_right_thigh")) / 2 for _, p in valid])
    trunk = float(np.median(np.linalg.norm(front - hind, axis=1)))
    if trunk < 1e-6:
        return fail("degenerate body length")

    travel = float(ref[-1, 0] - ref[0, 0])
    if abs(travel) < cfg.min_travel_trunks * trunk:
        return fail("cow is not walking across the frame", walking=False)
    direction = 1.0 if travel > 0 else -1.0

    # head bob: nose height relative to the body, linear trend (camera tilt / slope) removed
    head = np.array([_pt(p, "nose")[1] for _, p in valid]) - ref[:, 1]
    head = head - np.polyval(np.polyfit(t, head, 1), t)
    head_bob = _amp(head) / trunk

    # hind stride: how far each hind paw swings back and forth relative to the body
    def stride(paw: str) -> float:
        rel = (np.array([_pt(p, paw)[0] for _, p in valid]) - ref[:, 0]) * direction
        return _amp(rel) / trunk

    left, right = stride("hind_left_paw"), stride("hind_right_paw")
    mean_stride = (left + right) / 2
    hind_asym = abs(left - right) / mean_stride if mean_stride > 1e-6 else 0.0

    # back arch: height of the mid-back above the straight neck->tail line
    arches = []
    for _, p in valid:
        neck, tail, mid = _pt(p, "neck_base"), _pt(p, "tail_base"), _pt(p, "back_middle")
        if abs(tail[0] - neck[0]) < 1e-6:
            continue
        line_y = neck[1] + (tail[1] - neck[1]) * (mid[0] - neck[0]) / (tail[0] - neck[0])
        arches.append((line_y - mid[1]) / trunk)  # image y grows downward, so above = positive
    back_arch = float(np.median(arches)) if arches else None

    return GaitMetrics(n_frames, n_valid, True, True, "", head_bob, float(hind_asym), back_arch)
