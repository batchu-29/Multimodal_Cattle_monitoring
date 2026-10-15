"""Pick the best single frame of a cow from a video.

Cheap, no-Gemini step: a detector finds cattle, hard gates throw away frames where the
cow is cut off or occluded, and the survivors are ranked by side-on-ness, size, sharpness
and detector confidence. Only the winning frame goes on to pose / Gemini.

The detector sits behind `Detector` so the YOLO model can be swapped (or faked in tests).
Gates here use the bounding box only; once a pose backend exists, add a keypoint
visibility gate on top ("all parts visible" for real).
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Callable, Iterator, List, Optional, Protocol, Tuple

import cv2
import numpy as np

COCO_COW_CLASS = 19


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(self.w, 0.0) * max(self.h, 0.0)


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> List[Detection]: ...


class YoloCattleDetector:
    """Pretrained YOLO (COCO 'cow' class). Needs `pip install ultralytics`.

    COCO has no buffalo class; buffaloes are often detected as cow, but check on real footage.
    """

    def __init__(self, weights: str = "yolo11n.pt", min_conf: float = 0.25):
        from ultralytics import YOLO  # lazy: keep the module importable without it

        self._model = YOLO(weights)
        self._min_conf = min_conf

    def detect(self, frame: np.ndarray) -> List[Detection]:
        result = self._model.predict(
            frame, classes=[COCO_COW_CLASS], conf=self._min_conf, verbose=False
        )[0]
        return [
            Detection(*map(float, box.xyxy[0].tolist()), float(box.conf[0]))
            for box in result.boxes
        ]


@dataclass
class Config:
    sample_fps: float = 5.0
    min_conf: float = 0.5
    edge_margin: float = 0.02      # box must stay this fraction of the frame away from every edge
    max_overlap: float = 0.15      # other cows may cover at most this fraction of the target box
    min_area_frac: float = 0.10    # target box must fill at least this fraction of the frame
    side_aspect: Tuple[float, float] = (1.3, 2.4)  # width/height of a side-on standing cow
    weights: Tuple[float, float, float, float] = (0.35, 0.25, 0.25, 0.15)  # side, sharp, size, conf


@dataclass
class Candidate:
    frame_index: int
    time_s: float
    box: Detection
    side_score: float
    size_score: float
    sharpness: float          # raw Laplacian variance inside the box
    score: float = 0.0
    sharp_score: float = 0.0


def _overlap_frac(target: Detection, other: Detection) -> float:
    iw = min(target.x2, other.x2) - max(target.x1, other.x1)
    ih = min(target.y2, other.y2) - max(target.y1, other.y1)
    if iw <= 0 or ih <= 0 or target.area == 0:
        return 0.0
    return iw * ih / target.area


def _side_score(box: Detection, lo: float, hi: float) -> float:
    """1.0 inside the side-on aspect range, falling to 0 as the box gets squarer/taller."""
    if box.h <= 0:
        return 0.0
    aspect = box.w / box.h
    if lo <= aspect <= hi:
        return 1.0
    edge = lo if aspect < lo else hi
    return max(0.0, 1.0 - abs(aspect - edge) / edge)


def _sharpness(frame: np.ndarray, box: Detection) -> float:
    x1, y1, x2, y2 = (max(int(v), 0) for v in (box.x1, box.y1, box.x2, box.y2))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess_frame(
    frame: np.ndarray, detections: List[Detection], cfg: Config
) -> Optional[Candidate]:
    """Return a Candidate if the frame passes every gate, else None. Score is filled in later."""
    height, width = frame.shape[:2]
    cows = [d for d in detections if d.conf >= cfg.min_conf]
    if not cows:
        return None
    target = max(cows, key=lambda d: d.area)

    mx, my = cfg.edge_margin * width, cfg.edge_margin * height
    if target.x1 < mx or target.y1 < my or target.x2 > width - mx or target.y2 > height - my:
        return None  # cut off by the frame
    area_frac = target.area / (width * height)
    if area_frac < cfg.min_area_frac:
        return None  # too small to read the legs
    if any(_overlap_frac(target, o) > cfg.max_overlap for o in cows if o is not target):
        return None  # another cow in front of / behind her

    return Candidate(
        frame_index=-1,
        time_s=0.0,
        box=target,
        side_score=_side_score(target, *cfg.side_aspect),
        size_score=min(1.0, area_frac / 0.4),
        sharpness=_sharpness(frame, target),
    )


def iter_sampled_frames(path: str, sample_fps: float) -> Iterator[Tuple[int, float, np.ndarray]]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / sample_fps)))
    index = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if index % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    yield index, index / fps, frame
            index += 1
    finally:
        cap.release()


def rank(candidates: List[Candidate], cfg: Config) -> List[Candidate]:
    """Score candidates; sharpness is normalised against the sharpest candidate in this clip."""
    if not candidates:
        return []
    top_sharp = max(c.sharpness for c in candidates) or 1.0
    w_side, w_sharp, w_size, w_conf = cfg.weights
    for c in candidates:
        c.sharp_score = c.sharpness / top_sharp
        c.score = (
            w_side * c.side_score
            + w_sharp * c.sharp_score
            + w_size * c.size_score
            + w_conf * c.box.conf
        )
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def _spread(ranked: List[Candidate], top_k: int, min_gap_s: float) -> List[Candidate]:
    """Best-first, skipping frames closer than min_gap_s to one already chosen
    (neighbouring frames are near-duplicates and would waste pose calls)."""
    chosen: List[Candidate] = []
    for c in ranked:
        if all(abs(c.time_s - o.time_s) >= min_gap_s for o in chosen):
            chosen.append(c)
        if len(chosen) == top_k:
            break
    return chosen


def select_best_frames(
    video_path: str,
    detector: Detector,
    cfg: Optional[Config] = None,
    top_k: int = 1,
    min_gap_s: float = 0.0,
) -> List[Tuple[Candidate, np.ndarray]]:
    """Return the top_k (candidate, frame) pairs, best first. Empty list if no frame qualifies."""
    cfg = cfg or Config()
    candidates: List[Candidate] = []
    for index, time_s, frame in iter_sampled_frames(video_path, cfg.sample_fps):
        cand = assess_frame(frame, detector.detect(frame), cfg)
        if cand is not None:
            cand.frame_index, cand.time_s = index, time_s
            candidates.append(cand)

    best = _spread(rank(candidates, cfg), top_k, min_gap_s)
    if not best:
        return []

    # Re-read only the winners instead of holding every frame in memory.
    wanted = {c.frame_index: c for c in best}
    found = {}
    cap = cv2.VideoCapture(video_path)
    try:
        for idx in sorted(wanted):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                found[idx] = frame
    finally:
        cap.release()
    return [(c, found[c.frame_index]) for c in best if c.frame_index in found]


def save_result(out_dir: str, rank_no: int, cand: Candidate, frame: np.ndarray) -> str:
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"best_{rank_no}")
    cv2.imwrite(stem + ".jpg", frame)
    b = cand.box
    crop = frame[max(int(b.y1), 0):int(b.y2), max(int(b.x1), 0):int(b.x2)]
    cv2.imwrite(stem + "_crop.jpg", crop)
    meta = asdict(cand)
    meta["box"] = asdict(cand.box)
    with open(stem + ".json", "w") as fh:
        json.dump(meta, fh, indent=2)
    return stem + ".jpg"


def main(argv: Optional[List[str]] = None, detector_factory: Callable[[], Detector] = YoloCattleDetector) -> int:
    p = argparse.ArgumentParser(description="Extract the best, fully visible cattle frame from a video.")
    p.add_argument("video")
    p.add_argument("--out", default="out", help="output directory")
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--sample-fps", type=float, default=Config.sample_fps)
    p.add_argument("--weights", default="yolo11n.pt", help="YOLO weights file")
    args = p.parse_args(argv)

    cfg = Config(sample_fps=args.sample_fps)
    detector = detector_factory() if detector_factory is not YoloCattleDetector else YoloCattleDetector(args.weights)
    results = select_best_frames(args.video, detector, cfg, top_k=args.top_k)
    if not results:
        print("No frame passed the checks (cow cut off, occluded, too small, or not detected).")
        return 1
    for i, (cand, frame) in enumerate(results, 1):
        path = save_result(args.out, i, cand, frame)
        print(f"#{i} t={cand.time_s:.2f}s score={cand.score:.3f} "
              f"side={cand.side_score:.2f} sharp={cand.sharp_score:.2f} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
