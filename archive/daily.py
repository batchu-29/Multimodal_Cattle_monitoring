"""Daily gait check for one cow -> one record. Catches early gait change; not a diagnosis.

    python -m cattle_monitor.daily --cow Gauri --video clip.mp4                 # local, no cloud
    python -m cattle_monitor.daily --cow Gauri --video clip.mp4 --project P --init-table

Two modes:
  clip (default)    pose on the clip sampled at --sample-fps. Measures gait proper: head bob,
                    hind stride asymmetry and back arch, each against her own baseline.
  frames            pose on a few clean, spaced frames. Cheap quick check; posture only (back
                    arch), because stride and head bob need motion over time.

Either way: rule screen -> if anything is over its watch line, the best frame + the numbers go
to Gemini -> trust gate -> one row in the store. Without --project nothing touches the cloud
(record is printed, Gemini is skipped). A scheduler invokes this once per cow per day.
"""
from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .frame_select import Config as FrameConfig
from .frame_select import YoloCattleDetector, select_best_frames
from .gait import GaitConfig, GaitMetrics, compute_gait, compute_static
from .gait_score import Generate, score_gait, vertex_generate
from .pose.base import Pose
from .screening import ScreenConfig, screen
from .store import BigQueryStore, InMemoryStore, GaitRecord, Store

JpegAndTime = Tuple[bytes, float]
PoseFn = Callable[[str], Tuple[float, List[Pose]]]       # video -> (fps, poses)
FramePicker = Callable[[str], Optional[JpegAndTime]]     # video -> best (jpeg, time_s)
FramesPicker = Callable[[str], List[Tuple[np.ndarray, float]]]  # video -> [(frame, time_s)], best first
FramesPoseFn = Callable[[List[np.ndarray]], List[Pose]]


def _jpeg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("could not encode frame")
    return buf.tobytes()


def default_frame_picker(video_path: str) -> Optional[JpegAndTime]:
    best = select_best_frames(video_path, YoloCattleDetector(), FrameConfig(), top_k=1)
    return (_jpeg(best[0][1]), best[0][0].time_s) if best else None


def default_frames_picker(n: int = 3, min_gap_s: float = 0.5) -> FramesPicker:
    def pick(video_path: str) -> List[Tuple[np.ndarray, float]]:
        best = select_best_frames(video_path, YoloCattleDetector(), FrameConfig(), top_k=n, min_gap_s=min_gap_s)
        return [(frame, cand.time_s) for cand, frame in best]
    return pick


def _new_record(cow_name: str, video_path: str, gait: GaitMetrics, now: Optional[datetime]) -> GaitRecord:
    return GaitRecord(
        record_id=str(uuid.uuid4()),
        cow_name=cow_name,
        recorded_at=now or datetime.now(timezone.utc),
        status="insufficient_data",
        clip_path=video_path,
        n_valid_frames=gait.n_valid,
        rule_flag=False,
        head_bob=gait.head_bob,
        hind_asymmetry=gait.hind_asymmetry,
        back_arch=gait.back_arch,
    )


def _history(store: Store, cow_name: str) -> List[Dict[str, Any]]:
    """Her earlier days for the baseline. If the store can't answer (e.g. first run, table not
    created yet), screen with the absolute lines only rather than failing the daily run."""
    try:
        return store.history(cow_name, limit=30)
    except Exception:
        return []


def _decide(
    rec: GaitRecord,
    gait: GaitMetrics,
    get_frame: Callable[[], Optional[JpegAndTime]],
    generate: Optional[Generate],
    screen_cfg: Optional[ScreenConfig],
    history: Sequence[Dict[str, Any]] = (),
) -> GaitRecord:
    """Shared tail: rule screen, then Gemini only if flagged, then the trust gate."""
    if not gait.reliable:
        rec.rule_reasons = gait.reason
        return rec

    screening = screen(gait, screen_cfg or ScreenConfig(), history)
    rec.rule_flag = screening.flagged
    rec.rule_reasons = "; ".join(screening.reasons)

    if not screening.flagged:
        rec.status = "rules_normal"
        return rec
    if generate is None:
        rec.status = "gemini_skipped"
        return rec

    picked = get_frame()
    if picked is None:
        rec.status = "flagged_no_frame"
        return rec

    jpeg, rec.frame_time_s = picked
    rec.gemini_called = True
    try:
        result = score_gait(jpeg, gait, screening, generate)
    except Exception as exc:  # network, quota, auth: record it, never crash the batch
        rec.status = "gemini_error"
        rec.gemini_rationale = f"{type(exc).__name__}: {exc}"[:500]
        return rec

    if result is None:
        rec.status = "gemini_rejected"
    else:
        rec.status = "scored"
        rec.gait_score = result.score
        rec.gemini_confidence = result.confidence
        rec.gemini_rationale = result.rationale
    return rec


def run_daily_frames(
    cow_name: str,
    video_path: str,
    pick_frames: FramesPicker,
    pose_frames: FramesPoseFn,
    store: Store,
    generate: Optional[Generate] = None,
    gait_cfg: Optional[GaitConfig] = None,
    screen_cfg: Optional[ScreenConfig] = None,
    now: Optional[datetime] = None,
) -> GaitRecord:
    """Few-frame mode. The best-ranked frame is also the one Gemini sees."""
    picked = pick_frames(video_path)
    if not picked:
        empty = GaitMetrics(0, 0, False, False, "no clean frame found in the clip")
        rec = _decide(_new_record(cow_name, video_path, empty, now), empty, lambda: None, generate, screen_cfg)
    else:
        frames = [f for f, _ in picked]
        gait = compute_static(pose_frames(frames), gait_cfg)
        rec = _decide(
            _new_record(cow_name, video_path, gait, now), gait,
            lambda: (_jpeg(frames[0]), picked[0][1]), generate, screen_cfg,
            _history(store, cow_name),
        )
    store.add(rec)
    return rec


def run_daily(
    cow_name: str,
    video_path: str,
    pose_fn: PoseFn,
    store: Store,
    pick_frame: FramePicker = default_frame_picker,
    generate: Optional[Generate] = None,
    gait_cfg: Optional[GaitConfig] = None,
    screen_cfg: Optional[ScreenConfig] = None,
    now: Optional[datetime] = None,
) -> GaitRecord:
    """Full-clip mode: pose on every frame, all three gait numbers."""
    fps, poses = pose_fn(video_path)
    gait = compute_gait(poses, fps, gait_cfg)
    rec = _decide(
        _new_record(cow_name, video_path, gait, now), gait, lambda: pick_frame(video_path),
        generate, screen_cfg, _history(store, cow_name),
    )
    store.add(rec)
    return rec


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Daily gait check for one cow.")
    p.add_argument("--cow", required=True, help="cow name / tag")
    p.add_argument("--video", required=True, help="5 s side-view walking clip")
    p.add_argument("--mode", choices=("frames", "clip"), default="clip")
    p.add_argument("--sample-fps", type=float, default=10.0, help="frames/s the pose model runs on in clip mode")
    p.add_argument("--n-frames", type=int, default=3, help="frames to run pose on in frames mode")
    p.add_argument("--project", help="GCP project; enables Gemini and BigQuery")
    p.add_argument("--location", default="us-central1", help="Vertex AI region for Gemini")
    p.add_argument("--bq-location", default="US", help="BigQuery dataset location")
    p.add_argument("--dataset", default="cattle_monitor")
    p.add_argument("--table", default="gait_records")
    p.add_argument("--model", default=None, help="Gemini model id (default: $GEMINI_MODEL or gemini-2.5-flash)")
    p.add_argument("--dry-run", action="store_true", help="do not write to BigQuery")
    p.add_argument("--no-gemini", action="store_true")
    p.add_argument("--init-table", action="store_true", help="create the BigQuery dataset/table first")
    args = p.parse_args(argv)

    from .pose.superanimal import SuperAnimalBackend

    store: Store
    if args.project and not args.dry_run:
        bq = BigQueryStore(args.project, args.dataset, args.table, args.bq_location)
        if args.init_table:
            bq.ensure_table()
        store = bq
    else:
        store = InMemoryStore()

    generate = vertex_generate(args.project, args.location, args.model) if args.project and not args.no_gemini else None
    backend = SuperAnimalBackend()

    if args.mode == "frames":
        rec = run_daily_frames(
            args.cow, args.video, default_frames_picker(args.n_frames), backend.estimate_frames, store, generate=generate
        )
    else:
        rec = run_daily(
            args.cow, args.video, lambda v: backend.estimate_video(v, args.sample_fps), store, generate=generate
        )
    print(json.dumps(rec.to_row(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
