"""Gait detection v1: 5 frames -> pose -> Gemini -> one row in BigQuery.

    python gait_detection.py --cow Gauri --video clip.mp4
    python gait_detection.py --cow Gauri --video clip.mp4 --save-frames out/gait   # also dump frames
    python gait_detection.py --cow Gauri --video clip.mp4 --init-table             # create BQ table

Flow (deliberately simple; the earlier metric/baseline version is in archive/):
  1. frame_extraction picks the 5 best frames of the clip (>= 0.5 s apart)
  2. pose_estimation gives the keypoint coordinates for each frame
  3. Gemini gets all 5 images plus their coordinates and returns a 1-5 gait score + one line
  4. one BigQuery row: cow, time, the best frame's image, the one-line response, the score

Trust gate: if Gemini says image_quality_ok is not true, or the score is not an integer 1-5,
NO score is stored (the row is still written, score NULL, response says why).

PLACEHOLDERS: set the project below (or env GCP_PROJECT_ID) to go live. While it is still
the placeholder this runs as a dry run: frames and pose are real, Gemini and BigQuery are
not called, and the prompt and the row are printed instead.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ----------------------------------------------------------------------------- placeholders
PLACEHOLDER = "YOUR_GCP_PROJECT_ID"
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", PLACEHOLDER)
GEMINI_LOCATION = os.environ.get("GEMINI_LOCATION", "us-central1")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")   # check this id exists in your project
BQ_DATASET = os.environ.get("BQ_DATASET", "cattle_monitor")
BQ_TABLE = os.environ.get("BQ_TABLE", "gait_records")
BQ_LOCATION = os.environ.get("BQ_LOCATION", "US")
# -----------------------------------------------------------------------------

N_FRAMES = 5
MIN_GAP_S = 0.5
MIN_KP_CONF = 0.3
MAX_IMAGE_SIDE = 960

Pose = Dict[str, Tuple[float, float, float]]
Content = List[Union[str, bytes]]            # text parts are str, JPEG images are bytes
Generate = Callable[[Content], str]          # contents -> raw JSON text from the model
Frame = Tuple[np.ndarray, float]             # (BGR image, time in clip in seconds)

SCALE = """Gait / locomotion score (Sprecher 1997):
1 = normal: level back, smooth even stride.
2 = mildly lame: back arches when walking, gait slightly abnormal.
3 = moderately lame: back always arched, short strides with one or more legs.
4 = lame: obvious limp, favours one or more limbs, back always arched.
5 = severely lame: cannot or will barely bear weight on a limb, strong reluctance to move."""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "image_quality_ok": {"type": "BOOLEAN"},
        "gait_score": {"type": "INTEGER"},
        "summary": {"type": "STRING"},
    },
    "required": ["image_quality_ok", "gait_score", "summary"],
}

BQ_SCHEMA = [
    ("record_id", "STRING", "REQUIRED"),
    ("cow_name", "STRING", "REQUIRED"),
    ("recorded_at", "TIMESTAMP", "REQUIRED"),
    ("video_path", "STRING", "NULLABLE"),
    ("image", "BYTES", "NULLABLE"),        # the best frame, JPEG
    ("response", "STRING", "NULLABLE"),    # one line, e.g. "score=2 | slight head nod"
    ("score", "INTEGER", "NULLABLE"),      # NULL when the answer failed the trust gate
]


def cloud_configured() -> bool:
    return bool(PROJECT_ID) and PROJECT_ID != PLACEHOLDER


# ------------------------------------------------------------------ frames and pose
def pick_frames(video_path: str, n: int = N_FRAMES, min_gap_s: float = MIN_GAP_S) -> List[Frame]:
    """The n best frames of the clip, best first, from frame_extraction."""
    from frame_extraction import YoloCattleDetector, select_best_frames

    best = select_best_frames(video_path, YoloCattleDetector(), top_k=n, min_gap_s=min_gap_s)
    return [(frame, cand.time_s) for cand, frame in best]


def pose_frames(frames: List[np.ndarray]) -> List[Pose]:
    from pose_estimation import SuperAnimalBackend

    return SuperAnimalBackend().estimate_frames(frames)


def pose_summary(pose: Pose, min_conf: float = MIN_KP_CONF) -> Dict[str, List[float]]:
    """Confident keypoints only, as {name: [x, y, conf]} with pixel coordinates."""
    return {k: [round(x, 1), round(y, 1), round(c, 2)] for k, (x, y, c) in pose.items() if c >= min_conf}


def to_jpeg(frame: np.ndarray, max_side: int = MAX_IMAGE_SIDE, quality: int = 85) -> bytes:
    h, w = frame.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode frame")
    return buf.tobytes()


# ------------------------------------------------------------------ Gemini
def build_contents(frames: Sequence[Frame], poses: Sequence[Pose]) -> Content:
    """Instructions, then for each frame (in time order) its coordinates and its image."""
    h, w = frames[0][0].shape[:2]
    intro = f"""You are assisting a dairy-farm gait check on an Indian cattle breed. The goal is to
notice EARLY gait change before it becomes obvious lameness, so look closely for subtle signs and do not
under-score a slightly abnormal gait. Do not diagnose a cause.

{SCALE}

Below are {len(frames)} frames of the same cow walking, taken at least {MIN_GAP_S}s apart, in time order.
Each frame comes with body-pose keypoints from a pose model: {{name: [x, y, confidence]}} in pixels
(frame is {w}x{h}, origin top-left, y grows downward). Low-confidence points are left out. The frames are
sparse, so you cannot see a full stride; judge back line, head carriage and leg placement from the images,
and use the coordinates to check what you see (for example how far the mid-back sits above the line from
neck to tail, or how far apart the legs are).
If the images are blurred, the cow is cut off or hidden, or not side-on, set image_quality_ok to false.
Answer with JSON only: image_quality_ok, gait_score (1-5), summary (ONE short line explaining the score)."""
    contents: Content = [intro]
    for i, ((frame, t), pose) in enumerate(zip(frames, poses), 1):
        contents.append(f"Frame {i} at t={t:.1f}s keypoints: {json.dumps(pose_summary(pose))}")
        contents.append(to_jpeg(frame))
    return contents


def parse_response(text: Optional[str]) -> Tuple[Optional[int], str]:
    """The trust gate. Returns (score, one-line summary); score is None when it can't be trusted,
    and the line then says why."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None, "answer was not valid JSON"
    if not isinstance(data, dict):
        return None, "answer was not a JSON object"
    summary = str(data.get("summary", "")).strip()
    if data.get("image_quality_ok") is not True:
        return None, f"image quality not ok ({summary or 'no reason given'})"
    score = data.get("gait_score")
    if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
        return None, f"score {score!r} is not an integer 1-5"
    return score, summary


def one_line(score: Optional[int], text: str, limit: int = 300) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    line = f"score={score} | {text}" if score is not None else f"NO SCORE | {text}"
    return line if len(line) <= limit else line[: limit - 3] + "..."


def vertex_generate(
    project: str = PROJECT_ID, location: str = GEMINI_LOCATION, model: str = GEMINI_MODEL
) -> Generate:
    """Gemini on Vertex AI via google-genai. Uses Application Default Credentials
    (`gcloud auth application-default login`)."""
    from google import genai
    from google.genai import types

    client = genai.Client(vertexai=True, project=project, location=location)

    def generate(contents: Content) -> str:
        parts = [
            types.Part.from_bytes(data=c, mime_type="image/jpeg") if isinstance(c, bytes) else c
            for c in contents
        ]
        resp = client.models.generate_content(
            model=model,
            contents=parts,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=RESPONSE_SCHEMA, temperature=0.0
            ),
        )
        return resp.text

    return generate


# ------------------------------------------------------------------ BigQuery
class BigQueryWriter:
    """Appends rows with a load job (works on BigQuery sandbox, where streaming inserts do not)."""

    def __init__(self, project: str = PROJECT_ID, dataset: str = BQ_DATASET, table: str = BQ_TABLE,
                 location: str = BQ_LOCATION, client: Any = None):
        from google.cloud import bigquery

        self._bq = bigquery
        self.client = client or bigquery.Client(project=project, location=location)
        self.dataset_id = f"{project}.{dataset}"
        self.table_id = f"{project}.{dataset}.{table}"
        self.location = location

    def _schema(self):
        return [self._bq.SchemaField(n, t, m) for n, t, m in BQ_SCHEMA]

    def ensure_table(self) -> None:
        """Create the dataset and table if missing. A cloud write: run once on purpose."""
        ds = self._bq.Dataset(self.dataset_id)
        ds.location = self.location
        self.client.create_dataset(ds, exists_ok=True)
        table = self._bq.Table(self.table_id, schema=self._schema())
        table.time_partitioning = self._bq.TimePartitioning(field="recorded_at")
        table.clustering_fields = ["cow_name"]
        self.client.create_table(table, exists_ok=True)

    def add(self, row: Dict[str, Any]) -> None:
        job = self.client.load_table_from_json(
            [row], self.table_id,
            job_config=self._bq.LoadJobConfig(write_disposition="WRITE_APPEND", schema=self._schema()),
        )
        job.result()


# ------------------------------------------------------------------ the run
def run(
    cow_name: str,
    video_path: str,
    n_frames: int = N_FRAMES,
    *,
    pick: Callable[[str, int], List[Frame]] = pick_frames,
    pose_fn: Callable[[List[np.ndarray]], List[Pose]] = pose_frames,
    generate: Optional[Generate] = None,
    writer: Optional[Any] = None,
    now: Optional[datetime] = None,
    log: Callable[[str], None] = print,
    capture: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Returns the row (or None if no usable frames). With generate=None Gemini is not called;
    with writer=None nothing is written. `capture`, if given, is filled with the ranked frames,
    their poses and the exact contents sent to Gemini (for previews and debugging)."""
    ranked = pick(video_path, n_frames)             # best first
    if not ranked:
        log("No clean frame found in the clip; nothing to score.")
        return None
    poses = pose_fn([f for f, _ in ranked])
    log(f"picked {len(ranked)} frames at t=" + ", ".join(f"{t:.1f}s" for _, t in ranked)
        + "; confident keypoints per frame: " + ", ".join(str(len(pose_summary(p))) for p in poses))

    best_jpeg = to_jpeg(ranked[0][0])               # saved image = the best-ranked frame
    order = sorted(range(len(ranked)), key=lambda i: ranked[i][1])   # Gemini sees them in time order
    contents = build_contents([ranked[i] for i in order], [poses[i] for i in order])

    score: Optional[int] = None
    if generate is None:
        response = "DRY RUN | Gemini not called"
    else:
        try:
            score, text = parse_response(generate(contents))
            response = one_line(score, text)
        except Exception as exc:                     # network, quota, auth: keep an audit row
            response = one_line(None, f"ERROR {type(exc).__name__}: {exc}")

    row = {
        "record_id": str(uuid.uuid4()),
        "cow_name": cow_name,
        "recorded_at": (now or datetime.now(timezone.utc)).isoformat(),
        "video_path": video_path,
        "image": base64.b64encode(best_jpeg).decode("ascii"),   # BYTES in a JSON load job = base64
        "response": response,
        "score": score,
    }
    if capture is not None:
        capture.update(frames=[f for f, _ in ranked], poses=poses, contents=contents)
    if writer is not None:
        writer.add(row)
    return row


def _preview(contents: Content) -> str:
    return "\n".join(c if isinstance(c, str) else f"<image: {len(c)} bytes JPEG>" for c in contents)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Gait check: 5 frames -> pose -> Gemini -> BigQuery.")
    p.add_argument("--cow", required=True, help="cow name / tag")
    p.add_argument("--video", required=True, help="short side-view walking clip")
    p.add_argument("--frames", type=int, default=N_FRAMES)
    p.add_argument("--dry-run", action="store_true", help="never call Gemini or BigQuery")
    p.add_argument("--init-table", action="store_true", help="create the BigQuery dataset/table first")
    p.add_argument("--save-frames", metavar="DIR", help="write the picked frames with keypoints drawn")
    args = p.parse_args(argv)

    live = cloud_configured() and not args.dry_run
    if not live:
        why = "--dry-run" if args.dry_run else "GCP project is still the placeholder"
        print(f"DRY RUN ({why}): Gemini and BigQuery will not be called.")
    generate = vertex_generate() if live else None
    writer = BigQueryWriter() if live else None
    if live and args.init_table:
        writer.ensure_table()

    captured: Dict[str, Any] = {}
    row = run(args.cow, args.video, args.frames, generate=generate, writer=writer, capture=captured)
    if row is None:
        return 1

    if args.save_frames:
        os.makedirs(args.save_frames, exist_ok=True)
        for i, (img, pose) in enumerate(zip(captured["frames"], captured["poses"]), 1):
            img = img.copy()
            for _, (x, y, c) in pose.items():
                cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0) if c >= MIN_KP_CONF else (0, 0, 255), -1)
            cv2.imwrite(os.path.join(args.save_frames, f"frame{i}.jpg"), img)

    if not live:
        print("\n--- what Gemini would receive ---\n" + _preview(captured["contents"]))
    shown = dict(row, image=f"<{len(row['image'])} base64 chars>")
    print("\n--- row" + (" written to BigQuery ---" if live else " (not written) ---"))
    print(json.dumps(shown, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
