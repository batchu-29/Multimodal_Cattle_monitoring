"""Gait detection v1: 5 frames -> pose -> Gemini -> one record in Firebase (Firestore).

    python gait_detection.py --cow Gauri --video clip.mp4
    python gait_detection.py --cow Gauri --video clip.mp4 --save-frames out/gait   # also dump frames
    python gait_detection.py --cow Gauri --video clip.mp4 --dry-run                # never call out

Flow (deliberately simple; the earlier metric/baseline version is in archive/):
  1. frame_extraction picks the 5 best frames of the clip (>= 0.5 s apart)
  2. pose_estimation gives the keypoint coordinates for each frame
  3. Gemini gets all 5 images plus their coordinates and returns a 1-5 gait score + one line
  4. one Firestore document with the whole record: cow, time, the best frame's image, the same
     image with the pose overlay drawn on it, the keypoint coordinates (best frame, plus all 5
     frames with their times), the one-line response, the score. Gemini sees the clean frames,
     not the overlay.

Trust gate: if Gemini says image_quality_ok is not true, or the score is not an integer 1-5,
NO score is stored (the record is still written, score null, response says why).

CONFIG (copy .env.example to .env, which is git-ignored; never put secrets in code or in git):
  Gemini, one of two routes:
    GCP_PROJECT_ID        Vertex AI route: billed to your Google Cloud account/credits. Needs
                          `gcloud auth application-default login` once. (GCP_LOCATION, default us-central1)
    GEMINI_API_KEY        Gemini Developer API route: your key from AI Studio.
    GEMINI_BACKEND        optional: force "vertex" or "api_key" (auto = Vertex when a project is set)
  FIREBASE_CREDENTIALS  path to the Firebase service-account JSON (Project settings > Service accounts)
  FIREBASE_PROJECT_ID   optional, only if it is not inside the JSON
Whatever is missing is skipped and the run says so: with no Gemini route it is a dry run (frames
and pose are real, Gemini is not called, the prompt is printed); with Gemini but no Firebase
credentials it scores the cow and prints the record without saving it. Dry-run records are never saved.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load_env_file(path: str) -> None:
    """Minimal .env reader: KEY=VALUE lines. Never overrides a variable that is already set.
    utf-8-sig, because Windows editors and PowerShell often add an invisible BOM that would
    otherwise glue itself onto the first variable's name and silently lose it."""
    try:
        with open(path, encoding="utf-8-sig") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    except OSError:
        pass


load_env_file(os.path.join(HERE, ".env"))

# ----------------------------------------------------------------------------- config
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")           # set => Gemini on Vertex AI (uses gcloud login)
GCP_LOCATION = os.environ.get("GCP_LOCATION", "global")   # the 3.x models are only served from "global"
GEMINI_BACKEND = os.environ.get("GEMINI_BACKEND", "").lower()   # "vertex" | "api_key" | "" (auto)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")   # check this id exists on your backend
FIREBASE_CREDENTIALS = os.environ.get("FIREBASE_CREDENTIALS", "")
FIREBASE_PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID", "")
FIRESTORE_COLLECTION = os.environ.get("FIRESTORE_COLLECTION", "gait_records")
PLACEHOLDER_MARK = "YOUR_"        # values copied straight from .env.example count as "not set"
# -----------------------------------------------------------------------------

N_FRAMES = 5
MIN_GAP_S = 0.5
MIN_KP_CONF = 0.3
MAX_IMAGE_SIDE = 960
MAX_STORED_IMAGE_BYTES = 450_000   # a Firestore document is capped at 1 MiB in total; we store 2 images

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

# the fields of one record (the Firestore document id is record_id)
ROW_FIELDS = ("record_id", "cow_name", "recorded_at", "video_path", "image", "overlay_image",
              "frame_width", "frame_height", "keypoints", "frames", "response", "score")

# keypoint pairs joined by a line in the overlay (spine, head, and the four legs)
SKELETON = (
    ("nose", "neck_base"), ("neck_base", "back_middle"), ("back_middle", "tail_base"),
    ("front_left_thigh", "front_left_knee"), ("front_left_knee", "front_left_paw"),
    ("front_right_thigh", "front_right_knee"), ("front_right_knee", "front_right_paw"),
    ("hind_left_thigh", "hind_left_knee"), ("hind_left_knee", "hind_left_paw"),
    ("hind_right_thigh", "hind_right_knee"), ("hind_right_knee", "hind_right_paw"),
)


def _is_set(value: str) -> bool:
    return bool(value) and PLACEHOLDER_MARK not in value


def gemini_backend() -> Optional[str]:
    """Which Gemini route to use: "vertex" (Google Cloud project + gcloud login, billed to the
    cloud account) or "api_key" (Gemini Developer API). Auto: Vertex when a project is set."""
    if GEMINI_BACKEND == "vertex" or (GEMINI_BACKEND != "api_key" and _is_set(GCP_PROJECT_ID)):
        return "vertex" if _is_set(GCP_PROJECT_ID) else None
    return "api_key" if _is_set(GEMINI_API_KEY) else None


def adc_available() -> bool:
    """True if Google Application Default Credentials exist (`gcloud auth application-default login`)."""
    try:
        import google.auth

        google.auth.default()
        return True
    except Exception:
        return False


def gemini_configured() -> bool:
    backend = gemini_backend()
    return backend == "api_key" or (backend == "vertex" and adc_available())


def gemini_status() -> str:
    """One line for the run log; says exactly what is missing."""
    backend = gemini_backend()
    if backend == "api_key":
        return "live (Gemini API key)"
    if backend == "vertex":
        return (f"live (Vertex AI, project {GCP_PROJECT_ID}, {GCP_LOCATION})" if adc_available() else
                "OFF (no gcloud login found: run `gcloud auth application-default login`)")
    return "OFF (set GCP_PROJECT_ID for Vertex AI, or GEMINI_API_KEY, in .env)"


def firebase_configured() -> bool:
    return _is_set(FIREBASE_CREDENTIALS) and os.path.isfile(FIREBASE_CREDENTIALS)


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


def keypoints_record(pose: Pose) -> Dict[str, Dict[str, float]]:
    """All keypoints for storage as {name: {"x", "y", "conf"}} in pixels of the ORIGINAL frame.
    Firestore cannot hold arrays inside arrays, so each point is a small map. Low-confidence
    points are kept (with their conf) so nothing is lost; filter on conf when reading."""
    return {k: {"x": round(x, 1), "y": round(y, 1), "conf": round(c, 3)} for k, (x, y, c) in pose.items()}


def draw_overlay(frame: np.ndarray, pose: Pose, min_conf: float = MIN_KP_CONF) -> np.ndarray:
    """Copy of the frame with the pose drawn on it: confident keypoints green, skeleton lines
    yellow, low-confidence keypoints small red dots. Marker size scales with the image."""
    img = frame.copy()
    scale = max(img.shape[:2]) / 480.0
    radius, thick = max(2, int(round(3 * scale))), max(1, int(round(1.5 * scale)))
    for a, b in SKELETON:
        if a in pose and b in pose and pose[a][2] >= min_conf and pose[b][2] >= min_conf:
            cv2.line(img, (int(pose[a][0]), int(pose[a][1])), (int(pose[b][0]), int(pose[b][1])),
                     (0, 220, 255), thick, cv2.LINE_AA)
    for x, y, c in pose.values():
        if c >= min_conf:
            cv2.circle(img, (int(x), int(y)), radius, (0, 200, 0), -1, cv2.LINE_AA)
        else:
            cv2.circle(img, (int(x), int(y)), max(1, radius - 1), (0, 0, 255), -1, cv2.LINE_AA)
    return img


def to_jpeg(frame: np.ndarray, max_side: int = MAX_IMAGE_SIDE, quality: int = 85) -> bytes:
    h, w = frame.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode frame")
    return buf.tobytes()


def fit_jpeg(frame: np.ndarray, max_bytes: int = MAX_STORED_IMAGE_BYTES) -> bytes:
    """JPEG small enough to live inside a Firestore document: lower quality, then shrink."""
    side = MAX_IMAGE_SIDE
    for _ in range(8):
        for quality in (85, 70, 55, 40):
            data = to_jpeg(frame, side, quality)
            if len(data) <= max_bytes:
                return data
        side = int(side * 0.75)
    return data


# ------------------------------------------------------------------ Gemini
def build_contents(frames: Sequence[Frame], poses: Sequence[Pose]) -> Content:
    """Instructions, then for each frame (in time order) its coordinates and its image."""
    h, w = frames[0][0].shape[:2]
    intro = f"""You are assisting a dairy-farm gait check on an Indian cattle breed. The goal is to
notice EARLY gait change before it becomes obvious lameness, so look closely for subtle signs and do not
under-score a slightly abnormal gait. Do not diagnose a cause.

Breed note: Indian cattle vary a lot in build. Many have a natural hump over the shoulders, a sloping
rump, loose skin or a lean frame, and they look different from the Holstein-type animals the scale
is usually illustrated with. These traits are normal anatomy, not signs of lameness. Judge each animal
against her own build: look for a back that hunches upward behind the shoulders or loin, uneven or
shortened steps, a head that nods, or a leg held or placed oddly. Build alone must not raise the score.

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


def _redact(text: str, secret: str) -> str:
    """Error messages must never carry the API key into the database or the logs."""
    return text.replace(secret, "***") if secret else text


def gemini_generate(model: Optional[str] = None) -> Generate:
    """Gemini via Vertex AI (project + gcloud login) or the Gemini API (key), per gemini_backend()."""
    from google import genai
    from google.genai import types

    model = model or GEMINI_MODEL
    if gemini_backend() == "vertex":
        client = genai.Client(vertexai=True, project=GCP_PROJECT_ID, location=GCP_LOCATION)
    else:
        client = genai.Client(api_key=GEMINI_API_KEY)

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


# ------------------------------------------------------------------ Firebase
class FirestoreWriter:
    """One Firestore document per record, in the `gait_records` collection (id = record_id)."""

    def __init__(self, credentials_path: str = FIREBASE_CREDENTIALS, project_id: str = FIREBASE_PROJECT_ID,
                 collection: str = FIRESTORE_COLLECTION, client: Any = None):
        if client is None:
            import firebase_admin
            from firebase_admin import credentials, firestore

            if not firebase_admin._apps:
                firebase_admin.initialize_app(
                    credentials.Certificate(credentials_path), {"projectId": project_id} if project_id else None
                )
            client = firestore.client()
        self.client = client
        self.collection = collection

    def add(self, row: Dict[str, Any]) -> None:
        doc = {k: v for k, v in row.items() if k != "record_id"}
        self.client.collection(self.collection).document(row["record_id"]).set(doc)


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
    secret: str = "",
) -> Optional[Dict[str, Any]]:
    """Returns the record (or None if no usable frames). With generate=None Gemini is not called
    and nothing is saved; with writer=None nothing is saved. `capture`, if given, is filled with
    the ranked frames, their poses and the exact contents sent to Gemini (for previews)."""
    ranked = pick(video_path, n_frames)             # best first
    if not ranked:
        log("No clean frame found in the clip; nothing to score.")
        return None
    poses = pose_fn([f for f, _ in ranked])
    log(f"picked {len(ranked)} frames at t=" + ", ".join(f"{t:.1f}s" for _, t in ranked)
        + "; confident keypoints per frame: " + ", ".join(str(len(pose_summary(p))) for p in poses))

    order = sorted(range(len(ranked)), key=lambda i: ranked[i][1])   # Gemini sees them in time order
    contents = build_contents([ranked[i] for i in order], [poses[i] for i in order])

    score: Optional[int] = None
    if generate is None:
        response = "DRY RUN | Gemini not called"
    else:
        try:
            score, text = parse_response(generate(contents))
            response = one_line(score, text)
        except Exception as exc:                     # network, quota, auth: keep an audit record
            response = one_line(None, _redact(f"ERROR {type(exc).__name__}: {exc}", secret))

    row = {
        "record_id": str(uuid.uuid4()),
        "cow_name": cow_name,
        "recorded_at": now or datetime.now(timezone.utc),
        "video_path": video_path,
        "image": fit_jpeg(ranked[0][0]),             # saved image = the best-ranked frame
        "overlay_image": fit_jpeg(draw_overlay(ranked[0][0], poses[0])),   # same frame with the pose drawn
        "frame_width": int(ranked[0][0].shape[1]),   # keypoint coordinates are in these pixels
        "frame_height": int(ranked[0][0].shape[0]),
        "keypoints": keypoints_record(poses[0]),     # keypoints of the saved (best) frame
        "frames": [                                  # all frames Gemini saw, in time order
            {"time_s": round(ranked[i][1], 2), "rank": i + 1, "keypoints": keypoints_record(poses[i])}
            for i in order
        ],
        "response": response,
        "score": score,
    }
    if capture is not None:
        capture.update(frames=[f for f, _ in ranked], poses=poses, contents=contents)
    if writer is not None and generate is not None:  # dry-run records are never saved
        writer.add(row)
    return row


def _preview(contents: Content) -> str:
    return "\n".join(c if isinstance(c, str) else f"<image: {len(c)} bytes JPEG>" for c in contents)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Gait check: 5 frames -> pose -> Gemini -> Firestore.")
    p.add_argument("--cow", required=True, help="cow name / tag")
    p.add_argument("--video", required=True, help="short side-view walking clip")
    p.add_argument("--frames", type=int, default=N_FRAMES)
    p.add_argument("--dry-run", action="store_true", help="never call Gemini or Firebase")
    p.add_argument("--save-frames", metavar="DIR", help="write the picked frames with keypoints drawn")
    args = p.parse_args(argv)

    use_gemini = gemini_configured() and not args.dry_run
    use_firebase = use_gemini and firebase_configured()
    print("Gemini:   " + ("OFF (--dry-run)" if args.dry_run else gemini_status()))
    print("Firebase: " + ("live" if use_firebase else
                          "OFF (set FIREBASE_CREDENTIALS in .env; records will not be saved)"
                          if use_gemini else "OFF"))

    generate = gemini_generate() if use_gemini else None
    writer = FirestoreWriter() if use_firebase else None

    captured: Dict[str, Any] = {}
    row = run(args.cow, args.video, args.frames, generate=generate, writer=writer,
              capture=captured, secret=GEMINI_API_KEY)
    if row is None:
        return 1

    if args.save_frames:
        os.makedirs(args.save_frames, exist_ok=True)
        for i, (img, pose) in enumerate(zip(captured["frames"], captured["poses"]), 1):
            cv2.imwrite(os.path.join(args.save_frames, f"frame{i}.jpg"), draw_overlay(img, pose))

    if generate is None:
        print("\n--- what Gemini would receive ---\n" + _preview(captured["contents"]))
    shown = dict(row, image=f"<{len(row['image'])} bytes JPEG>",
                 overlay_image=f"<{len(row['overlay_image'])} bytes JPEG>",
                 keypoints=f"<{len(row['keypoints'])} keypoints>",
                 frames=[f"t={f['time_s']}s rank {f['rank']}: {len(f['keypoints'])} keypoints" for f in row["frames"]],
                 recorded_at=row["recorded_at"].isoformat())
    print("\n--- record" + (" saved to Firestore ---" if writer is not None else " (not saved) ---"))
    print(json.dumps(shown, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
