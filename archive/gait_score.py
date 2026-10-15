"""Gemini gait scoring: one good frame + measured gait numbers -> structured JSON.

Aimed at EARLY gait change (scores 1 vs 2 and any drift from her normal). Obvious lameness
(4-5) is something the farmer sees without us; we are not diagnosing, we are catching the
change before it gets that far.

Gemini sees the image and the numbers we measured; it does no arithmetic and sees no history.
Anything we can't trust is dropped here: bad JSON, image_quality_ok != true, or a score
outside 1-5 all return None and nothing gets stored as a score.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, List, Optional

from .gait import GaitMetrics
from .screening import Screening

# generate(jpeg_bytes, prompt) -> raw JSON text from the model
Generate = Callable[[bytes, str], str]

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
        "confidence": {"type": "NUMBER"},
        "observations": {"type": "ARRAY", "items": {"type": "STRING"}},
        "rationale": {"type": "STRING"},
    },
    "required": ["image_quality_ok", "gait_score", "confidence", "observations", "rationale"],
}


@dataclass
class Assessment:
    score: int
    confidence: float
    observations: List[str]
    rationale: str


def _num(v: Optional[float]) -> str:
    return "not measured" if v is None else f"{v:.3f}"


def build_prompt(m: GaitMetrics, s: Screening) -> str:
    return f"""You are assisting a dairy-farm gait check on an Indian cattle breed. The goal is to
notice EARLY gait change (score 1 vs 2) before it becomes obvious lameness, so look closely for
subtle signs and do not under-score a slightly abnormal gait. Do not diagnose a cause.
The image is one clear side-view frame of a walking cow.

{SCALE}

Numbers measured from body-pose keypoints (already computed; do not recompute them).
Distances are divided by the cow's trunk length. "not measured" means we had too few frames to
measure movement, so judge that aspect from the image only.
- head_bob (vertical head swing over time): {_num(m.head_bob)}
- hind_asymmetry (left vs right hind stride difference, 0 = symmetric): {_num(m.hind_asymmetry)}
- back_arch (mid-back above neck-to-tail line): {_num(m.back_arch)}
Rule-based screen flagged: {"; ".join(s.reasons) or "nothing"}.

A single frame cannot show movement, so use the image for posture (back line, head carriage,
leg placement). If the image is blurred, the cow is cut off, heavily occluded, or not
side-on, set image_quality_ok to false.
Return JSON only."""


def parse_assessment(text: str) -> Optional[Assessment]:
    """The trust gate. None means: do not store a score."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("image_quality_ok") is not True:
        return None
    score = data.get("gait_score")
    if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
        return None
    conf = data.get("confidence")
    obs = data.get("observations")
    return Assessment(
        score=score,
        confidence=float(conf) if isinstance(conf, (int, float)) and not isinstance(conf, bool) else 0.0,
        observations=[str(o) for o in obs] if isinstance(obs, list) else [],
        rationale=str(data.get("rationale", "")),
    )


def score_gait(
    jpeg: bytes, m: GaitMetrics, s: Screening, generate: Generate
) -> Optional[Assessment]:
    return parse_assessment(generate(jpeg, build_prompt(m, s)))


def vertex_generate(
    project: str, location: str = "us-central1", model: Optional[str] = None
) -> Generate:
    """Gemini on Vertex AI via the google-genai SDK. Uses Application Default Credentials."""
    from google import genai  # lazy: tests and rule-only runs don't need it
    from google.genai import types

    model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    client = genai.Client(vertexai=True, project=project, location=location)

    def generate(jpeg: bytes, prompt: str) -> str:
        resp = client.models.generate_content(
            model=model,
            contents=[types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
                temperature=0.0,
            ),
        )
        return resp.text

    return generate
