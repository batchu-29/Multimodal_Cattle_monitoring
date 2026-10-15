"""Daily rule-based screen: is today unusual for THIS cow? Rules decide IF, Gemini scores the rest.

Two layers, per metric:
  1. Own baseline (needs `min_history` earlier days): flag when today is above her median by
     more than max(k * robust sigma, min_delta). A zebu hump or a naturally bobbing head is
     absorbed into her baseline, so it stops being a false alarm.
  2. Absolute watch line, always applied, and the only layer for a cow with no history yet.
     Deliberately loose; it only catches gross cases and keeps a slowly worsening cow from
     hiding inside her own drifting baseline.

Only normal-looking days feed the baseline (rules_normal, or Gemini score 1-2), so a lame
streak cannot teach the system that lame is normal.

All numbers below are PLACEHOLDERS from two sound-looking zebu clips (back_arch 0.12-0.15).
Calibrate on labelled sound/lame clips before trusting the flag rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any, Dict, List, Optional, Sequence

from .gait import GaitMetrics

METRICS = ("head_bob", "hind_asymmetry", "back_arch")


@dataclass
class ScreenConfig:
    """Absolute watch lines (loose)."""
    head_bob: float = 0.15
    hind_asymmetry: float = 0.30
    back_arch: float = 0.20


@dataclass
class BaselineConfig:
    min_history: int = 5          # earlier usable days needed before her own baseline is used
    k: float = 3.0                # robust sigmas above her median
    min_delta: Dict[str, float] = field(
        default_factory=lambda: {"head_bob": 0.03, "hind_asymmetry": 0.10, "back_arch": 0.03}
    )


@dataclass
class Baseline:
    median: float
    sigma: float                  # robust (1.4826 * MAD)
    n: int


@dataclass
class Screening:
    flagged: bool
    reasons: List[str] = field(default_factory=list)


def _counts_as_normal(row: Dict[str, Any]) -> bool:
    status = row.get("status")
    if status == "rules_normal":
        return True
    score = row.get("gait_score")
    return status == "scored" and isinstance(score, int) and score <= 2


def baselines_from_history(
    rows: Sequence[Dict[str, Any]], cfg: BaselineConfig = BaselineConfig()
) -> Dict[str, Optional[Baseline]]:
    """None for a metric means 'not enough normal days yet'."""
    out: Dict[str, Optional[Baseline]] = {}
    normal = [r for r in rows if _counts_as_normal(r)]
    for name in METRICS:
        vals = [float(r[name]) for r in normal if r.get(name) is not None]
        if len(vals) < cfg.min_history:
            out[name] = None
            continue
        med = median(vals)
        mad = median(abs(v - med) for v in vals)
        out[name] = Baseline(med, 1.4826 * mad, len(vals))
    return out


def screen(
    m: GaitMetrics,
    cfg: ScreenConfig = ScreenConfig(),
    history: Sequence[Dict[str, Any]] = (),
    bcfg: BaselineConfig = BaselineConfig(),
) -> Screening:
    """Unreliable metrics are never 'normal': the pipeline records insufficient_data instead."""
    baselines = baselines_from_history(history, bcfg)
    reasons: List[str] = []
    for name in METRICS:
        value = getattr(m, name)
        if value is None:
            continue
        line = getattr(cfg, name)
        base = baselines[name]
        if base is not None:
            allowed = max(bcfg.k * base.sigma, bcfg.min_delta[name])
            if value - base.median > allowed:
                reasons.append(
                    f"{name} {value:.3f} is +{value - base.median:.3f} above her baseline "
                    f"{base.median:.3f} (n={base.n})"
                )
                continue
        if value > line:
            suffix = "" if base is not None else " (no baseline yet)"
            reasons.append(f"{name} {value:.3f} > {line}{suffix}")
    return Screening(bool(reasons), reasons)
