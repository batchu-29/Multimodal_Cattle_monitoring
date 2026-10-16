"""Feeding metrics from a day of bite/chew predictions.

Extracted from livestock-monitoring (backend/api/app/cron.py, commit a4a2c94): the pure
calculation only. The original also queried a database for the cow's last 10 days, compared
today against them, and sent SMS alerts; that part is not copied here.

    sessions = parse_predictions(predictions)       # predictions = [(datetime, label), ...]
    metrics = calculate_metrics(predictions)
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Sequence, Tuple

FEEDING_LABELS = ("chew", "bite", "chew-bite")
SILENCE_THRESHOLD_SECONDS = 300          # a gap longer than this (5 min) starts a new meal

Prediction = Tuple[datetime, str]


def parse_predictions(predictions: Sequence[Prediction]) -> List[List[Prediction]]:
    """Group feeding events into meals; a long silence between two events splits them."""
    sessions: List[List[Prediction]] = []
    current: List[Prediction] = []
    last_time = None

    for timestamp, label in predictions:
        if label in FEEDING_LABELS:
            if last_time and (timestamp - last_time).total_seconds() > SILENCE_THRESHOLD_SECONDS:
                if current:
                    sessions.append(current)
                current = []
            current.append((timestamp, label))
            last_time = timestamp

    if current:
        sessions.append(current)
    return sessions


def calculate_metrics(predictions: Sequence[Prediction]) -> Dict[str, object]:
    sessions = parse_predictions(predictions)
    total_feeding_time = timedelta()
    meal_durations: List[float] = []
    feeding_rates: List[float] = []

    for session in sessions:
        duration = session[-1][0] - session[0][0]
        minutes = duration.total_seconds() / 60.0
        meal_durations.append(minutes)
        total_feeding_time += duration
        feeding_rates.append(len(session) / minutes if minutes > 0 else 0)

    feeding_time = total_feeding_time.total_seconds() / 60.0
    frequency = len(sessions)
    intervals = [
        (sessions[i + 1][0][0] - sessions[i][-1][0]).total_seconds() / 3600.0   # hours
        for i in range(len(sessions) - 1)
    ]
    return {
        "Feeding Time (FT)": feeding_time,                                  # minutes
        "Feeding Frequency (FF)": frequency,                                # meals
        "Meal Durations (MD)": meal_durations,                              # minutes each
        "Average Feeding Time (AFT)": feeding_time / frequency if frequency > 0 else 0,
        "Inter-Meal Intervals (IMI)": intervals,                            # hours
        "Feeding Rates (FR)": feeding_rates,                                # events per minute
        "Total Chews/Bites Per Day": len(predictions),
    }


def average(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0
