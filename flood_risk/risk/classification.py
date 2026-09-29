"""Stage 8: convert a flood probability into an understandable risk level.

    Flood probability   Risk level
    0  - 25 %           Low
    25 - 50 %           Moderate
    50 - 75 %           High
    75 - 100 %          Severe

The thresholds live in ``config.yaml -> risk_levels`` so they can be refined
after checking calibration (as the report suggests).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskLevel:
    name: str
    min: float
    max: float
    color: str
    rank: int


def risk_levels(cfg: dict) -> list[RiskLevel]:
    levels = sorted(cfg["risk_levels"], key=lambda lv: lv["min"])
    return [RiskLevel(lv["name"], float(lv["min"]), float(lv["max"]), lv["color"], i) for i, lv in enumerate(levels)]


def classify_probability(prob, levels: list[RiskLevel]) -> np.ndarray:
    """Vectorised mapping probability -> risk-level name (lower bound inclusive)."""
    prob = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)
    bounds = np.array([lv.min for lv in levels[1:]])
    idx = np.searchsorted(bounds, prob, side="right")
    names = np.array([lv.name for lv in levels], dtype=object)
    return names[idx]


def level_order(levels: list[RiskLevel]) -> pd.CategoricalDtype:
    return pd.CategoricalDtype([lv.name for lv in levels], ordered=True)


def level_colors(levels: list[RiskLevel]) -> dict[str, str]:
    return {lv.name: lv.color for lv in levels}
