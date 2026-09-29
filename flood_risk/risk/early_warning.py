"""Stage 10: early warning & decision support.

Turns region-level predictions into
* recommended actions for each risk level, and
* alert messages for regions at or above ``early_warning.alert_min_level``.

This is a decision-support aid meant to be used alongside official IMD / CWC /
NDMA warnings, not in place of them.
"""

from __future__ import annotations

import pandas as pd

from .classification import RiskLevel

RECOMMENDATIONS: dict[str, list[str]] = {
    "Low": [
        "Continue routine monitoring of rainfall and river gauges.",
        "Keep drainage channels and culverts clear.",
    ],
    "Moderate": [
        "Increase monitoring frequency of river levels and rainfall forecasts.",
        "Alert district control room and verify emergency contact lists.",
        "Check readiness of pumps, boats and relief material.",
    ],
    "High": [
        "Issue a precautionary flood warning to communities in low-lying areas.",
        "Pre-position rescue teams (SDRF/NDRF), boats and medical supplies.",
        "Prepare relief shelters and identify evacuation routes.",
        "Coordinate with CWC / IMD for the latest official forecasts.",
    ],
    "Severe": [
        "Recommend evacuation of vulnerable, low-lying and riverbank settlements.",
        "Activate the district emergency operations centre and mobilise NDRF/SDRF.",
        "Open relief shelters and secure drinking water, food and medicines.",
        "Restrict movement near rivers, embankments and flood-prone roads.",
    ],
}

HEADLINES = {
    "Low": "No significant flood threat expected.",
    "Moderate": "Elevated flood risk - stay alert.",
    "High": "Flood likely - take precautionary action.",
    "Severe": "Severe flood threat - immediate preparedness action required.",
}


def recommendations_for(level: str) -> list[str]:
    return RECOMMENDATIONS.get(level, [])


def generate_alerts(predictions: pd.DataFrame, levels: list[RiskLevel], min_level: str = "High") -> pd.DataFrame:
    """Build one alert row per region whose risk level is at or above ``min_level``.

    ``predictions`` needs: region_id, district, state, date, flood_probability,
    risk_level and optionally ``top_factors`` (list of human-readable strings).
    """
    rank = {lv.name: lv.rank for lv in levels}
    threshold = rank[min_level]
    at_risk = predictions[predictions["risk_level"].map(rank) >= threshold].copy()
    if at_risk.empty:
        return at_risk.assign(headline=[], message=[], actions=[])

    def message(row) -> str:
        factors = row.get("top_factors")
        why = f" Main drivers: {'; '.join(factors)}." if isinstance(factors, list) and factors else ""
        return (f"[{row['risk_level'].upper()}] {row['district']}, {row['state']} - "
                f"{row['flood_probability']:.0%} probability of flooding on "
                f"{pd.Timestamp(row['forecast_date']).date()}.{why}")

    at_risk["headline"] = at_risk["risk_level"].map(HEADLINES)
    at_risk["message"] = at_risk.apply(message, axis=1)
    at_risk["actions"] = at_risk["risk_level"].map(lambda lv: " | ".join(recommendations_for(lv)))
    return at_risk.sort_values("flood_probability", ascending=False).reset_index(drop=True)
