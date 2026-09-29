import pandas as pd

from flood_risk.config import load_config
from flood_risk.risk import classify_probability, generate_alerts, recommendations_for, risk_levels

LEVELS = risk_levels(load_config())


def test_risk_level_boundaries_follow_report_table():
    probs = [0.0, 0.10, 0.2499, 0.25, 0.49, 0.50, 0.7499, 0.75, 0.99, 1.0]
    expected = ["Low", "Low", "Low", "Moderate", "Moderate", "High", "High", "Severe", "Severe", "Severe"]
    assert list(classify_probability(probs, LEVELS)) == expected


def test_out_of_range_probabilities_are_clipped():
    assert list(classify_probability([-0.2, 1.7], LEVELS)) == ["Low", "Severe"]


def test_every_level_has_recommendations():
    for lv in LEVELS:
        assert recommendations_for(lv.name)


def test_alerts_only_for_high_and_severe():
    preds = pd.DataFrame({
        "region_id": ["A", "B", "C", "D"], "district": ["a", "b", "c", "d"], "state": ["s"] * 4,
        "forecast_date": pd.Timestamp("2023-07-02"),
        "flood_probability": [0.1, 0.3, 0.6, 0.9],
    })
    preds["risk_level"] = classify_probability(preds["flood_probability"], LEVELS)
    alerts = generate_alerts(preds, LEVELS, "High")
    assert list(alerts["region_id"]) == ["D", "C"]           # sorted by probability
    assert alerts["message"].str.contains("SEVERE").iloc[0]
