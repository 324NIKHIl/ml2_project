"""End-to-end smoke test on a short simulation (about 30 s)."""

import numpy as np
import pandas as pd

from flood_risk.models import FloodRiskPredictor
from flood_risk.pipeline import run_pipeline


def test_full_pipeline_runs_and_produces_useful_model(small_cfg):
    result = run_pipeline(small_cfg, skip_eda=True)
    P = small_cfg["paths"]

    for artefact in ["best_model.joblib", "logistic_regression.joblib", "random_forest.joblib", "xgboost.joblib"]:
        assert (P["models_dir"] / artefact).exists()
    for artefact in ["metrics.json", "model_comparison.csv", "test_predictions.parquet", "alerts_test_period.csv"]:
        assert (P["reports_dir"] / artefact).exists()

    comp = result["comparison"].set_index("model")
    assert set(comp.index) == {"logistic_regression", "random_forest", "xgboost"}
    assert (comp["roc_auc"] > 0.85).all()                 # far better than chance (0.5)

    feats = pd.read_parquet(P["processed_dir"] / "features.parquet")
    predictor = FloodRiskPredictor.load(P["models_dir"] / "best_model.joblib")
    rows = feats.sample(50, random_state=0)
    out = predictor.predict(rows)
    assert out["flood_probability"].between(0, 1).all()
    assert set(out["risk_level"]) <= {"Low", "Moderate", "High", "Severe"}

    tables = predictor.explain(rows.head(3), top_n=4)
    assert len(tables) == 3 and all(len(t) == 4 for t in tables)
    assert np.isfinite(tables[0]["contribution"]).all()
