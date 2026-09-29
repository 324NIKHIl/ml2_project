"""Stage 7: probability prediction with the trained model bundle.

``FloodRiskPredictor`` is the single entry point used by the CLI, the
dashboard and the tests. Given feature rows it returns the flood probability,
the risk level, the recommended actions and the contributing factors.

Contributing factors are computed with a model-agnostic *occlusion* method:
each feature in turn is reset to its typical (training-median) value, and the
change in the model score is that feature's contribution. Positive means the
current value pushes flood risk up.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..features.engineering import describe
from ..risk.classification import RiskLevel, classify_probability
from ..risk.early_warning import recommendations_for


class FloodRiskPredictor:
    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.model = bundle["model"]              # calibrated model -> reliable probabilities
        self.base_model = bundle["base_model"]    # uncalibrated model -> smooth scores for explanations
        self.features: list[str] = bundle["features"]
        self.reference: dict[str, float] = bundle["reference"]
        self.threshold: float = bundle["threshold"]
        self.levels: list[RiskLevel] = bundle["levels"]
        self.model_name: str = bundle["model_name"]
        self.horizon: int = bundle["horizon"]

    @classmethod
    def load(cls, path: str | Path) -> "FloodRiskPredictor":
        return cls(joblib.load(path))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df[self.features])[:, 1]

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        prob = self.predict_proba(df)
        keep = [c for c in ["region_id", "district", "state", "latitude", "longitude", "date"] if c in df.columns]
        out = df[keep].copy().reset_index(drop=True)
        if "date" in out:
            out["forecast_date"] = pd.to_datetime(out["date"]) + pd.Timedelta(days=self.horizon)
        out["flood_probability"] = prob
        out["risk_level"] = classify_probability(prob, self.levels)
        out["predicted_flood"] = (prob >= self.threshold).astype(int)
        out["recommendations"] = out["risk_level"].map(recommendations_for)
        return out

    def explain(self, df: pd.DataFrame, top_n: int = 5) -> list[pd.DataFrame]:
        """Per-row table of the features that raise / lower the flood score the most."""
        X = df[self.features].reset_index(drop=True)
        n, f = X.shape
        base = self.base_model.predict_proba(X)[:, 1]

        occluded = pd.concat([X] * f, ignore_index=True)            # block j = feature j reset
        for j, feat in enumerate(self.features):
            occluded.loc[j * n:(j + 1) * n - 1, feat] = self.reference[feat]
        scores = self.base_model.predict_proba(occluded)[:, 1].reshape(f, n).T   # (n, f)
        contrib = base[:, None] - scores

        tables = []
        for i in range(n):
            t = pd.DataFrame({
                "feature": self.features,
                "factor": [describe(c) for c in self.features],
                "value": X.iloc[i].to_numpy(),
                "typical_value": [self.reference[c] for c in self.features],
                "contribution": contrib[i],
            }).sort_values("contribution", ascending=False, key=np.abs).head(top_n)
            tables.append(t.reset_index(drop=True))
        return tables

    @staticmethod
    def factor_strings(table: pd.DataFrame, only_positive: bool = True) -> list[str]:
        rows = table[table["contribution"] > 0] if only_positive else table
        return [f"{r.factor} = {r.value:,.2f} (typical {r.typical_value:,.2f})" for r in rows.itertuples()]
