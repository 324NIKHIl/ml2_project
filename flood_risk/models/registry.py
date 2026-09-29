"""The three models proposed in the report (section 14).

* Logistic Regression - interpretable, probabilistic baseline (with scaling).
* Random Forest       - bagged trees for non-linear relationships.
* XGBoost             - gradient-boosted trees, the advanced ensemble.

Class imbalance (floods are rare) is handled with class weights /
``scale_pos_weight``. Weighting distorts the raw probabilities, which is why
every model is calibrated afterwards (see ``train.py``).
"""

from __future__ import annotations

from scipy.stats import loguniform, randint, uniform
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def build_models(cfg: dict, pos_weight: float) -> dict[str, ClassifierMixin]:
    seed = cfg["project"]["random_seed"]
    m = cfg["models"]
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", random_state=seed, **m["logistic_regression"])),
        ]),
        "random_forest": RandomForestClassifier(
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed, **m["random_forest"]),
        "xgboost": XGBClassifier(
            scale_pos_weight=pos_weight, eval_metric="aucpr", tree_method="hist",
            n_jobs=-1, random_state=seed, **m["xgboost"]),
    }


# Search spaces for the optional --tune mode (RandomizedSearchCV + TimeSeriesSplit).
SEARCH_SPACES = {
    "logistic_regression": {"clf__C": loguniform(1e-3, 1e2)},
    "random_forest": {
        "n_estimators": randint(150, 500), "max_depth": randint(6, 20),
        "min_samples_leaf": randint(5, 60), "max_features": ["sqrt", 0.3, 0.5],
    },
    "xgboost": {
        "n_estimators": randint(200, 900), "max_depth": randint(3, 8),
        "learning_rate": loguniform(0.01, 0.2), "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.5, 0.5), "min_child_weight": randint(1, 20),
    },
}
