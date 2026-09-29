"""Stage 6: model training, calibration and selection.

Protocol (chronological, leakage-free):
    train split       -> fit each model (optionally hyper-parameter search with TimeSeriesSplit)
    validation split  -> pick the best model (ranking metric on raw scores) and
                         fit an isotonic calibrator so that probabilities are reliable
    test split        -> untouched until the final evaluation (evaluate.py)
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from ..utils import get_logger
from .registry import MODEL_NAMES, SEARCH_SPACES, build_models

log = get_logger(__name__)

SELECTION_METRICS = {"pr_auc": average_precision_score, "roc_auc": roc_auc_score}


def _tune(name, estimator, X, y, cfg):
    t = cfg["models"]["tuning"]
    search = RandomizedSearchCV(
        estimator, SEARCH_SPACES[name], n_iter=t["n_iter"], scoring="average_precision",
        cv=TimeSeriesSplit(n_splits=t["cv_splits"]), n_jobs=1, random_state=cfg["project"]["random_seed"],
        refit=True,
    )
    search.fit(X, y)
    log.info("  tuned %s: best CV PR-AUC %.4f with %s", name, search.best_score_, search.best_params_)
    return search.best_estimator_, search.best_params_


def train_models(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], target: str,
                 cfg: dict, tune: bool = False, only: list[str] | None = None) -> dict:
    """Fit, calibrate and compare all models. Returns a dict with models + validation scores."""
    # rows must be in time order for TimeSeriesSplit
    train = train.sort_values("date")
    X_tr, y_tr = train[features], train[target]
    X_va, y_va = valid[features], valid[target]
    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    metric_name = cfg["models"]["selection_metric"]

    candidates = build_models(cfg, pos_weight)
    if only:
        candidates = {k: v for k, v in candidates.items() if k in only}

    results: dict = {"models": {}, "base_models": {}, "validation": {}, "params": {}}
    for name, estimator in candidates.items():
        start = time.time()
        log.info("Training %s ...", MODEL_NAMES[name])
        if tune:
            base, params = _tune(name, estimator, X_tr, y_tr, cfg)
        else:
            base, params = clone(estimator).fit(X_tr, y_tr), {}
        raw_valid = base.predict_proba(X_va)[:, 1]

        if cfg["models"]["calibrate"]:
            model = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic").fit(X_va, y_va)
        else:
            model = base

        results["models"][name] = model
        results["base_models"][name] = base
        results["params"][name] = params or base.get_params()
        results["validation"][name] = {
            "pr_auc": float(average_precision_score(y_va, raw_valid)),
            "roc_auc": float(roc_auc_score(y_va, raw_valid)),
            "train_seconds": round(time.time() - start, 1),
        }
        log.info("  %-20s valid PR-AUC %.4f | ROC-AUC %.4f | %.1fs", MODEL_NAMES[name],
                 results["validation"][name]["pr_auc"], results["validation"][name]["roc_auc"],
                 results["validation"][name]["train_seconds"])

    best = max(results["validation"], key=lambda n: results["validation"][n][metric_name])
    results["best_model"] = best
    results["selection_metric"] = metric_name
    results["pos_weight"] = pos_weight
    log.info("Best model on validation (%s): %s", metric_name, MODEL_NAMES[best])
    return results


def reference_values(train: pd.DataFrame, features: list[str]) -> dict[str, float]:
    """'Typical' value of each feature (training median) - the baseline for explanations."""
    return {f: float(np.median(train[f])) for f in features}
