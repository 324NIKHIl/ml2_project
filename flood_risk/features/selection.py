"""Stage 5b: feature selection - identify the factors that matter most for flood risk.

Runs on the *training split only*:

1. drop constant features;
2. rank features by two complementary measures -
   * mutual information (model-free, captures non-linear dependence),
   * Random-Forest impurity importance (captures interactions);
   and average the two ranks;
3. walk down the ranking and skip any feature that is more correlated than
   ``correlation_threshold`` with one already kept (removes redundancy);
4. keep the ``top_k`` best.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

from ..utils import get_logger
from .engineering import describe

log = get_logger(__name__)


def rank_features(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> pd.DataFrame:
    X = X.loc[:, X.std() > 0]
    discrete = [X[c].nunique() <= 2 for c in X.columns]
    mi = mutual_info_classif(X, y, discrete_features=discrete, random_state=seed)
    rf = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=20,
                                class_weight="balanced_subsample", n_jobs=-1, random_state=seed).fit(X, y)
    ranking = pd.DataFrame({
        "feature": X.columns,
        "description": [describe(c) for c in X.columns],
        "mutual_information": mi,
        "rf_importance": rf.feature_importances_,
    })
    ranking["mi_rank"] = ranking["mutual_information"].rank(ascending=False)
    ranking["rf_rank"] = ranking["rf_importance"].rank(ascending=False)
    ranking["combined_rank"] = (ranking["mi_rank"] + ranking["rf_rank"]) / 2
    return ranking.sort_values("combined_rank").reset_index(drop=True)


def select_features(train: pd.DataFrame, candidates: list[str], target: str, cfg: dict) -> tuple[list[str], pd.DataFrame]:
    sel = cfg["features"]["selection"]
    seed = cfg["project"]["random_seed"]
    sample = train.sample(min(len(train), sel["sample_size"]), random_state=seed)
    X, y = sample[candidates], sample[target]

    ranking = rank_features(X, y, seed)
    corr = X[ranking["feature"]].corr().abs()

    kept: list[str] = []
    ranking["selected"] = False
    ranking["dropped_reason"] = ""
    for i, feat in enumerate(ranking["feature"]):
        partner = next((k for k in kept if corr.loc[feat, k] > sel["correlation_threshold"]), None)
        if partner is not None:
            ranking.loc[i, "dropped_reason"] = f"correlated with {partner} (r={corr.loc[feat, partner]:.2f})"
        elif len(kept) >= sel["top_k"]:
            ranking.loc[i, "dropped_reason"] = "outside top_k"
        else:
            kept.append(feat)
            ranking.loc[i, "selected"] = True

    dropped_constant = sorted(set(candidates) - set(ranking["feature"]))
    log.info("Feature selection: %d candidates -> %d selected (constant dropped: %s)",
             len(candidates), len(kept), dropped_constant or "none")
    return kept, ranking


def plot_feature_ranking(ranking: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = ranking.head(25).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    colors = np.where(top["selected"], "#e45756", "#bab0ac")
    axes[0].barh(top["description"], top["mutual_information"], color=colors)
    axes[0].set(title="Mutual information with flood", xlabel="MI (nats)")
    axes[1].barh(top["description"], top["rf_importance"], color=colors)
    axes[1].set(title="Random-Forest importance", xlabel="Mean decrease in impurity")
    fig.suptitle("Feature ranking (red = selected for modelling)", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
