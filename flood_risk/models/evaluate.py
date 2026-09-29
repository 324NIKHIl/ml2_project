"""Stage 9: model evaluation on the untouched test period (report section 17).

Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, plus the
probabilistic checks the report calls for - PR-AUC (better suited to rare
events), Brier score and a reliability (calibration) curve. The risk framework
is validated too: the observed flood frequency inside each risk level should
rise from Low to Severe.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)

from ..features.engineering import describe  # noqa: E402
from ..risk.classification import classify_probability, level_colors, risk_levels  # noqa: E402
from ..utils import get_logger  # noqa: E402
from .registry import MODEL_NAMES  # noqa: E402

log = get_logger(__name__)
COLORS = {"logistic_regression": "#4c78a8", "random_forest": "#54a24b", "xgboost": "#e45756"}


def compute_metrics(y_true, prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    pred = (np.asarray(prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob),
        "pr_auc": average_precision_score(y_true, prob),
        "brier": brier_score_loss(y_true, prob),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def risk_level_table(y_true, prob, cfg: dict) -> pd.DataFrame:
    levels = risk_levels(cfg)
    df = pd.DataFrame({"risk_level": classify_probability(prob, levels), "flood": np.asarray(y_true), "prob": prob})
    table = df.groupby("risk_level").agg(region_days=("flood", "size"), actual_floods=("flood", "sum"),
                                         mean_predicted_probability=("prob", "mean"),
                                         observed_flood_rate=("flood", "mean"))
    table = table.reindex([lv.name for lv in levels]).fillna(0)
    table["share_of_all_floods"] = table["actual_floods"] / max(table["actual_floods"].sum(), 1)
    return table.reset_index()


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def evaluate_models(results: dict, test: pd.DataFrame, features: list[str], target: str,
                    cfg: dict, figures_dir: Path) -> dict:
    figures_dir = Path(figures_dir)
    threshold = cfg["models"]["decision_threshold"]
    X, y = test[features], test[target].to_numpy()
    probs = {name: m.predict_proba(X)[:, 1] for name, m in results["models"].items()}

    rows = []
    for name, p in probs.items():
        rows.append({"model": name, "model_name": MODEL_NAMES[name], **compute_metrics(y, p, threshold),
                     "valid_pr_auc": results["validation"][name]["pr_auc"]})
        log.info("TEST %-20s acc %.3f | prec %.3f | rec %.3f | F1 %.3f | ROC-AUC %.3f | PR-AUC %.3f | Brier %.4f",
                 MODEL_NAMES[name], rows[-1]["accuracy"], rows[-1]["precision"], rows[-1]["recall"],
                 rows[-1]["f1"], rows[-1]["roc_auc"], rows[-1]["pr_auc"], rows[-1]["brier"])
    comparison = pd.DataFrame(rows)

    # ---------- ROC & PR curves ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        ax1.plot(fpr, tpr, color=COLORS[name], label=f"{MODEL_NAMES[name]} (AUC={roc_auc_score(y, p):.3f})")
        ax2.plot(rec, prec, color=COLORS[name], label=f"{MODEL_NAMES[name]} (AP={average_precision_score(y, p):.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    ax1.set(title="ROC curve (test period)", xlabel="False positive rate", ylabel="True positive rate")
    ax2.axhline(y.mean(), color="k", ls="--", lw=1, label=f"No-skill ({y.mean():.3f})")
    ax2.set(title="Precision-Recall curve (test period)", xlabel="Recall", ylabel="Precision")
    ax1.legend(loc="lower right")
    ax2.legend(loc="upper right")
    _save(fig, figures_dir / "eval_01_roc_pr_curves.png")

    # ---------- confusion matrices ----------
    fig, axes = plt.subplots(1, len(probs), figsize=(5 * len(probs), 4.3))
    for ax, (name, p) in zip(np.atleast_1d(axes), probs.items()):
        ConfusionMatrixDisplay(confusion_matrix(y, (p >= threshold).astype(int), labels=[0, 1]),
                               display_labels=["No flood", "Flood"]).plot(ax=ax, cmap="Blues", colorbar=False, values_format=",")
        ax.set_title(f"{MODEL_NAMES[name]} (threshold {threshold})")
        ax.grid(False)
    _save(fig, figures_dir / "eval_02_confusion_matrices.png")

    # ---------- calibration ----------
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
    for name, p in probs.items():
        frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", color=COLORS[name], label=MODEL_NAMES[name])
    ax.set(title="Reliability diagram (test period)", xlabel="Predicted flood probability",
           ylabel="Observed flood frequency")
    ax.legend()
    _save(fig, figures_dir / "eval_03_calibration.png")

    # ---------- best model: risk-level validation & permutation importance ----------
    best = results["best_model"]
    risk_table = risk_level_table(y, probs[best], cfg)
    colors = level_colors(risk_levels(cfg))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.bar(risk_table["risk_level"], risk_table["region_days"], color=[colors[lv] for lv in risk_table["risk_level"]])
    ax1.set(title=f"Test region-days per risk level ({MODEL_NAMES[best]})", ylabel="Region-days", yscale="log")
    ax2.bar(risk_table["risk_level"], risk_table["observed_flood_rate"] * 100,
            color=[colors[lv] for lv in risk_table["risk_level"]])
    for i, v in enumerate(risk_table["observed_flood_rate"]):
        ax2.text(i, v * 100, f"{v:.1%}", ha="center", va="bottom")
    ax2.set(title="Observed flood frequency within each risk level", ylabel="Floods (%)", ylim=(0, 105))
    _save(fig, figures_dir / "eval_04_risk_level_validation.png")

    sample = test.sample(min(len(test), 20000), random_state=cfg["project"]["random_seed"])
    perm = permutation_importance(results["models"][best], sample[features], sample[target],
                                  scoring="average_precision", n_repeats=3,
                                  random_state=cfg["project"]["random_seed"], n_jobs=1)
    importance = pd.DataFrame({"feature": features, "description": [describe(f) for f in features],
                               "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}
                              ).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    top = importance.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["description"], top["importance_mean"], xerr=top["importance_std"], color="#e45756")
    ax.set(title=f"Most important flood-risk factors ({MODEL_NAMES[best]}, permutation importance)",
           xlabel="Drop in PR-AUC when the feature is shuffled")
    _save(fig, figures_dir / "eval_05_feature_importance.png")

    return {"comparison": comparison, "risk_table": risk_table, "importance": importance, "test_probs": probs}
