"""End-to-end pipeline, following the methodology in report section 13:

Data Collection -> Data Integration -> Preprocessing -> EDA -> Feature Engineering
-> Feature Selection -> Model Training -> Probability Prediction -> Risk Classification
-> Evaluation -> Visualization & Early Warning

Artefacts are written to data/, models/ and reports/ (see README).
"""

from __future__ import annotations

import time

import joblib
import pandas as pd

from .data import integrate_sources, load_sources, preprocess
from .eda import run_eda
from .features import TARGET, build_features, feature_columns, plot_feature_ranking, select_features, split_by_time
from .models import MODEL_NAMES, FloodRiskPredictor, evaluate_models, reference_values, train_models
from .risk import generate_alerts, risk_levels
from .utils import get_logger, save_json, write_table

log = get_logger("flood_risk.pipeline")


def _banner(step: int, title: str) -> None:
    log.info("=" * 78)
    log.info("STEP %02d | %s", step, title)
    log.info("=" * 78)


def run_pipeline(cfg: dict, regenerate: bool = False, tune: bool = False, skip_eda: bool = False) -> dict:
    t0 = time.time()
    P = cfg["paths"]
    levels = risk_levels(cfg)
    horizon = cfg["target"]["forecast_horizon_days"]

    # 1-2 ----------------------------------------------------------------------------------------
    _banner(1, "Data collection & integration")
    sources = load_sources(cfg, regenerate=regenerate)
    integrated, integration_report = integrate_sources(sources)
    write_table(integrated, P["interim_dir"] / "integrated.parquet")

    # 3 --------------------------------------------------------------------------------------------
    _banner(2, "Data preprocessing")
    clean, prep_report = preprocess(integrated)
    write_table(clean, P["interim_dir"] / "cleaned.parquet")
    save_json({"integration": integration_report, "preprocessing": prep_report},
              P["reports_dir"] / "data_quality_report.json")

    # 4 --------------------------------------------------------------------------------------------
    if not skip_eda:
        _banner(3, "Exploratory data analysis")
        run_eda(integrated, clean, P["figures_dir"], P["reports_dir"])

    # 5 --------------------------------------------------------------------------------------------
    _banner(4, f"Feature engineering (target = flood in t+{horizon} day)")
    feats = build_features(clean, cfg["features"]["rain_windows"], horizon)
    write_table(feats, P["processed_dir"] / "features.parquet")
    candidates = feature_columns(feats)
    train, valid, test = split_by_time(feats, cfg["split"]["train_end"], cfg["split"]["valid_end"])
    split_info = {name: {"rows": len(d), "start": str(d["date"].min().date()), "end": str(d["date"].max().date()),
                         "flood_rate": float(d[TARGET].mean())}
                  for name, d in [("train", train), ("valid", valid), ("test", test)]}
    log.info("%d candidate features | split: %s", len(candidates),
             {k: f"{v['start']}..{v['end']} ({v['rows']:,} rows, {v['flood_rate']:.2%} floods)" for k, v in split_info.items()})

    _banner(5, "Feature selection")
    if cfg["features"]["selection"]["enabled"]:
        features, ranking = select_features(train, candidates, TARGET, cfg)
        ranking.to_csv(P["reports_dir"] / "feature_ranking.csv", index=False)
        plot_feature_ranking(ranking, P["figures_dir"] / "features_01_ranking.png")
    else:
        features = candidates
    save_json({"selected_features": features, "candidates": candidates}, P["reports_dir"] / "selected_features.json")
    log.info("Selected features: %s", features)

    # 6 --------------------------------------------------------------------------------------------
    _banner(6, "Model training (Logistic Regression, Random Forest, XGBoost)" + (" with tuning" if tune else ""))
    results = train_models(train, valid, features, TARGET, cfg, tune=tune)
    reference = reference_values(train, features)

    bundles = {}
    for name in results["models"]:
        bundles[name] = {
            "model": results["models"][name], "base_model": results["base_models"][name],
            "model_key": name, "model_name": MODEL_NAMES[name], "features": features,
            "reference": reference, "threshold": cfg["models"]["decision_threshold"],
            "levels": levels, "horizon": horizon, "split": split_info,
            "validation": results["validation"][name],
        }
        joblib.dump(bundles[name], P["models_dir"] / f"{name}.joblib")

    # 7-9 -------------------------------------------------------------------------------------------
    _banner(7, "Evaluation on the test period")
    evaluation = evaluate_models(results, test, features, TARGET, cfg, P["figures_dir"])
    best = results["best_model"]
    comparison = evaluation["comparison"]
    bundles[best]["test_metrics"] = comparison.set_index("model").loc[best].drop("model_name").to_dict()
    joblib.dump(bundles[best], P["models_dir"] / "best_model.joblib")

    comparison.to_csv(P["reports_dir"] / "model_comparison.csv", index=False)
    evaluation["risk_table"].to_csv(P["reports_dir"] / "risk_level_validation.csv", index=False)
    evaluation["importance"].to_csv(P["reports_dir"] / "feature_importance.csv", index=False)
    save_json({
        "best_model": best, "best_model_name": MODEL_NAMES[best],
        "selection_metric": results["selection_metric"],
        "decision_threshold": cfg["models"]["decision_threshold"],
        "forecast_horizon_days": horizon, "split": split_info, "features": features,
        "validation": results["validation"], "test": comparison.set_index("model").to_dict(orient="index"),
        "risk_level_validation": evaluation["risk_table"].to_dict(orient="records"),
    }, P["reports_dir"] / "metrics.json")

    # 8, 10 ------------------------------------------------------------------------------------------
    _banner(8, "Risk classification & early-warning alerts (test period)")
    predictor = FloodRiskPredictor(bundles[best])
    preds = predictor.predict(test)
    preds["actual_flood"] = test[TARGET].to_numpy()
    for col in ["rainfall_mm", "rain_7d", "river_above_danger_m", "soil_moisture"]:
        if col in test:
            preds[col] = test[col].to_numpy()
    write_table(preds.drop(columns=["recommendations"]), P["reports_dir"] / "test_predictions.parquet")

    alerts = generate_alerts(preds, levels, cfg["early_warning"]["alert_min_level"])
    alert_cols = ["forecast_date", "region_id", "district", "state", "flood_probability", "risk_level",
                  "actual_flood", "headline", "message", "actions"]
    alerts[alert_cols].to_csv(P["reports_dir"] / "alerts_test_period.csv", index=False)
    hit_rate = alerts["actual_flood"].mean() if len(alerts) else float("nan")
    log.info("%d alerts raised in the test period (%.1f%% followed by an actual flood); "
             "%d of %d actual flood days were alerted.",
             len(alerts), 100 * hit_rate, int(alerts["actual_flood"].sum()), int(preds["actual_flood"].sum()))

    log.info("Pipeline finished in %.1f s. Best model: %s", time.time() - t0, MODEL_NAMES[best])
    return {"comparison": comparison, "best_model": best, "features": features, "alerts": alerts}


def load_test_predictions(cfg: dict) -> pd.DataFrame:
    return pd.read_parquet(cfg["paths"]["reports_dir"] / "test_predictions.parquet")
