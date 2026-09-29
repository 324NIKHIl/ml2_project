"""Command-line flood-risk forecast and early-warning report for one date.

Usage (from the project root, after running the pipeline):
    python scripts/predict.py --date 2023-07-20
    python scripts/predict.py --date 2023-07-20 --region R05
    python scripts/predict.py --date 2023-07-20 --alerts-only --save reports/alerts_2023-07-20.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from flood_risk.config import load_config  # noqa: E402
from flood_risk.models import FloodRiskPredictor  # noqa: E402
from flood_risk.risk import generate_alerts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", required=True, help="observation date YYYY-MM-DD (forecast is for the next day)")
    parser.add_argument("--region", default=None, help="region id (e.g. R05) or district name")
    parser.add_argument("--alerts-only", action="store_true", help="only list districts at High/Severe risk")
    parser.add_argument("--top-factors", type=int, default=3, help="contributing factors shown per district")
    parser.add_argument("--save", default=None, help="optional CSV output path")
    args = parser.parse_args()

    cfg = load_config()
    predictor = FloodRiskPredictor.load(cfg["paths"]["models_dir"] / "best_model.joblib")
    feats = pd.read_parquet(cfg["paths"]["processed_dir"] / "features.parquet")

    date = pd.Timestamp(args.date)
    rows = feats[feats["date"] == date]
    if args.region:
        key = args.region.lower()
        rows = rows[(rows["region_id"].str.lower() == key) | (rows["district"].str.lower() == key)]
    if rows.empty:
        sys.exit(f"No data for date={args.date} region={args.region}. "
                 f"Available dates: {feats['date'].min().date()} .. {feats['date'].max().date()}")
    if date <= pd.Timestamp(cfg["split"]["valid_end"]):
        print(f"Note: {date.date()} falls inside the training/validation period - results are in-sample.\n")

    preds = predictor.predict(rows)
    tables = predictor.explain(rows, top_n=args.top_factors)
    preds["top_factors"] = [FloodRiskPredictor.factor_strings(t) for t in tables]
    preds["actual_flood"] = rows["flood_next"].to_numpy()

    out = generate_alerts(preds, predictor.levels, cfg["early_warning"]["alert_min_level"]) if args.alerts_only else \
        preds.sort_values("flood_probability", ascending=False)

    print(f"Flood-risk forecast for {(date + pd.Timedelta(days=predictor.horizon)).date()} "
          f"(conditions observed {date.date()}, model: {predictor.model_name})\n")
    if out.empty:
        print("No district at High or Severe risk.")
    for r in out.itertuples():
        truth = "flood occurred" if r.actual_flood else "no flood"
        print(f"[{r.risk_level:<8}] {r.flood_probability:6.1%}  {r.district} ({r.state})  - ground truth: {truth}")
        for f in r.top_factors:
            print(f"             ^ {f}")
    if args.save:
        cols = ["forecast_date", "region_id", "district", "state", "flood_probability", "risk_level", "actual_flood"]
        out.assign(top_factors=out["top_factors"].map("; ".join))[cols + ["top_factors"]].to_csv(args.save, index=False)
        print(f"\nSaved -> {args.save}")


if __name__ == "__main__":
    main()
