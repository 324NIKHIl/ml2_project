"""Run the complete flood-risk pipeline: data -> EDA -> features -> models -> evaluation -> alerts.

Usage (from the project root):
    python scripts/run_pipeline.py                 # full run
    python scripts/run_pipeline.py --regenerate    # re-simulate the raw synthetic data first
    python scripts/run_pipeline.py --tune          # randomized hyper-parameter search (slower)
    python scripts/run_pipeline.py --skip-eda      # skip the EDA figures
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flood_risk.config import load_config  # noqa: E402
from flood_risk.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="path to a config YAML (default: config/config.yaml)")
    parser.add_argument("--regenerate", action="store_true", help="regenerate synthetic raw data")
    parser.add_argument("--tune", action="store_true", help="hyper-parameter tuning with TimeSeriesSplit")
    parser.add_argument("--skip-eda", action="store_true", help="skip exploratory data analysis")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = run_pipeline(cfg, regenerate=args.regenerate, tune=args.tune, skip_eda=args.skip_eda)

    cols = ["model_name", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]
    print("\nModel comparison on the test period:")
    print(result["comparison"][cols].round(4).to_string(index=False))
    print(f"\nBest model -> models/best_model.joblib ({result['best_model']})")
    print("Launch the dashboard with:  streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
