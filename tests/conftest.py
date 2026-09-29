import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flood_risk.config import load_config  # noqa: E402


@pytest.fixture
def small_cfg(tmp_path):
    """Config pointing at a temp folder with a short simulation period (fast tests)."""
    cfg = load_config(root=tmp_path)
    cfg["data"]["start_date"] = "2016-01-01"
    cfg["data"]["end_date"] = "2019-12-31"
    cfg["split"]["train_end"] = "2017-12-31"
    cfg["split"]["valid_end"] = "2018-12-31"
    cfg["features"]["selection"]["sample_size"] = 20000
    cfg["models"]["random_forest"]["n_estimators"] = 40
    cfg["models"]["xgboost"]["n_estimators"] = 80
    return cfg
