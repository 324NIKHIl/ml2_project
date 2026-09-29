import numpy as np
import pandas as pd

from flood_risk.data import integrate_sources, load_sources, preprocess
from flood_risk.data.integration import expand_flood_events
from flood_risk.data.preprocessing import isolated_spike_mask
from flood_risk.features import TARGET, build_features, feature_columns, split_by_time


def _toy_clean(n_days=40):
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    rows = []
    for rid in ["A", "B"]:
        for i, d in enumerate(dates):
            rows.append(dict(region_id=rid, district=rid, state="S", date=d, latitude=0.0, longitude=0.0,
                             rainfall_mm=float(i if rid == "A" else 0), river_level_m=10.0 + i * 0.1,
                             river_discharge_cumecs=100.0, soil_moisture=0.3, temperature_c=25.0,
                             humidity_pct=60.0, elevation_m=10.0, slope_deg=1.0, distance_to_river_km=2.0,
                             urban_fraction=0.1, land_use="agriculture", danger_level_m=12.0,
                             flood=int(rid == "A" and i in (20, 21))))
    return pd.DataFrame(rows)


def test_flood_events_expand_to_daily_labels():
    events = pd.DataFrame({"region_id": ["A"], "start_date": pd.to_datetime(["2020-01-03"]),
                           "end_date": pd.to_datetime(["2020-01-05"])})
    days = expand_flood_events(events)
    assert list(days["date"].dt.day) == [3, 4, 5]
    assert days["flood"].eq(1).all()


def test_isolated_spike_detected_but_flood_wave_kept():
    spike = pd.Series([10, 10, 30, 10, 10], dtype=float)
    wave = pd.Series([10, 20, 40, 60, 55, 45], dtype=float)
    assert isolated_spike_mask(spike).tolist() == [False, False, True, False, False]
    assert not isolated_spike_mask(wave).any()


def test_target_is_next_day_flood_per_region():
    feats = build_features(_toy_clean(), rain_windows=[3, 7], horizon=1)
    a = feats[feats["region_id"] == "A"].set_index("date")
    assert a.loc["2020-01-20", TARGET] == 1        # flood on Jan 21 -> label on Jan 20
    assert a.loc["2020-01-22", TARGET] == 0
    # last day of each region has no future label and is dropped
    assert feats.groupby("region_id")["date"].max().eq(pd.Timestamp("2020-02-08")).all()


def test_rolling_features_use_only_past_and_do_not_mix_regions():
    feats = build_features(_toy_clean(), rain_windows=[3], horizon=1)
    a = feats[feats["region_id"] == "A"].reset_index(drop=True)
    b = feats[feats["region_id"] == "B"].reset_index(drop=True)
    assert a.loc[5, "rain_3d"] == 3 + 4 + 5        # trailing window ending today
    assert (b["rain_3d"] == 0).all()               # region B never sees region A's rain
    assert a.loc[21, "flood_days_prev_365"] == 1   # counts only floods strictly before today


def test_feature_columns_exclude_identifiers_and_target():
    cols = feature_columns(build_features(_toy_clean(), [3], 1))
    for banned in ["region_id", "date", "flood", TARGET, "latitude", "river_level_m", "danger_level_m"]:
        assert banned not in cols


def test_split_is_chronological():
    feats = build_features(_toy_clean(), [3], 1)
    tr, va, te = split_by_time(feats, "2020-01-15", "2020-01-25")
    assert tr["date"].max() < va["date"].min() and va["date"].max() < te["date"].min()


def test_synthetic_sources_integrate_and_clean(small_cfg):
    sources = load_sources(small_cfg)
    merged, report = integrate_sources(sources)
    n_days = len(pd.date_range(small_cfg["data"]["start_date"], small_cfg["data"]["end_date"]))
    assert len(merged) == 34 * n_days                               # full region x day grid, duplicates removed
    assert 0.01 < report["flood_rate"] < 0.08
    clean, prep = preprocess(merged)
    sensors = ["rainfall_mm", "river_level_m", "river_discharge_cumecs", "soil_moisture", "temperature_c", "humidity_pct"]
    assert clean[sensors].isna().sum().sum() == 0
    assert (clean["rainfall_mm"] >= 0).all() and clean["humidity_pct"].le(100).all()
    assert sum(prep["invalid_values"].values()) > 0 and sum(prep["spikes_removed"].values()) > 0
    assert np.isfinite(clean[sensors].to_numpy()).all()
