"""Synthetic generator for the six data sources in the system architecture.

Real historical sources (IMD rainfall, CWC river gauges, NASA SMAP soil moisture,
ERA5 reanalysis, SRTM elevation, government flood records) need registrations,
API keys or manual downloads. So that the project runs end-to-end out of the box,
this module simulates them with simple physically based models:

* Rainfall      - wet/dry Markov chain + gamma-distributed amounts, monthly
                  climatology per monsoon regime, year-to-year variability and
                  injected extreme-rainfall spells (depressions / cloudbursts).
* Weather       - seasonal temperature cycle with rain cooling; humidity driven
                  by the monsoon and rain days.
* Soil moisture - single-bucket water balance (infiltration, evapotranspiration,
                  drainage).
* River         - linear-reservoir catchment storage -> stage (m) via a power
                  law, discharge via a rating curve; CWC-style danger level.
* Floods        - latent hazard that combines riverine, pluvial (urban) and
                  flash-flood mechanisms, soil saturation and geography.
                  Consecutive flood days are stored as flood *events*, the way
                  historical flood records are usually published.

After simulation, the sensor tables are corrupted on purpose (missing values,
sentinel codes, spikes, duplicate rows) so that the preprocessing stage has real
work to do. Each table is written to ``data/raw`` in the same schema a real
source would use (see docs/data_schema.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import get_logger
from .regions import HIDDEN_COLUMNS, RAIN_PROFILES, get_regions

log = get_logger(__name__)

RAW_FILES = {
    "rainfall": "rainfall_imd.csv",
    "river": "river_cwc.csv",
    "soil": "soil_moisture_nasa.csv",
    "weather": "weather_era5.csv",
    "geography": "geography_srtm.csv",
    "flood_events": "flood_events.csv",
}


def _rolling_sum(a: np.ndarray, w: int) -> np.ndarray:
    """Trailing rolling sum along axis 0 (time) of a (days, regions) array."""
    c = np.cumsum(a, axis=0)
    out = c.copy()
    out[w:] = c[w:] - c[:-w]
    return out


def _simulate_rainfall(rng, regions, dates) -> np.ndarray:
    n_d, n_r = len(dates), len(regions)
    month = dates.month.values - 1
    profiles = np.array([RAIN_PROFILES[p] for p in regions["rain_profile"]])
    profiles = profiles / profiles.sum(axis=1, keepdims=True)          # (n_r, 12)

    years = dates.year.values
    uniq_years = np.unique(years)
    year_factor = rng.lognormal(0.0, 0.22, size=(len(uniq_years), n_r))
    yf = year_factor[np.searchsorted(uniq_years, years)]                # (n_d, n_r)

    w = profiles[:, month].T                                            # (n_d, n_r)
    p_wet = np.clip(0.04 + 2.4 * w, 0.03, 0.85)
    monthly_total = regions["annual_rain_mm"].values[None, :] * w * yf
    mean_wet_amount = monthly_total / (dates.days_in_month.values[:, None] * p_wet)

    # Wet/dry persistence: P(wet | wet) > P(wet | dry) while keeping P(wet) = p_wet.
    wet = np.zeros((n_d, n_r), dtype=bool)
    u = rng.random((n_d, n_r))
    wet[0] = u[0] < p_wet[0]
    for t in range(1, n_d):
        p = p_wet[t]
        p_ww = p + 0.3 * (1 - p)
        p_dw = 0.7 * p
        wet[t] = np.where(wet[t - 1], u[t] < p_ww, u[t] < p_dw)

    shape = 0.75
    rain = np.where(wet, rng.gamma(shape, mean_wet_amount / shape), 0.0)

    # Extreme spells (monsoon depressions, cyclones, cloudbursts): 2-4 day bursts.
    start_prob = 0.0035 * (w * 12)
    starts = np.argwhere(rng.random((n_d, n_r)) < start_prob)
    for t, r in starts:
        length = rng.integers(2, 5)
        factor = rng.uniform(2.5, 5.0)
        base = max(mean_wet_amount[t, r], 5.0)
        rain[t:t + length, r] = np.maximum(rain[t:t + length, r], base) * factor
    return rain


def _simulate_weather(rng, regions, dates, rain):
    n_d, n_r = rain.shape
    doy = dates.dayofyear.values[:, None]
    lat = regions["latitude"].values[None, :]
    elev = regions["elevation_m"].values[None, :]

    t_mean = 28.0 - 0.0065 * elev - 0.25 * np.maximum(lat - 15, 0)
    amplitude = 2.5 + 0.35 * np.maximum(lat - 10, 0)
    seasonal = amplitude * np.cos(2 * np.pi * (doy - 150) / 365.25)
    wet = rain > 1.0
    temp = t_mean + seasonal - 1.8 * wet - 0.02 * np.minimum(rain, 150) + rng.normal(0, 1.2, (n_d, n_r))

    profiles = np.array([RAIN_PROFILES[p] for p in regions["rain_profile"]])
    monsoonality = (profiles / profiles.sum(1, keepdims=True))[:, dates.month.values - 1].T * 12
    coastal = (elev < 20).astype(float) * 8
    arid = (regions["land_use"].values[None, :] == "arid") * 15
    humidity = 42 + 14 * monsoonality + 12 * wet + coastal - arid + rng.normal(0, 6, (n_d, n_r))
    return temp, np.clip(humidity, 8, 100)


def _simulate_soil(regions, rain, temp) -> tuple[np.ndarray, np.ndarray]:
    n_d, n_r = rain.shape
    porosity = np.select(
        [regions["land_use"].values == "wetland", regions["land_use"].values == "arid"],
        [0.50, 0.35], 0.45,
    )
    sm = np.zeros((n_d, n_r))
    s = porosity * 0.4
    for t in range(n_d):
        infiltration = 0.9 * (rain[t] / (rain[t] + 40.0)) * (porosity - s)
        et = 0.02 * (np.maximum(temp[t], 0) / 30.0) * s
        drainage = 0.01 * s
        s = np.clip(s + infiltration - et - drainage, 0.03, porosity)
        sm[t] = s
    return sm, porosity


def _simulate_river(rng, regions, rain, sm, porosity):
    n_d, n_r = rain.shape
    catch = regions["catchment_factor"].values
    urban = regions["urban_fraction"].values
    runoff_coef = 0.15 + 0.85 * (sm / porosity) ** 2 + 0.3 * urban

    storage = np.zeros((n_d, n_r))
    upstream = np.zeros(n_r)
    s = np.zeros(n_r)
    for t in range(n_d):
        # Upstream catchment rain arrives smoothed and delayed.
        upstream = 0.75 * upstream + 0.25 * rain[t] * rng.lognormal(0, 0.3, n_r)
        s = 0.85 * s + catch * (runoff_coef[t] * rain[t] + 0.6 * upstream)
        storage[t] = s

    scale = np.quantile(storage, 0.95, axis=0) + 1e-6
    rise = 5.5 * (storage / scale) ** 0.8                               # metres above base stage
    base_stage = rng.uniform(40, 120, n_r) + regions["elevation_m"].values * 0.9
    level = base_stage + rise + rng.normal(0, 0.05, (n_d, n_r))
    discharge = 180 * catch * (rise + 0.3) ** 1.7 * rng.lognormal(0, 0.05, (n_d, n_r))

    # CWC-style danger level: exceeded more often on large flood-prone rivers.
    exceed = np.clip(0.004 + 0.011 * catch ** 2, 0.002, 0.05)
    danger = np.array([np.quantile(level[:, r], 1 - exceed[r]) for r in range(n_r)])
    return level, discharge, danger


def _simulate_floods(rng, regions, rain, sm, porosity, level, danger, target_rate):
    n_d, n_r = rain.shape
    urban = regions["urban_fraction"].values
    slope = regions["slope_deg"].values
    elev = regions["elevation_m"].values
    dist = regions["distance_to_river_km"].values

    r3 = _rolling_sum(rain, 3)
    h_river = 1.6 * (level - danger)                                 # riverine
    h_urban = urban * (rain - 90.0) / 22.0                           # pluvial / drainage overload
    h_flash = (slope / 20.0) * (r3 - 160.0) / 30.0                    # flash floods in hills
    mechanism = np.logaddexp(np.logaddexp(h_river, h_urban), h_flash)

    eps = np.zeros((n_d, n_r))
    for t in range(1, n_d):
        eps[t] = 0.7 * eps[t - 1] + rng.normal(0, 0.45, n_r)

    z = (mechanism
         + 2.0 * (sm / porosity - 0.75)
         + 0.35 * (np.log1p(r3) - 3.0)
         - elev / 900.0
         - dist / 18.0
         + eps)
    # Choose the intercept so that the overall share of flood days matches the target.
    b0 = -np.quantile(z, 1 - target_rate)
    z = z + b0
    return z > 0, z


def _events_from_daily(flood: np.ndarray, z: np.ndarray, regions, dates) -> pd.DataFrame:
    rows = []
    for r, region_id in enumerate(regions["region_id"]):
        f = flood[:, r].astype(int)
        edges = np.diff(np.concatenate([[0], f, [0]]))
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0] - 1
        for s, e in zip(starts, ends):
            peak = z[s:e + 1, r].max()
            severity = "Minor" if peak < 1.0 else "Moderate" if peak < 2.5 else "Major"
            rows.append((region_id, dates[s].date(), dates[e].date(), e - s + 1, severity))
    events = pd.DataFrame(rows, columns=["region_id", "start_date", "end_date", "duration_days", "severity"])
    events.insert(0, "event_id", [f"EV{i:05d}" for i in range(1, len(events) + 1)])
    events["source"] = "synthetic-govt-flood-records"
    return events


def _long(arr: np.ndarray, name: str) -> pd.Series:
    return pd.Series(arr.T.ravel(), name=name)


def _corrupt(df: pd.DataFrame, cols: list[str], rng, missing_rate: float, outlier_rate: float,
             duplicate_rate: float, kind: str) -> pd.DataFrame:
    """Inject the data-quality problems typical of real sensor archives."""
    df = df.copy()
    n = len(df)
    for c in cols:
        df.loc[rng.random(n) < missing_rate, c] = np.nan
        bad = rng.random(n) < outlier_rate
        if kind == "rainfall":
            df.loc[bad, c] = -999.0                                 # IMD-style missing sentinel
        elif kind == "river":
            df.loc[bad, c] = df.loc[bad, c] * rng.uniform(1.8, 3.0, bad.sum())   # gauge spikes
        elif kind == "soil":
            df.loc[bad, c] = rng.choice([-1.0, 9.99], bad.sum())    # invalid retrievals
        elif kind == "weather":
            if c == "humidity_pct":
                df.loc[bad, c] = rng.uniform(101, 140, bad.sum())
            else:
                df.loc[bad, c] = rng.choice([-99.0, 70.0], bad.sum())
    dups = df.sample(frac=duplicate_rate, random_state=int(rng.integers(1e9)))
    return pd.concat([df, dups]).sort_values(["region_id", "date"]).reset_index(drop=True)


def generate_synthetic_sources(cfg: dict) -> dict[str, pd.DataFrame]:
    """Simulate every raw source table. Returns ``{source_name: DataFrame}``."""
    syn = cfg["data"]["synthetic"]
    rng = np.random.default_rng(cfg["project"]["random_seed"])
    regions = get_regions()
    dates = pd.date_range(cfg["data"]["start_date"], cfg["data"]["end_date"], freq="D")
    log.info("Simulating %d regions x %d days (%s .. %s)", len(regions), len(dates),
             dates[0].date(), dates[-1].date())

    rain = _simulate_rainfall(rng, regions, dates)
    temp, humidity = _simulate_weather(rng, regions, dates, rain)
    sm, porosity = _simulate_soil(regions, rain, temp)
    level, discharge, danger = _simulate_river(rng, regions, rain, sm, porosity)
    flood, z = _simulate_floods(rng, regions, rain, sm, porosity, level, danger, syn["target_flood_rate"])

    keys = pd.DataFrame({
        "region_id": np.repeat(regions["region_id"].values, len(dates)),
        "date": np.tile(dates.values, len(regions)),
    })

    def table(**cols):
        return pd.concat([keys] + [_long(a, n) for n, a in cols.items()], axis=1)

    corrupt = dict(rng=rng, missing_rate=syn["missing_rate"], outlier_rate=syn["outlier_rate"],
                   duplicate_rate=syn["duplicate_rate"])
    rainfall = _corrupt(table(rainfall_mm=rain.round(1)), ["rainfall_mm"], kind="rainfall", **corrupt)
    river = _corrupt(table(river_level_m=level.round(2), river_discharge_cumecs=discharge.round(1)),
                     ["river_level_m", "river_discharge_cumecs"], kind="river", **corrupt)
    soil = _corrupt(table(soil_moisture=sm.round(4)), ["soil_moisture"], kind="soil", **corrupt)
    weather = _corrupt(table(temperature_c=temp.round(1), humidity_pct=humidity.round(1)),
                       ["temperature_c", "humidity_pct"], kind="weather", **corrupt)

    geography = regions.drop(columns=HIDDEN_COLUMNS).copy()
    geography["danger_level_m"] = danger.round(2)
    events = _events_from_daily(flood, z, regions, dates)

    log.info("Flood days: %d (%.2f%% of region-days) in %d events",
             flood.sum(), 100 * flood.mean(), len(events))
    return {"rainfall": rainfall, "river": river, "soil": soil, "weather": weather,
            "geography": geography, "flood_events": events}


def write_sources(sources: dict[str, pd.DataFrame], raw_dir: Path) -> None:
    for name, df in sources.items():
        path = Path(raw_dir) / RAW_FILES[name]
        df.to_csv(path, index=False)
        log.info("Wrote %-13s -> %s (%d rows)", name, path.name, len(df))
