"""Stage 5a: feature engineering.

Every feature for day *t* uses only information available at the end of day *t*
(trailing windows, past labels). The target is a flood on day *t + horizon*, so
the model learns to issue a warning *before* the flood.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.regions import LAND_USE_CATEGORIES

ID_COLUMNS = ["region_id", "district", "state", "date", "latitude", "longitude"]
TARGET = "flood_next"

# Human-readable names, used for plots, the dashboard and alert messages.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "rainfall_mm": "Rainfall today (mm)",
    "rain_max_3d": "Max daily rainfall, last 3 days (mm)",
    "rain_intensity_ratio": "Share of weekly rain that fell today",
    "wet_days_7d": "Rainy days in last 7 days",
    "river_above_danger_m": "River level vs danger level (m)",
    "river_level_change_1d": "River level change, 1 day (m)",
    "river_level_change_3d": "River level change, 3 days (m)",
    "river_discharge_cumecs": "River discharge (m3/s)",
    "discharge_3d_mean": "River discharge, 3-day mean (m3/s)",
    "discharge_change_ratio_3d": "Discharge growth over 3 days (ratio)",
    "soil_moisture": "Soil moisture (fraction)",
    "soil_moisture_change_3d": "Soil moisture change, 3 days",
    "temperature_c": "Temperature (C)",
    "humidity_pct": "Relative humidity (%)",
    "humidity_3d_mean": "Relative humidity, 3-day mean (%)",
    "month_sin": "Season (sin of month)",
    "month_cos": "Season (cos of month)",
    "is_monsoon": "Monsoon season (Jun-Sep)",
    "elevation_m": "Elevation (m)",
    "slope_deg": "Terrain slope (deg)",
    "distance_to_river_km": "Distance to river (km)",
    "urban_fraction": "Urbanised area (fraction)",
    "flood_days_prev_365": "Flood days in the past year",
}


def describe(feature: str) -> str:
    if feature.startswith("rain_") and feature.endswith("d") and feature[5:-1].isdigit():
        return f"Cumulative rainfall, last {feature[5:-1]} days (mm)"
    if feature.startswith("land_use_"):
        return f"Land use: {feature[9:]}"
    return FEATURE_DESCRIPTIONS.get(feature, feature)


def build_features(df: pd.DataFrame, rain_windows: list[int], horizon: int = 1,
                   drop_unlabelled: bool = True) -> pd.DataFrame:
    """Create model features + the ``flood_next`` target from the cleaned table."""
    df = df.sort_values(["region_id", "date"]).reset_index(drop=True).copy()
    g = df.groupby("region_id", sort=False)

    def roll(col: str, window: int, how: str) -> pd.Series:
        r = g[col].rolling(window, min_periods=1)
        return getattr(r, how)().reset_index(level=0, drop=True)

    # --- Rainfall: intensity and accumulation -------------------------------------------------
    for w in rain_windows:
        df[f"rain_{w}d"] = roll("rainfall_mm", w, "sum")
    df["rain_max_3d"] = roll("rainfall_mm", 3, "max")
    weekly = df["rain_7d"] if 7 in rain_windows else roll("rainfall_mm", 7, "sum")
    df["rain_intensity_ratio"] = df["rainfall_mm"] / (weekly + 1.0)
    df["wet_days_7d"] = (df["rainfall_mm"] > 2.5).astype(float).groupby(df["region_id"]).transform(
        lambda s: s.rolling(7, min_periods=1).sum())

    # --- River: level relative to the CWC danger level, and its trend -------------------------
    df["river_above_danger_m"] = df["river_level_m"] - df["danger_level_m"]
    df["river_level_change_1d"] = g["river_level_m"].diff(1).fillna(0)
    df["river_level_change_3d"] = g["river_level_m"].diff(3).fillna(0)
    df["discharge_3d_mean"] = roll("river_discharge_cumecs", 3, "mean")
    df["discharge_change_ratio_3d"] = (df["river_discharge_cumecs"]
                                       / (g["river_discharge_cumecs"].shift(3) + 1.0)).fillna(1.0)

    # --- Soil & weather ------------------------------------------------------------------------
    df["soil_moisture_change_3d"] = g["soil_moisture"].diff(3).fillna(0)
    df["humidity_3d_mean"] = roll("humidity_pct", 3, "mean")

    # --- Season ----------------------------------------------------------------------------------
    month = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["is_monsoon"] = month.between(6, 9).astype(int)

    # --- Historical flood records: how flood-prone has this district been recently? -----------
    df["flood_days_prev_365"] = g["flood"].transform(lambda s: s.shift(1).rolling(365, min_periods=1).sum()).fillna(0)

    # --- Geography: one-hot land use (fixed categories -> stable columns) -----------------------
    for cat in LAND_USE_CATEGORIES:
        df[f"land_use_{cat}"] = (df["land_use"] == cat).astype(int)

    # --- Target: flood on day t + horizon --------------------------------------------------------
    df[TARGET] = g["flood"].shift(-horizon)
    if drop_unlabelled:
        df = df.dropna(subset=[TARGET]).reset_index(drop=True)
        df[TARGET] = df[TARGET].astype(int)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All candidate model inputs (identifiers, raw/absolute stage values and labels excluded)."""
    excluded = set(ID_COLUMNS) | {TARGET, "flood", "land_use", "river_level_m", "danger_level_m"}
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def split_by_time(df: pd.DataFrame, train_end: str, valid_end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train / validation / test split (no shuffling)."""
    train = df[df["date"] <= train_end]
    valid = df[(df["date"] > train_end) & (df["date"] <= valid_end)]
    test = df[df["date"] > valid_end]
    return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)
