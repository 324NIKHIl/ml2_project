"""Stage 3: data preprocessing.

1. Invalid values -> missing: sentinel codes (-999), physically impossible
   readings (humidity > 100 %, negative soil moisture, ...).
2. Outlier removal: gauge spikes in river level / discharge are detected as
   *isolated* jumps (a reading more than 1.5x above / below BOTH neighbouring
   days) and set to missing; discharge jumps that are backed by a matching
   change in river stage are kept. Genuine flood waves rise and recede over several
   days, so they are kept - a global z-score or IQR rule would wrongly delete
   exactly the extreme readings the model needs.
3. Missing-value imputation, region by region:
   * continuous sensors - time interpolation over gaps of up to 3 days, then
     the region x calendar-month median;
   * rainfall - region x calendar-month median (rain cannot be interpolated).

Normalisation / standardisation is deliberately *not* done here. It is done
inside the model pipelines and fitted on the training split only, which avoids
leaking test-set statistics into training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import get_logger

log = get_logger(__name__)

VALID_RANGES: dict[str, tuple[float, float]] = {
    "rainfall_mm": (0.0, 1000.0),
    "river_level_m": (0.0, 9000.0),
    "river_discharge_cumecs": (0.0, 1e6),
    "soil_moisture": (0.0, 0.6),
    "temperature_c": (-30.0, 55.0),
    "humidity_pct": (0.0, 100.0),
}
SPIKE_COLUMNS = ["river_level_m", "river_discharge_cumecs"]
STAGE_CHANGE_TOLERANCE_M = 0.1
INTERPOLATE_COLUMNS = ["river_level_m", "river_discharge_cumecs", "soil_moisture", "temperature_c", "humidity_pct"]


def isolated_spike_mask(s: pd.Series, ratio: float = 1.5) -> pd.Series:
    """True where a reading is ``ratio`` times above (or below) both its neighbours."""
    neighbours = pd.concat([s.shift(1), s.shift(-1)], axis=1)
    # both neighbours required: never delete a reading without evidence on both sides
    hi, lo = neighbours.max(axis=1, skipna=False), neighbours.min(axis=1, skipna=False)
    return ((s > ratio * hi) | (s * ratio < lo)).fillna(False)


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.sort_values(["region_id", "date"]).reset_index(drop=True).copy()
    report: dict = {"missing_before": {}, "invalid_values": {}, "spikes_removed": {}, "missing_after": {}}
    sensor_cols = list(VALID_RANGES)

    for c in sensor_cols:
        report["missing_before"][c] = int(df[c].isna().sum())

    # 1. invalid / sentinel values
    for c, (lo, hi) in VALID_RANGES.items():
        bad = (df[c] < lo) | (df[c] > hi)
        report["invalid_values"][c] = int(bad.sum())
        df.loc[bad, c] = np.nan

    # 2. gauge spikes
    stage = df.groupby("region_id")["river_level_m"]
    stage_change = pd.concat([stage.diff().abs(), stage.diff(-1).abs()], axis=1).min(axis=1)
    for c in SPIKE_COLUMNS:
        mask = df.groupby("region_id", group_keys=False)[c].apply(isolated_spike_mask).reindex(df.index)
        if c == "river_discharge_cumecs":
            # Stage-discharge consistency: a genuine discharge jump is accompanied by a
            # change in water level; a jump with a flat stage is a sensor error.
            mask &= ~(stage_change > STAGE_CHANGE_TOLERANCE_M)
        report["spikes_removed"][c] = int(mask.sum())
        df.loc[mask, c] = np.nan

    # 3. imputation
    df = df.set_index("date")
    for c in INTERPOLATE_COLUMNS:
        df[c] = df.groupby("region_id")[c].transform(lambda s: s.interpolate(method="time", limit=3))
    df = df.reset_index()
    month = df["date"].dt.month
    for c in sensor_cols:
        df[c] = df[c].fillna(df.groupby([df["region_id"], month])[c].transform("median"))
        # last resort: a region with no readings at all in that calendar month
        df[c] = df[c].fillna(df.groupby("region_id")[c].transform("median"))
        report["missing_after"][c] = int(df[c].isna().sum())

    log.info("Preprocessing: %d invalid values, %d gauge spikes removed, %d missing values imputed",
             sum(report["invalid_values"].values()), sum(report["spikes_removed"].values()),
             sum(report["missing_before"].values()) + sum(report["invalid_values"].values())
             + sum(report["spikes_removed"].values()))
    return df, report
