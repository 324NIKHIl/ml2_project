"""Stage 1-2a: collect the raw source tables.

With ``data.source: synthetic`` the tables are simulated (and cached in
``data/raw``); with ``data.source: local`` your own CSVs in ``data/raw`` are read.
Either way every table is validated against the schema below, so the rest of the
pipeline never needs to know where the data came from.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..utils import get_logger
from .synthetic import RAW_FILES, generate_synthetic_sources, write_sources

log = get_logger(__name__)

SCHEMA: dict[str, list[str]] = {
    "rainfall": ["region_id", "date", "rainfall_mm"],
    "river": ["region_id", "date", "river_level_m", "river_discharge_cumecs"],
    "soil": ["region_id", "date", "soil_moisture"],
    "weather": ["region_id", "date", "temperature_c", "humidity_pct"],
    "geography": ["region_id", "district", "state", "latitude", "longitude", "elevation_m",
                  "slope_deg", "distance_to_river_km", "urban_fraction", "land_use", "danger_level_m"],
    "flood_events": ["event_id", "region_id", "start_date", "end_date", "severity"],
}
DATE_COLUMNS = {"flood_events": ["start_date", "end_date"], "geography": []}


def _validate(name: str, df: pd.DataFrame) -> pd.DataFrame:
    missing = set(SCHEMA[name]) - set(df.columns)
    if missing:
        raise ValueError(f"Source '{name}' is missing required columns: {sorted(missing)}")
    for col in DATE_COLUMNS.get(name, ["date"]):
        df[col] = pd.to_datetime(df[col])
    df["region_id"] = df["region_id"].astype(str)
    return df


def load_sources(cfg: dict, regenerate: bool = False) -> dict[str, pd.DataFrame]:
    raw_dir: Path = cfg["paths"]["raw_dir"]
    paths = {name: raw_dir / fname for name, fname in RAW_FILES.items()}
    source = cfg["data"]["source"]

    if source == "synthetic":
        if regenerate or not all(p.exists() for p in paths.values()):
            log.info("Generating synthetic source data -> %s", raw_dir)
            write_sources(generate_synthetic_sources(cfg), raw_dir)
    elif source == "local":
        absent = [p.name for p in paths.values() if not p.exists()]
        if absent:
            raise FileNotFoundError(f"data.source=local but these files are missing in {raw_dir}: {absent}")
    else:
        raise ValueError(f"Unknown data.source '{source}' (expected 'synthetic' or 'local')")

    sources = {name: _validate(name, pd.read_csv(path)) for name, path in paths.items()}
    for name, df in sources.items():
        log.info("Loaded %-13s %8d rows x %2d cols", name, *df.shape)
    return sources
