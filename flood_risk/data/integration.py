"""Stage 2b: integrate all sources by *location* (region_id) and *time* (date).

* time-series sources are de-duplicated and outer-joined on a complete
  region x day calendar, so gaps show up as explicit missing values;
* static geography is attached per region;
* flood events (start/end ranges) are expanded to a daily 0/1 label.
"""

from __future__ import annotations

import pandas as pd

from ..utils import get_logger

log = get_logger(__name__)

TIME_SERIES_SOURCES = ["rainfall", "river", "soil", "weather"]


def expand_flood_events(events: pd.DataFrame) -> pd.DataFrame:
    """Turn event rows (start..end) into one row per flooded region-day."""
    if events.empty:
        return pd.DataFrame({"region_id": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]"),
                             "flood": pd.Series(dtype=int)})
    days = events.assign(date=[pd.date_range(s, e, freq="D") for s, e in
                               zip(events["start_date"], events["end_date"])])
    days = days.explode("date")[["region_id", "date"]].drop_duplicates()
    days["date"] = pd.to_datetime(days["date"])
    days["flood"] = 1
    return days


def integrate_sources(sources: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    report: dict = {"duplicates_removed": {}}

    start = min(sources[s]["date"].min() for s in TIME_SERIES_SOURCES)
    end = max(sources[s]["date"].max() for s in TIME_SERIES_SOURCES)
    regions = sources["geography"]["region_id"].unique()
    merged = pd.MultiIndex.from_product(
        [regions, pd.date_range(start, end, freq="D")], names=["region_id", "date"]
    ).to_frame(index=False)

    for name in TIME_SERIES_SOURCES:
        df = sources[name]
        before = len(df)
        df = df.drop_duplicates(subset=["region_id", "date"], keep="first")
        report["duplicates_removed"][name] = before - len(df)
        merged = merged.merge(df, on=["region_id", "date"], how="left")

    merged = merged.merge(sources["geography"], on="region_id", how="left")

    labels = expand_flood_events(sources["flood_events"])
    merged = merged.merge(labels, on=["region_id", "date"], how="left")
    merged["flood"] = merged["flood"].fillna(0).astype(int)

    merged = merged.sort_values(["region_id", "date"]).reset_index(drop=True)
    report.update(rows=len(merged), regions=len(regions), start=str(start.date()), end=str(end.date()),
                  flood_days=int(merged["flood"].sum()), flood_rate=float(merged["flood"].mean()))
    log.info("Integrated table: %d rows, %d regions, flood rate %.2f%%, duplicates removed %s",
             len(merged), len(regions), 100 * report["flood_rate"], report["duplicates_removed"])
    return merged, report
