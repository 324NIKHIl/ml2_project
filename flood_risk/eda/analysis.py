"""Stage 4: exploratory data analysis (report section 12).

Every figure is saved to ``reports/figures/eda_*.png`` and a machine-readable
summary to ``reports/eda_summary.json``:

* class balance (flood vs non-flood)                      -> class imbalance
* missing values per source column (before cleaning)      -> data quality
* distributions of rainfall & environmental variables
* flood vs non-flood comparison of key variables
* correlation matrix of weather / hydrological variables
* regional flood frequency (which districts flood most)
* temporal patterns: monthly seasonality and yearly trend
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from ..utils import get_logger, save_json  # noqa: E402

log = get_logger(__name__)

SENSOR_COLUMNS = ["rainfall_mm", "river_level_m", "river_discharge_cumecs",
                  "soil_moisture", "temperature_c", "humidity_pct"]
PALETTE = {0: "#4c78a8", 1: "#e45756"}
sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _with_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few derived columns that make the EDA plots more informative."""
    out = df.copy()
    g = out.groupby("region_id")["rainfall_mm"]
    out["rain_7d_mm"] = g.transform(lambda s: s.rolling(7, min_periods=1).sum())
    out["river_above_danger_m"] = out["river_level_m"] - out["danger_level_m"]
    out["label"] = out["flood"].map({0: "No flood", 1: "Flood"})
    return out


def run_eda(raw_integrated: pd.DataFrame, clean: pd.DataFrame, figures_dir: Path, reports_dir: Path) -> dict:
    figures_dir, reports_dir = Path(figures_dir), Path(reports_dir)
    df = _with_context(clean)
    summary: dict = {}

    # 1. class balance
    counts = df["label"].value_counts()
    summary["class_balance"] = counts.to_dict()
    summary["flood_rate"] = float(df["flood"].mean())
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index,
                palette=["#4c78a8", "#e45756"], legend=False, ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / counts.sum():.1%})", ha="center", va="bottom")
    ax.set(title="Class balance (region-days)", xlabel="", ylabel="Count")
    ax.set_ylim(0, counts.max() * 1.15)
    _save(fig, figures_dir / "eda_01_class_balance.png")

    # 2. missing values before cleaning
    missing = raw_integrated[SENSOR_COLUMNS].isna().mean().sort_values() * 100
    summary["missing_pct_before_cleaning"] = missing.round(3).to_dict()
    fig, ax = plt.subplots(figsize=(7, 4))
    missing.plot.barh(ax=ax, color="#72b7b2")
    ax.set(title="Missing values per variable (before preprocessing)", xlabel="% of rows missing")
    _save(fig, figures_dir / "eda_02_missing_values.png")

    # 3. distributions
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.ravel(), SENSOR_COLUMNS):
        data = df[col]
        if col in ("rainfall_mm", "river_discharge_cumecs"):
            data = np.log1p(data)
            ax.set_xlabel(f"log(1 + {col})")
        else:
            ax.set_xlabel(col)
        sns.histplot(data, bins=50, ax=ax, color="#4c78a8")
        ax.set_title(col)
    fig.suptitle("Distribution of weather & environmental variables", fontsize=14)
    _save(fig, figures_dir / "eda_03_distributions.png")
    summary["variable_stats"] = df[SENSOR_COLUMNS].describe().round(3).to_dict()

    # 4. flood vs non-flood comparison
    compare_cols = ["rainfall_mm", "rain_7d_mm", "river_above_danger_m", "soil_moisture", "humidity_pct", "temperature_c"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    sample = pd.concat([g.sample(min(len(g), 15000), random_state=0) for _, g in df.groupby("flood")])
    for ax, col in zip(axes.ravel(), compare_cols):
        y = np.log1p(sample[col]) if col.startswith("rain") else sample[col]
        sns.boxplot(x=sample["label"], y=y, hue=sample["label"], ax=ax, showfliers=False,
                    palette={"No flood": "#4c78a8", "Flood": "#e45756"}, legend=False)
        ax.set(title=col, xlabel="", ylabel=f"log(1 + {col})" if col.startswith("rain") else col)
    fig.suptitle("Conditions during flood vs non-flood days", fontsize=14)
    _save(fig, figures_dir / "eda_04_flood_vs_nonflood.png")
    summary["flood_vs_nonflood_means"] = df.groupby("label")[compare_cols].mean().round(3).to_dict()

    # 5. correlation matrix
    corr_cols = compare_cols + ["river_discharge_cumecs", "elevation_m", "urban_fraction", "flood"]
    corr = df[corr_cols].corr()
    summary["correlation_with_flood"] = corr["flood"].drop("flood").sort_values(ascending=False).round(3).to_dict()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
                annot_kws={"size": 8}, square=True, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation matrix")
    _save(fig, figures_dir / "eda_05_correlation.png")

    # 6. regional flood frequency
    years = df["date"].dt.year.nunique()
    regional = (df.groupby(["district", "state"])["flood"].sum() / years).sort_values(ascending=True)
    summary["flood_days_per_year_by_district"] = {f"{d} ({s})": round(v, 1) for (d, s), v in regional.items()}
    fig, ax = plt.subplots(figsize=(8, 10))
    labels = [f"{d} ({s})" for d, s in regional.index]
    ax.barh(labels, regional.values, color=matplotlib.colormaps["Reds"](0.3 + 0.7 * regional.values / regional.values.max()))
    ax.set(title="Average flood days per year by district", xlabel="Flood days / year")
    _save(fig, figures_dir / "eda_06_regional_flood_frequency.png")

    # 7. temporal patterns
    monthly = df.groupby(df["date"].dt.month).agg(flood_rate=("flood", "mean"), rain=("rainfall_mm", "mean"))
    yearly = df.groupby(df["date"].dt.year)["flood"].sum()
    summary["monthly_flood_rate"] = monthly["flood_rate"].round(4).to_dict()
    summary["yearly_flood_days"] = yearly.to_dict()
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(14, 4.5))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax1.bar(months, monthly["rain"], color="#9ecae9", label="Mean daily rainfall (mm)")
    ax1.set_ylabel("Mean daily rainfall (mm)")
    ax2 = ax1.twinx()
    ax2.plot(months, monthly["flood_rate"] * 100, color="#e45756", marker="o", label="Flood rate (%)")
    ax2.set_ylabel("Flood days (% of region-days)")
    ax2.grid(False)
    ax1.set_title("Seasonality: rainfall and flood occurrence by month")
    ax3.plot(yearly.index, yearly.values, marker="o", color="#e45756")
    ax3.set(title="Total flood days per year (all districts)", xlabel="Year", ylabel="Flood days")
    _save(fig, figures_dir / "eda_07_temporal_patterns.png")

    save_json(summary, reports_dir / "eda_summary.json")
    log.info("EDA complete: 7 figures -> %s", figures_dir)
    return summary
