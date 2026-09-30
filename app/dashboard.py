"""Flood-risk visualisation & early-warning dashboard (report section 18).

Run from the project root, after the pipeline has been executed once:

    streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flood_risk.config import load_config  # noqa: E402
from flood_risk.features.engineering import describe  # noqa: E402
from flood_risk.models.predictor import FloodRiskPredictor  # noqa: E402
from flood_risk.risk import HEADLINES, classify_probability, generate_alerts, level_colors, recommendations_for  # noqa: E402
from flood_risk.utils import load_json  # noqa: E402

st.set_page_config(page_title="Flood Risk Early Warning", page_icon="🌊", layout="wide")

# Streamlit >= 1.50 uses width="stretch"; older releases only understand use_container_width.
_ST_VERSION = tuple(int(part) for part in st.__version__.split(".")[:2])
FULL_WIDTH = {"width": "stretch"} if _ST_VERSION >= (1, 50) else {"use_container_width": True}

CFG = load_config()
P = CFG["paths"]
REQUIRED = [P["models_dir"] / "best_model.joblib", P["processed_dir"] / "features.parquet",
            P["reports_dir"] / "test_predictions.parquet", P["reports_dir"] / "metrics.json"]
DISCLAIMER = ("Decision-support tool only. Always follow official warnings from IMD, CWC, NDMA and "
              "your State Disaster Management Authority.")


# ----------------------------------------------------------------------------- data loading
@st.cache_resource(show_spinner="Loading model ...")
def load_predictor() -> FloodRiskPredictor:
    return FloodRiskPredictor.load(P["models_dir"] / "best_model.joblib")


@st.cache_data(show_spinner="Loading data ...")
def load_data():
    feats = pd.read_parquet(P["processed_dir"] / "features.parquet")
    feats = feats[feats["date"] > CFG["split"]["valid_end"]].reset_index(drop=True)
    preds = pd.read_parquet(P["reports_dir"] / "test_predictions.parquet")
    metrics = load_json(P["reports_dir"] / "metrics.json")
    return feats, preds, metrics


if not all(p.exists() for p in REQUIRED):
    st.error("Model artefacts not found. Run the pipeline first:\n\n`python scripts/run_pipeline.py`")
    st.stop()

predictor = load_predictor()
features_df, test_preds, metrics = load_data()
COLORS = level_colors(predictor.levels)
LEVEL_NAMES = [lv.name for lv in predictor.levels]
regions = features_df[["region_id", "district", "state"]].drop_duplicates().sort_values(["state", "district"])
region_label = {r.region_id: f"{r.district} ({r.state})" for r in regions.itertuples()}


@st.cache_data
def explain_rows(rows: pd.DataFrame, top_n: int) -> list[pd.DataFrame]:
    return predictor.explain(rows, top_n=top_n)


def risk_badge(level: str) -> str:
    return (f"<span style='background:{COLORS[level]};color:white;padding:3px 10px;border-radius:12px;"
            f"font-weight:600'>{level}</span>")


def gauge(prob: float, title: str) -> go.Figure:
    steps = [{"range": [lv.min * 100, lv.max * 100], "color": lv.color} for lv in predictor.levels]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number={"suffix": "%", "valueformat": ".1f"},
        title={"text": title},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#263238", "thickness": 0.25}, "steps": steps},
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def contribution_chart(table: pd.DataFrame) -> go.Figure:
    t = table.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=t["contribution"], y=t["factor"], orientation="h",
        marker_color=np.where(t["contribution"] > 0, "#e45756", "#4c78a8"),
        customdata=np.stack([t["value"], t["typical_value"]], axis=1),
        hovertemplate="%{y}<br>current %{customdata[0]:,.2f} | typical %{customdata[1]:,.2f}"
                      "<br>impact %{x:+.3f}<extra></extra>",
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                      title="Contributing factors (red = raises risk, blue = lowers risk)",
                      xaxis_title="Impact on model flood score")
    return fig


# ----------------------------------------------------------------------------- sidebar
dates = sorted(features_df["date"].dt.date.unique())
busiest = (test_preds[test_preds["risk_level"].isin(["High", "Severe"])]
           .groupby(test_preds["date"].dt.date).size())
default_date = busiest.idxmax() if len(busiest) else dates[-1]

with st.sidebar:
    st.title("🌊 Flood Early Warning")
    obs_date = st.date_input("Observation date", value=default_date, min_value=dates[0], max_value=dates[-1],
                             help="Conditions observed on this date are used to forecast flood risk "
                                  f"{predictor.horizon} day(s) ahead. Only the out-of-sample test period is shown.")
    st.caption(f"Forecast for: **{pd.Timestamp(obs_date) + pd.Timedelta(days=predictor.horizon):%d %b %Y}**")
    map_mode = st.radio("Map style", ["Street map (online)", "Outline map (offline)"], index=0)
    st.divider()
    t = metrics["test"][metrics["best_model"]]
    st.markdown(f"**Model:** {metrics['best_model_name']}  \n"
                f"**Test ROC-AUC:** {t['roc_auc']:.3f}  \n**Test PR-AUC:** {t['pr_auc']:.3f}  \n"
                f"**Recall @ High+:** {t['recall']:.1%}  \n**Precision @ High+:** {t['precision']:.1%}")
    st.divider()
    st.caption(DISCLAIMER)

day = features_df[features_df["date"].dt.date == obs_date].reset_index(drop=True)
if day.empty:
    st.warning("No data for the selected date.")
    st.stop()
current = predictor.predict(day)
current["actual_flood"] = day["flood_next"].to_numpy()
current["label"] = current["region_id"].map(region_label)

# ----------------------------------------------------------------------------- header
st.title("AI-Based Flood Risk Prediction & Early Warning System")
st.caption(f"Region-wise flood risk for **{current['forecast_date'].iloc[0]:%A, %d %B %Y}** · "
           f"{len(current)} districts · model: {predictor.model_name}")
cols = st.columns(5)
for col, level in zip(cols, LEVEL_NAMES):
    n = int((current["risk_level"] == level).sum())
    col.markdown(f"<div style='border-left:6px solid {COLORS[level]};padding:6px 12px;background:rgba(0,0,0,0.03);"
                 f"border-radius:4px'><div style='font-size:0.85rem;opacity:0.7'>{level} risk</div>"
                 f"<div style='font-size:1.8rem;font-weight:700'>{n}</div></div>", unsafe_allow_html=True)
alerts = generate_alerts(current, predictor.levels, CFG["early_warning"]["alert_min_level"])
cols[4].markdown(f"<div style='border-left:6px solid #263238;padding:6px 12px;background:rgba(0,0,0,0.03);"
                 f"border-radius:4px'><div style='font-size:0.85rem;opacity:0.7'>Active alerts</div>"
                 f"<div style='font-size:1.8rem;font-weight:700'>{len(alerts)}</div></div>", unsafe_allow_html=True)
st.write("")

tab_map, tab_region, tab_sim, tab_perf, tab_eda, tab_about = st.tabs(
    ["🗺️ Risk map & alerts", "📍 Region detail", "🧪 What-if simulator", "📊 Model performance", "🔎 EDA", "ℹ️ About"])

# ----------------------------------------------------------------------------- tab: map & alerts
with tab_map:
    left, right = st.columns([3, 2])
    with left:
        plot_df = current.assign(probability_pct=(current["flood_probability"] * 100).round(1),
                                 size=5 + 25 * current["flood_probability"])
        common = dict(color="risk_level", size="size", hover_name="label", size_max=22,
                      color_discrete_map=COLORS, category_orders={"risk_level": LEVEL_NAMES},
                      hover_data={"probability_pct": True, "risk_level": True, "size": False,
                                  "latitude": False, "longitude": False})
        if map_mode.startswith("Street"):
            fig = px.scatter_map(plot_df, lat="latitude", lon="longitude", zoom=3.6,
                                 center={"lat": 22.5, "lon": 82.5}, map_style="carto-positron", **common)
        else:
            fig = px.scatter_geo(plot_df, lat="latitude", lon="longitude", **common)
            fig.update_geos(fitbounds="locations", showcountries=True, showland=True, landcolor="#f3f3f3",
                            showocean=True, oceancolor="#dbe9f6", resolution=50)
        fig.update_layout(height=560, margin=dict(l=0, r=0, t=0, b=0), legend_title_text="Risk level")
        st.plotly_chart(fig, **FULL_WIDTH)
    with right:
        st.subheader("Early-warning alerts")
        if alerts.empty:
            st.success("No district is at High or Severe risk for this date. Continue routine monitoring.")
        else:
            idx = [current.index[current["region_id"] == rid][0] for rid in alerts["region_id"]]
            tables = explain_rows(day.loc[idx], CFG["early_warning"]["top_factors"])
            for (_, a), table in zip(alerts.iterrows(), tables):
                with st.expander(f"{a['risk_level']} · {a['district']}, {a['state']} — {a['flood_probability']:.0%}",
                                 expanded=a["risk_level"] == "Severe"):
                    st.markdown(f"{risk_badge(a['risk_level'])} &nbsp; **{HEADLINES[a['risk_level']]}**",
                                unsafe_allow_html=True)
                    drivers = FloodRiskPredictor.factor_strings(table)
                    if drivers:
                        st.markdown("**Main drivers**\n" + "\n".join(f"- {d}" for d in drivers))
                    st.markdown("**Recommended actions**\n" + "\n".join(f"- {r}" for r in recommendations_for(a["risk_level"])))

    st.subheader("All districts")
    table = current[["label", "flood_probability", "risk_level", "actual_flood"]].rename(columns={
        "label": "District", "flood_probability": "Flood probability", "risk_level": "Risk level",
        "actual_flood": "Flood occurred (ground truth)"})
    table["Flood occurred (ground truth)"] = table["Flood occurred (ground truth)"].map({1: "Yes", 0: "No"})
    st.dataframe(table.sort_values("Flood probability", ascending=False), hide_index=True, **FULL_WIDTH,
                 column_config={"Flood probability": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1)})

# ----------------------------------------------------------------------------- tab: region detail
with tab_region:
    ranked = current.sort_values("flood_probability", ascending=False)["region_id"].tolist()
    rid = st.selectbox("District", ranked, format_func=region_label.get, key="region_detail")
    row = current[current["region_id"] == rid].iloc[0]
    c1, c2 = st.columns([1, 2])
    with c1:
        st.plotly_chart(gauge(row["flood_probability"], "Flood probability"), **FULL_WIDTH)
        st.markdown(f"{risk_badge(row['risk_level'])} &nbsp; {HEADLINES[row['risk_level']]}", unsafe_allow_html=True)
        st.markdown("**Recommended actions**\n" + "\n".join(f"- {r}" for r in row["recommendations"]))
    with c2:
        table = explain_rows(day[day["region_id"] == rid], 8)[0]
        st.plotly_chart(contribution_chart(table), **FULL_WIDTH)

    window = st.slider("Trend window (days either side of the selected date)", 15, 180, 60, step=15)
    hist = test_preds[(test_preds["region_id"] == rid)
                      & (test_preds["date"].between(pd.Timestamp(obs_date) - pd.Timedelta(days=window),
                                                    pd.Timestamp(obs_date) + pd.Timedelta(days=window)))]
    fig = go.Figure()
    for lv in predictor.levels:
        fig.add_hrect(y0=lv.min * 100, y1=lv.max * 100, fillcolor=lv.color, opacity=0.08, line_width=0)
    fig.add_trace(go.Bar(x=hist["forecast_date"], y=hist["rainfall_mm"], name="Rainfall (mm)", yaxis="y2",
                         marker_color="#9ecae9", opacity=0.7))
    fig.add_trace(go.Scatter(x=hist["forecast_date"], y=hist["flood_probability"] * 100, name="Flood probability (%)",
                             line=dict(color="#263238", width=2)))
    floods = hist[hist["actual_flood"] == 1]
    fig.add_trace(go.Scatter(x=floods["forecast_date"], y=np.full(len(floods), 102), mode="markers",
                             name="Actual flood day", marker=dict(symbol="triangle-down", size=9, color="#c62828")))
    fig.add_vline(x=pd.Timestamp(row["forecast_date"]), line_dash="dash", line_color="grey")
    fig.update_layout(height=380, title="Risk trend", margin=dict(l=10, r=10, t=40, b=10),
                      yaxis=dict(title="Flood probability (%)", range=[0, 105]),
                      yaxis2=dict(title="Rainfall (mm)", overlaying="y", side="right", showgrid=False),
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, **FULL_WIDTH)

# ----------------------------------------------------------------------------- tab: what-if
with tab_sim:
    st.markdown("Adjust the observed conditions for a district and see how the predicted flood risk responds. "
                "The other inputs stay at the values observed on the selected date.")
    rid_s = st.selectbox("District", ranked, format_func=region_label.get, key="region_sim")
    base = day[day["region_id"] == rid_s].iloc[[0]].copy()
    sliders = {  # feature: (min, max, step)
        "rainfall_mm": (0.0, 500.0, 5.0), "rain_3d": (0.0, 800.0, 10.0), "rain_7d": (0.0, 1200.0, 10.0),
        "rain_14d": (0.0, 1600.0, 20.0), "rain_30d": (0.0, 2500.0, 25.0),
        "river_above_danger_m": (-12.0, 8.0, 0.25), "river_level_change_1d": (-3.0, 3.0, 0.1),
        "river_discharge_cumecs": (0.0, 30000.0, 100.0), "soil_moisture": (0.03, 0.50, 0.01),
        "humidity_pct": (10.0, 100.0, 1.0),
    }
    active = [f for f in sliders if f in predictor.features]
    scenario = base.copy()
    cols = st.columns(2)
    for i, feat in enumerate(active):
        lo, hi, step = sliders[feat]
        val = float(np.clip(base[feat].iloc[0], lo, hi))
        scenario[feat] = cols[i % 2].slider(describe(feat), lo, hi, val, step=step, key=f"sim_{rid_s}_{feat}")
    # keep the rainfall windows physically consistent (a longer window can't hold less rain)
    rain_chain = [f for f in ["rainfall_mm", "rain_3d", "rain_7d", "rain_14d", "rain_30d"] if f in scenario]
    for shorter, longer in zip(rain_chain, rain_chain[1:]):
        scenario[longer] = np.maximum(scenario[longer], scenario[shorter])
    if "rain_max_3d" in scenario and "rainfall_mm" in scenario:
        scenario["rain_max_3d"] = np.maximum(scenario["rain_max_3d"], scenario["rainfall_mm"])

    p_base = predictor.predict_proba(base)[0]
    p_new = predictor.predict_proba(scenario)[0]
    level_new = classify_probability([p_new], predictor.levels)[0]
    g1, g2 = st.columns(2)
    g1.plotly_chart(gauge(p_base, "Observed conditions"), **FULL_WIDTH)
    g2.plotly_chart(gauge(p_new, "Scenario"), **FULL_WIDTH)
    st.markdown(f"Scenario risk level: {risk_badge(level_new)} &nbsp; {HEADLINES[level_new]}", unsafe_allow_html=True)
    st.markdown("**Recommended actions**\n" + "\n".join(f"- {r}" for r in recommendations_for(level_new)))

# ----------------------------------------------------------------------------- tab: performance
with tab_perf:
    comp = pd.read_csv(P["reports_dir"] / "model_comparison.csv")
    st.subheader("Model comparison (test period, never seen during training)")
    show = comp[["model_name", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier", "valid_pr_auc"]]
    st.dataframe(show.rename(columns={"model_name": "Model", "accuracy": "Accuracy", "precision": "Precision",
                                      "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC",
                                      "brier": "Brier", "valid_pr_auc": "Validation PR-AUC"}).round(4),
                 hide_index=True, **FULL_WIDTH)
    st.caption(f"Best model selected on validation {metrics['selection_metric'].upper()}: **{metrics['best_model_name']}**. "
               f"Decision threshold for Precision/Recall/F1: probability ≥ {metrics['decision_threshold']} (High or Severe).")

    st.subheader("Does the risk framework work?")
    rt = pd.read_csv(P["reports_dir"] / "risk_level_validation.csv")
    st.dataframe(rt.rename(columns={"risk_level": "Risk level", "region_days": "Region-days",
                                    "actual_floods": "Actual floods",
                                    "mean_predicted_probability": "Mean predicted probability",
                                    "observed_flood_rate": "Observed flood rate",
                                    "share_of_all_floods": "Share of all floods"}).round(3),
                 hide_index=True, **FULL_WIDTH)
    figs = sorted((P["figures_dir"]).glob("eval_*.png")) + sorted((P["figures_dir"]).glob("features_*.png"))
    for f1, f2 in zip(figs[::2], figs[1::2] + [None]):
        c1, c2 = st.columns(2)
        c1.image(str(f1), **FULL_WIDTH)
        if f2:
            c2.image(str(f2), **FULL_WIDTH)

# ----------------------------------------------------------------------------- tab: EDA
with tab_eda:
    captions = {
        "eda_01": "Floods are rare: strong class imbalance, handled with class weights and calibration.",
        "eda_02": "Missing values per sensor before preprocessing.",
        "eda_03": "Distributions of weather and environmental variables.",
        "eda_04": "Flood days have much higher river levels, soil moisture and accumulated rainfall.",
        "eda_05": "Correlation between variables and with flood occurrence.",
        "eda_06": "Flood-prone districts: Assam and north Bihar top the list.",
        "eda_07": "Floods follow the monsoon (Jun–Sep); year-to-year variability is large.",
    }
    for f in sorted(P["figures_dir"].glob("eda_*.png")):
        st.image(str(f), caption=captions.get(f.stem[:6], f.stem), **FULL_WIDTH)

# ----------------------------------------------------------------------------- tab: about
with tab_about:
    st.markdown(f"""
### About this system
Phase-1 project for **Machine Learning – II (BAI702)**, CMR Institute of Technology, Bengaluru.

**Pipeline:** Data collection → integration → preprocessing → EDA → feature engineering → feature selection →
model training (Logistic Regression, Random Forest, XGBoost) → probability prediction → risk classification →
evaluation → visualisation & early warning.

**Forecast:** conditions observed on day *t* → probability of flooding on day *t + {predictor.horizon}*.

**Risk levels:** {" · ".join(f"{lv.name} {lv.min:.0%}–{lv.max:.0%}" for lv in predictor.levels)}

**Model inputs ({len(predictor.features)}):** {", ".join(describe(f) for f in predictor.features)}

**Data:** by default, realistic *synthetic* IMD / CWC / NASA / ERA5 / SRTM-style records for 34 Indian districts
(2010–2023). Replace the CSVs in `data/raw` with real data (see `docs/data_schema.md`) and re-run the pipeline.

> ⚠️ {DISCLAIMER}
""")
    st.image(str(ROOT / "docs" / "architecture_from_report.png"), caption="System architecture (from the project report)",
             width=600)
