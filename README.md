# AI-Based Flood Risk Prediction and Early Warning System

**Machine Learning – II (BAI702)** · Project-based assignment
Dept. of AI & ML, CMR Institute of Technology, Bengaluru
Nidhi Mishra (1CR23AI073) · Nikhil Chaudhary (1CR23AI075)

The system looks at the weather, river, soil and geographic conditions observed in a district
today. From these it estimates the **probability of a flood tomorrow**. It then turns that
probability into a **risk level** (Low / Moderate / High / Severe) and issues **early-warning
alerts** with recommended actions and the main contributing factors. The results appear in an
interactive map dashboard.

> ⚠️ This is a decision-support tool. Use it alongside official IMD / CWC / NDMA warnings,
> not in place of them.

---

## 1. Quick start

```bash
cd flood_risk_prediction
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

python scripts/run_pipeline.py    # data -> EDA -> features -> 3 models -> evaluation -> alerts (~1 min)
streamlit run app/dashboard.py    # open http://localhost:8501
```

Tested on Python 3.11 with pandas 2.3 and 3.0, scikit-learn 1.7 and 1.9, and Streamlit 1.46 and 1.64.

Other commands:

```bash
python scripts/predict.py --date 2023-07-13 --alerts-only   # early-warning report for one day
python scripts/predict.py --date 2023-07-13 --region Darbhanga
python scripts/run_pipeline.py --regenerate                 # re-simulate the raw data
python scripts/run_pipeline.py --tune                       # hyper-parameter search (slower)
python -m pytest                                            # 12 unit + end-to-end tests
```

---

## 2. How the code maps to the report

The architecture in report section 15 (`docs/architecture_from_report.png`) is built one
module per stage:

| # | Report stage | Code |
|---|---|---|
| 1 | Data sources: rainfall, river, soil, weather, geography, flood records | `flood_risk/data/synthetic.py`, `flood_risk/data/regions.py` |
| 2 | Data collection & integration, by time and location | `flood_risk/data/ingestion.py`, `flood_risk/data/integration.py` |
| 3 | Preprocessing: missing values, outliers, format | `flood_risk/data/preprocessing.py` |
| 4 | Exploratory data analysis | `flood_risk/eda/analysis.py` |
| 5 | Feature engineering & selection | `flood_risk/features/engineering.py`, `flood_risk/features/selection.py` |
| 6 | Model training: Logistic Regression, Random Forest, XGBoost | `flood_risk/models/registry.py`, `flood_risk/models/train.py` |
| 7 | Flood probability prediction | `flood_risk/models/predictor.py` |
| 8 | Risk classification (report section 16) | `flood_risk/risk/classification.py` |
| 9 | Evaluation (report section 17) | `flood_risk/models/evaluate.py` |
| 10 | Visualization dashboard + early warning & decision support | `app/dashboard.py`, `flood_risk/risk/early_warning.py` |
| – | End-to-end orchestration | `flood_risk/pipeline.py`, `scripts/run_pipeline.py` |

## 3. Project structure

```
flood_risk_prediction/
├── config/config.yaml          # every setting: dates, split, features, model params, risk thresholds
├── flood_risk/                 # the Python package (one sub-package per pipeline stage)
│   ├── data/                   #   sources, integration, preprocessing
│   ├── eda/                    #   exploratory analysis & figures
│   ├── features/               #   feature engineering & selection
│   ├── models/                 #   model zoo, training, evaluation, predictor
│   ├── risk/                   #   risk levels, alerts & recommendations
│   ├── utils/                  #   logging, file I/O
│   ├── config.py
│   └── pipeline.py             #   runs all stages in order
├── scripts/
│   ├── run_pipeline.py         # train everything
│   └── predict.py              # forecast / alerts for a given date (CLI)
├── app/dashboard.py            # Streamlit + Plotly dashboard
├── tests/                      # pytest suite
├── docs/                       # data schema (for real data), architecture figure
├── data/{raw,interim,processed}/   <- generated
├── models/                         <- generated (*.joblib)
└── reports/ (+ figures/)           <- generated metrics, CSVs, plots, alerts
```

---

## 4. Data

The report proposes public sources: IMD, OGD India, CWC, Copernicus ERA5, NASA, USGS and
Kaggle. These need registration, API keys or manual downloads. So that the project runs out of
the box, `data.source: synthetic` **simulates those six source tables** for **34 real Indian
districts, 2010–2023** (~174 k region-days). The simulation uses simple physical models: monsoon
rainfall regimes, a soil-moisture water balance, river storage with CWC-style danger levels,
and riverine, urban and flash-flood mechanisms. It also injects real-world data problems:
missing values, `-999` sentinels, gauge spikes and duplicates.

**To use real data**, drop CSVs with the documented columns into `data/raw/`, set
`data.source: local` and re-run. See [docs/data_schema.md](docs/data_schema.md).

> Because the default data is simulated, the metrics below show that the method works. They
> are not claims about real-world flood-forecast skill.

## 5. Methodology

* **Target:** flood on day *t + 1* (`target.forecast_horizon_days`), predicted from conditions
  known at the end of day *t*. This is a real early warning, not a same-day nowcast.
* **Integration:** the four time-series sources are joined on a full region × day calendar.
  Geography is attached per region, and flood *events* (start–end) are expanded to daily 0/1
  labels.
* **Preprocessing:**
  * Values outside physical ranges become missing.
  * Isolated gauge spikes are removed, with a stage–discharge consistency check so that real
    flood peaks are kept.
  * Gaps are filled by time interpolation, then by the region × month median.
  * Scaling is done inside the model pipeline, fitted on the training split only.
* **Features (32 candidates):**
  * rainfall accumulation over 3/7/14/30 days, maximum and intensity, and wet-day count;
  * river level relative to the danger level, level trend, discharge and its growth;
  * soil moisture and its trend, humidity and temperature;
  * season, including a monsoon flag;
  * elevation, slope, distance to the river, urbanisation and land use;
  * flood days in the past year (from historical flood records).
* **Selection:** mutual information and Random-Forest importance, averaged by rank. A
  correlation filter (|r| > 0.95) removes redundant features, and the top 22 are kept. All of
  this runs on the training data only.
* **Split (chronological):** train 2010–2019 · validation 2020–2021 · test 2022–2023.
* **Models:** Logistic Regression (scaled baseline), Random Forest and XGBoost. Class
  imbalance (about 3.5% flood days) is handled with class weights / `scale_pos_weight`.
  Each model then gets **isotonic calibration** on the validation set, so a "70%" really
  means about 70%.
* **Model selection:** the best validation PR-AUC. PR-AUC is chosen over accuracy because
  floods are rare: a model that always says "no flood" would be 96.5% accurate.
* **Explanations:** occlusion. Each feature is reset to its typical value, and the change in
  the model score is that feature's contribution. This works for any model.

## 6. Results (default run, test period 2022–2023, 24,786 region-days)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.977 | 0.742 | 0.434 | 0.548 | 0.957 | 0.611 | 0.0182 |
| Random Forest | 0.978 | 0.767 | 0.445 | 0.563 | 0.959 | 0.616 | 0.0180 |
| **XGBoost** (selected) | **0.978** | 0.757 | **0.473** | **0.582** | 0.958 | **0.630** | **0.0174** |

Precision, recall and F1 use the threshold *probability ≥ 0.5*, which means risk level High or
Severe.

**Risk-framework validation (XGBoost).** The observed flood rate rises steadily with the risk
level and closely matches the predicted probability:

| Risk level | Region-days | Mean predicted probability | Observed flood rate |
|---|---|---|---|
| Low (0–25%) | 24,010 | 1.4% | 1.4% |
| Moderate (25–50%) | 278 | 36.8% | 32.0% |
| High (50–75%) | 209 | 58.7% | 50.2% |
| Severe (75–100%) | 289 | 96.1% | 94.1% |

**Most important factors** (permutation importance): river level relative to the danger level,
river discharge, soil moisture, and 14- and 30-day cumulative rainfall.

Exact numbers are regenerated in `reports/metrics.json` and `reports/model_comparison.csv`.

## 7. Outputs

| File | Content |
|---|---|
| `reports/figures/eda_*.png` | class balance, missing values, distributions, flood vs non-flood, correlation, regional & temporal patterns |
| `reports/figures/features_01_ranking.png` | feature ranking (MI and RF importance) |
| `reports/figures/eval_*.png` | ROC/PR curves, confusion matrices, calibration, risk-level validation, feature importance |
| `reports/metrics.json`, `model_comparison.csv` | all metrics, validation and test |
| `reports/data_quality_report.json` | duplicates, invalid values, spikes and imputation counts |
| `reports/feature_ranking.csv`, `selected_features.json`, `feature_importance.csv` | feature analysis |
| `reports/test_predictions.parquet` | probability and risk level for every district-day in the test period |
| `reports/alerts_test_period.csv` | every High/Severe alert with its message and recommended actions |
| `models/*.joblib` | the three trained models; `best_model.joblib` is used by the dashboard and CLI |

## 8. Dashboard

`streamlit run app/dashboard.py` has six tabs:

1. **Risk map & alerts:** region-wise map coloured by risk level, KPI tiles, alert cards
   (headline, main drivers, recommended actions) and a table of all districts with ground truth.
2. **Region detail:** probability gauge, contributing-factor chart, recommendations, and a risk
   trend with rainfall and actual flood days.
3. **What-if simulator:** change rainfall, river level, soil moisture and other inputs, and see
   the risk change.
4. **Model performance:** model comparison, risk-framework validation and evaluation plots.
5. **EDA:** all exploratory figures.
6. **About:** method summary and the architecture diagram.

Pick the observation date in the sidebar. The default is the busiest alert day in the test
period. Use "Outline map (offline)" if map tiles cannot load.

## 9. Configuration

Everything can be changed in `config/config.yaml` without touching the code:

* date range and train/validation/test split;
* forecast horizon;
* rainfall windows, number of features and correlation threshold;
* model hyper-parameters, selection metric, decision threshold and calibration;
* **risk thresholds and colours**;
* the minimum alert level.

## 10. Limitations & future scope

These follow report sections 21–22.

**Limitations**
* Accuracy depends on data quality and coverage.
* Floods are rare (class imbalance).
* Events that never occurred in the history cannot be learned.
* District-level aggregation hides local variation.
* The default data is synthetic.

**Future scope**
* Real-time IMD/CWC feeds and IoT river sensors.
* Satellite flood extents (Sentinel-1).
* Sequence models (LSTM/Temporal CNN) for multi-day horizons.
* More regions and other hazards.
