# Data schema: using real data

The pipeline reads six CSV files from `data/raw/`. By default they are **simulated**
(`data.source: synthetic` in `config/config.yaml`). To run the system on real data,
put your own files in `data/raw/` with the names and columns below, set
`data.source: local`, and re-run `python scripts/run_pipeline.py`.

Extra columns are ignored. Dates use the `YYYY-MM-DD` format. `region_id` is the join key
across all files, so use the same id everywhere (for example a district code).

| File | Real-world source | Required columns |
|---|---|---|
| `rainfall_imd.csv` | IMD gridded rainfall, data.gov.in (OGD), ERA5 `tp` | `region_id, date, rainfall_mm` |
| `river_cwc.csv` | CWC flood-forecasting stations / India-WRIS | `region_id, date, river_level_m, river_discharge_cumecs` |
| `soil_moisture_nasa.csv` | NASA SMAP / ERA5-Land `swvl1` (volumetric, 0–0.6) | `region_id, date, soil_moisture` |
| `weather_era5.csv` | Copernicus ERA5 (`t2m` in °C, relative humidity) | `region_id, date, temperature_c, humidity_pct` |
| `geography_srtm.csv` | USGS EarthExplorer SRTM DEM, land-cover maps, CWC danger levels | `region_id, district, state, latitude, longitude, elevation_m, slope_deg, distance_to_river_km, urban_fraction, land_use, danger_level_m` |
| `flood_events.csv` | Govt. flood reports, NDMA/SDMA, Dartmouth Flood Observatory, Kaggle | `event_id, region_id, start_date, end_date, severity` |

Notes:

* `land_use` must be one of `agriculture, arid, forest, urban, wetland` (see
  `flood_risk/data/regions.py`). Add a category there if you need another one.
* `danger_level_m` is the CWC danger level of the river gauge serving the region, in the
  same datum as `river_level_m`.
* Sentinel values such as `-999`, gaps and duplicate rows are fine. The preprocessing stage
  cleans them.
* Keep `split.train_end` / `split.valid_end` in the config inside your data's date range.

## Synthetic data: how it is generated

`flood_risk/data/synthetic.py` simulates 34 real Indian districts from 2010 to 2023 with
simple physical models:

* **Rainfall:** a wet/dry Markov chain with gamma-distributed amounts, a monsoon regime per
  region (south-west, north-east India, Kerala, Tamil Nadu north-east monsoon, Kashmir, south
  interior), year-to-year variability and extreme-rain spells.
* **Soil moisture:** a bucket water balance (infiltration, evapotranspiration, drainage).
* **River:** catchment storage converted to stage and discharge, with danger levels.
* **Floods:** a latent hazard combining riverine, urban (pluvial) and flash-flood mechanisms,
  soil saturation, elevation and distance to the river. Consecutive flood days are stored as
  events.

It then injects realistic data problems: about 2% missing values, `-999` sentinels, gauge
spikes, humidity above 100% and about 0.1% duplicate rows.
