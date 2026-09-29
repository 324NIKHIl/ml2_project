"""Catalogue of the districts covered by the system.

Coordinates, elevation and climate are approximate real-world values for 34 Indian
districts. They range from chronically flood-prone districts (Assam, north Bihar,
coastal Odisha, Kerala backwaters) through urban-flooding hotspots (Mumbai,
Chennai, Hyderabad) and Himalayan flash-flood districts (Chamoli, Kullu) to
low-risk control regions (Jodhpur, Jaipur).

Columns marked *hidden* drive the synthetic simulator but are NOT published in
the geography table, because they would not normally be directly observable.
"""

from __future__ import annotations

import pandas as pd

# Share of annual rainfall falling in each month (Jan..Dec) for each climate regime.
RAIN_PROFILES: dict[str, list[float]] = {
    # South-west monsoon (most of India)
    "sw": [.01, .01, .02, .03, .06, .15, .25, .23, .15, .06, .02, .01],
    # North-east India: strong pre-monsoon + long monsoon
    "ne_india": [.01, .02, .04, .08, .12, .17, .18, .15, .12, .07, .02, .01],
    # Kerala / Western Ghats: early onset and a second NE-monsoon peak
    "kerala": [.01, .01, .02, .04, .08, .22, .20, .14, .09, .11, .06, .02],
    # Tamil Nadu coast: north-east (winter) monsoon
    "ne": [.03, .01, .01, .02, .04, .05, .08, .11, .11, .22, .23, .09],
    # Kashmir valley: western disturbances in winter / spring
    "western": [.09, .11, .14, .12, .08, .05, .08, .08, .04, .04, .04, .06],
    # South interior Karnataka: peak in Sep-Oct
    "south_interior": [.01, .01, .02, .05, .12, .09, .10, .14, .20, .17, .07, .02],
}

_COLUMNS = [
    "region_id", "district", "state", "latitude", "longitude",
    "elevation_m", "slope_deg", "distance_to_river_km", "urban_fraction", "land_use",
    "annual_rain_mm", "rain_profile", "catchment_factor",  # the last three are *hidden*
]

_REGIONS = [
    ("R01", "Dhemaji",         "Assam",          27.48, 94.58,  100,  1.0,  2.0, .03, "wetland",     3200, "ne_india",       1.8),
    ("R02", "Barpeta",         "Assam",          26.32, 91.00,   35,  0.5,  3.0, .04, "wetland",     2400, "ne_india",       1.7),
    ("R03", "Lakhimpur",       "Assam",          27.23, 94.10,  100,  1.2,  3.0, .04, "agriculture", 3000, "ne_india",       1.6),
    ("R04", "Dibrugarh",       "Assam",          27.47, 94.91,  108,  1.0,  2.5, .10, "agriculture", 2800, "ne_india",       1.6),
    ("R05", "Darbhanga",       "Bihar",          26.15, 85.90,   50,  0.3,  4.0, .07, "agriculture", 1250, "sw",             1.9),
    ("R06", "Madhubani",       "Bihar",          26.35, 86.07,   55,  0.3,  5.0, .05, "agriculture", 1270, "sw",             1.8),
    ("R07", "Sitamarhi",       "Bihar",          26.60, 85.48,   60,  0.4,  3.0, .05, "agriculture", 1300, "sw",             1.9),
    ("R08", "Purnia",          "Bihar",          25.78, 87.47,   36,  0.3,  6.0, .08, "agriculture", 1500, "sw",             1.7),
    ("R09", "Supaul",          "Bihar",          26.12, 86.60,   55,  0.3,  2.0, .04, "wetland",     1400, "sw",             2.0),
    ("R10", "Gorakhpur",       "Uttar Pradesh",  26.76, 83.37,   84,  0.4,  5.0, .15, "agriculture", 1200, "sw",             1.5),
    ("R11", "Bahraich",        "Uttar Pradesh",  27.57, 81.60,  125,  0.6,  6.0, .06, "agriculture", 1100, "sw",             1.5),
    ("R12", "Malda",           "West Bengal",    25.01, 88.14,   30,  0.4,  5.0, .10, "agriculture", 1450, "sw",             1.4),
    ("R13", "Murshidabad",     "West Bengal",    24.18, 88.27,   20,  0.3,  4.0, .09, "agriculture", 1400, "sw",             1.4),
    ("R14", "Kendrapara",      "Odisha",         20.50, 86.42,   10,  0.2,  5.0, .05, "wetland",     1600, "sw",             1.4),
    ("R15", "Jagatsinghpur",   "Odisha",         20.26, 86.17,    8,  0.2,  6.0, .06, "agriculture", 1500, "sw",             1.3),
    ("R16", "Puri",            "Odisha",         19.81, 85.83,    5,  0.3, 10.0, .12, "agriculture", 1450, "sw",             1.0),
    ("R17", "Alappuzha",       "Kerala",          9.49, 76.33,    2,  0.1,  3.0, .20, "wetland",     2900, "kerala",         1.3),
    ("R18", "Ernakulam",       "Kerala",          9.98, 76.30,    8,  1.5,  6.0, .45, "urban",       3200, "kerala",         1.1),
    ("R19", "Mumbai Suburban", "Maharashtra",    19.12, 72.85,   14,  2.0, 12.0, .90, "urban",       2400, "sw",             0.6),
    ("R20", "Kolhapur",        "Maharashtra",    16.70, 74.24,  550,  3.0,  4.0, .20, "agriculture", 1000, "sw",             1.8),
    ("R21", "Chennai",         "Tamil Nadu",     13.08, 80.27,    7,  0.5,  8.0, .95, "urban",       1400, "ne",             0.7),
    ("R22", "Surat",           "Gujarat",        21.17, 72.83,   13,  0.5,  3.0, .60, "urban",       1200, "sw",             1.3),
    ("R23", "Krishna",         "Andhra Pradesh", 16.51, 80.64,   25,  0.8,  5.0, .30, "agriculture", 1000, "sw",             1.2),
    ("R24", "Hyderabad",       "Telangana",      17.39, 78.49,  505,  2.0, 12.0, .85, "urban",        800, "sw",             0.5),
    ("R25", "Chamoli",         "Uttarakhand",    30.40, 79.33, 1300, 25.0,  1.5, .02, "forest",      1700, "sw",             1.0),
    ("R26", "Kullu",           "Himachal Pradesh", 31.96, 77.11, 1220, 22.0, 1.0, .05, "forest",     1400, "sw",             1.0),
    ("R27", "Srinagar",        "Jammu & Kashmir", 34.08, 74.80, 1585,  3.0,  2.0, .40, "urban",        700, "western",        1.3),
    ("R28", "New Delhi",       "Delhi",          28.61, 77.21,  216,  0.5,  6.0, .95, "urban",        800, "sw",             1.2),
    ("R29", "Jaipur",          "Rajasthan",      26.91, 75.79,  431,  1.5, 25.0, .60, "urban",        600, "sw",             0.3),
    ("R30", "Jodhpur",         "Rajasthan",      26.24, 73.02,  231,  1.0, 40.0, .30, "arid",         360, "sw",             0.2),
    ("R31", "Bengaluru Urban", "Karnataka",      12.97, 77.59,  920,  1.5, 20.0, .85, "urban",        950, "south_interior", 0.3),
    ("R32", "Kodagu",          "Karnataka",      12.42, 75.74, 1100, 15.0,  3.0, .03, "forest",      2700, "kerala",         1.3),
    ("R33", "Bhopal",          "Madhya Pradesh", 23.26, 77.41,  527,  1.5, 15.0, .55, "urban",       1150, "sw",             0.5),
    ("R34", "Ludhiana",        "Punjab",         30.90, 75.85,  244,  0.5,  5.0, .50, "agriculture",  700, "sw",             1.2),
]

LAND_USE_CATEGORIES = ["agriculture", "arid", "forest", "urban", "wetland"]
HIDDEN_COLUMNS = ["annual_rain_mm", "rain_profile", "catchment_factor"]


def get_regions() -> pd.DataFrame:
    """Full region catalogue including simulator-only (hidden) parameters."""
    return pd.DataFrame(_REGIONS, columns=_COLUMNS)
