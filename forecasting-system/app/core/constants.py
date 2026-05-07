"""
Application Constants.

Central location for constant values used across the system.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Column Names
# ---------------------------------------------------------------------------
COL_STATE = "State"
COL_DATE = "Date"
COL_TOTAL = "Total"
COL_CATEGORY = "Category"
TARGET = COL_TOTAL

# Prophet expected column names
PROPHET_DS = "ds"
PROPHET_Y = "y"

# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------
MODEL_NAMES = ["arima", "prophet", "xgboost", "lstm"]

MODEL_FILE_EXTENSIONS = {
    "arima": ".pkl",
    "prophet": ".pkl",
    "xgboost": ".pkl",
    "lstm": ".keras",
}

# ---------------------------------------------------------------------------
# Feature Names
# ---------------------------------------------------------------------------
LAG_FEATURES = ["lag_1", "lag_7", "lag_30"]
ROLLING_FEATURES = [
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_30",
]
DATE_FEATURES = [
    "day_of_week",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "is_weekend",
]
HOLIDAY_FEATURE = "is_holiday"

ALL_ENGINEERED_FEATURES = LAG_FEATURES + ROLLING_FEATURES + DATE_FEATURES + [HOLIDAY_FEATURE]

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
PRIMARY_METRIC = "rmse"

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_TITLE = "Time Series Forecasting System"
API_DESCRIPTION = (
    "Production-grade forecasting backend that trains SARIMAX, Prophet, "
    "XGBoost, and LSTM models per US state and exposes 8-week sales forecasts "
    "through REST APIs."
)
API_VERSION = "1.0.0"
