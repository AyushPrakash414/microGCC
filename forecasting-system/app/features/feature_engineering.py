"""
Feature Engineering Module.

Creates advanced time-series features for ML models:
- Lag features
- Rolling statistics
- Calendar / date features
- US holiday indicator
"""

from __future__ import annotations

from typing import List

import holidays as holidays_lib
import numpy as np
import pandas as pd

from app.core.config import get_yaml_config
from app.core.constants import COL_TOTAL, TARGET
from app.core.logger import get_logger

logger = get_logger(__name__)
cfg = get_yaml_config()


def add_lag_features(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Add lag_1, lag_7, lag_30 features (based on config)."""
    for lag in cfg.lag_periods:
        col = f"lag_{lag}"
        df[col] = df[target].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Add rolling mean and std features."""
    for window in cfg.rolling_windows:
        df[f"rolling_mean_{window}"] = (
            df[target].shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f"rolling_std_{window}"] = (
            df[target].shift(1).rolling(window=window, min_periods=1).std().fillna(0)
        )
    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar features from the DatetimeIndex."""
    idx = df.index

    df["day_of_week"] = idx.dayofweek
    df["week_of_year"] = idx.isocalendar().week.astype(int).values
    df["month"] = idx.month
    df["quarter"] = idx.quarter
    df["year"] = idx.year
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    return df


def add_holiday_feature(df: pd.DataFrame, country: str | None = None) -> pd.DataFrame:
    """Flag rows whose date falls on (or within the week of) a US holiday."""
    country = country or cfg.holiday_country
    years = sorted(df.index.year.unique())
    us_holidays = holidays_lib.country_holidays(country, years=years)

    # For weekly data, mark a week as holiday if the week contains a holiday
    df["is_holiday"] = 0
    for date in df.index:
        week_start = date - pd.Timedelta(days=6)
        week_dates = pd.date_range(week_start, date)
        if any(d in us_holidays for d in week_dates):
            df.at[date, "is_holiday"] = 1

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the complete feature engineering pipeline.

    The input must have a ``DatetimeIndex`` and a ``Total`` column.
    """
    logger.info("Engineering features (%d rows) …", len(df))
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_date_features(df)
    df = add_holiday_feature(df)

    # Drop rows where lags are NaN (burn-in period)
    max_lag = max(cfg.lag_periods)
    df = df.iloc[max_lag:]
    df = df.fillna(0)

    logger.info("Feature engineering complete: %d rows, %d cols", *df.shape)
    return df


def get_feature_columns() -> List[str]:
    """Return the list of engineered feature column names (excludes target)."""
    feature_cols: List[str] = []
    for lag in cfg.lag_periods:
        feature_cols.append(f"lag_{lag}")
    for window in cfg.rolling_windows:
        feature_cols.append(f"rolling_mean_{window}")
        feature_cols.append(f"rolling_std_{window}")
    feature_cols.extend([
        "day_of_week", "week_of_year", "month", "quarter", "year",
        "is_weekend", "is_holiday",
    ])
    return feature_cols
