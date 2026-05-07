"""
Data Preprocessing Pipeline.

Handles:
- Loading raw Excel data
- Date conversion & sorting
- Duplicate removal
- Missing-value imputation
- Negative-value clamping
- Weekly resampling
- State-wise dataset generation
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from app.core.config import get_settings, get_yaml_config
from app.core.constants import COL_DATE, COL_STATE, COL_TOTAL, COL_CATEGORY
from app.core.logger import get_logger
from app.utils.validators import validate_raw_dataframe, validate_state_series

logger = get_logger(__name__)
settings = get_settings()
cfg = get_yaml_config()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_raw_data(filepath: Path | None = None) -> pd.DataFrame:
    """Load the Excel dataset and perform initial type conversions."""
    filepath = filepath or settings.data_filepath
    logger.info("Loading raw data from %s", filepath)
    df = pd.read_excel(filepath, engine="openpyxl")
    df[COL_DATE] = pd.to_datetime(df[COL_DATE])
    df = df.sort_values([COL_STATE, COL_DATE]).reset_index(drop=True)
    logger.info("Raw data loaded: %d rows, %d columns", *df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cleaning steps to the raw dataframe.

    Steps:
    1. Validate data quality
    2. Remove duplicates
    3. Clamp negative sales to 0
    4. Forward-fill missing target values
    """
    logger.info("Cleaning data …")
    validate_raw_dataframe(df)

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=[COL_STATE, COL_DATE], keep="last")
    logger.info("Removed %d duplicate rows", before - len(df))

    # Clamp negative sales
    neg_mask = df[COL_TOTAL] < 0
    if neg_mask.any():
        df.loc[neg_mask, COL_TOTAL] = 0
        logger.info("Clamped %d negative sales to 0", neg_mask.sum())

    # Fill missing target values
    df[COL_TOTAL] = df.groupby(COL_STATE)[COL_TOTAL].transform(
        lambda s: s.ffill().bfill().fillna(0)
    )

    return df


def resample_weekly(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Resample data to weekly frequency per state.

    Returns:
        Mapping of *state → weekly DataFrame* (DatetimeIndex).
    """
    logger.info("Resampling to weekly frequency …")
    state_datasets: Dict[str, pd.DataFrame] = {}
    min_weeks = cfg.min_data_weeks

    for state, group in df.groupby(COL_STATE):
        ts = (
            group.set_index(COL_DATE)[[COL_TOTAL]]
            .resample(cfg.frequency)
            .sum()
        )

        # Fill any gaps introduced by resampling
        ts[COL_TOTAL] = ts[COL_TOTAL].ffill().fillna(0)

        valid, msg = validate_state_series(str(state), ts, min_weeks)
        if not valid:
            logger.warning("Skipping state: %s — %s", state, msg)
            continue

        state_datasets[str(state)] = ts

    logger.info(
        "Weekly resampling complete: %d states retained", len(state_datasets)
    )
    return state_datasets


def run_preprocessing_pipeline(
    filepath: Path | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Execute the full preprocessing pipeline end-to-end.

    Returns:
        Dict mapping state name to its cleaned, weekly-resampled DataFrame.
    """
    raw = load_raw_data(filepath)
    cleaned = clean_data(raw)
    state_data = resample_weekly(cleaned)
    return state_data


def get_available_states(filepath: Path | None = None) -> List[str]:
    """Return sorted list of states present in the dataset."""
    filepath = filepath or settings.data_filepath
    df = pd.read_excel(filepath, engine="openpyxl")
    return sorted(df[COL_STATE].unique().tolist())
