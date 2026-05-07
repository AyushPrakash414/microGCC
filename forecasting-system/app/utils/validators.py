"""
Data Validation Utilities.

Validates raw and processed data before it enters the training pipeline.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from app.core.constants import COL_DATE, COL_STATE, COL_TOTAL
from app.core.logger import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Raised when data validation fails."""


def validate_raw_dataframe(df: pd.DataFrame) -> List[str]:
    """
    Run a battery of checks on the raw dataset.

    Returns:
        A list of warning/info messages.  Raises ``ValidationError``
        for critical failures.
    """
    warnings: List[str] = []

    # --- Required columns ---
    required = {COL_STATE, COL_DATE, COL_TOTAL}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValidationError(f"Missing required columns: {missing_cols}")

    # --- Empty dataframe ---
    if df.empty:
        raise ValidationError("Dataset is empty.")

    # --- Duplicate rows ---
    n_dups = df.duplicated(subset=[COL_STATE, COL_DATE]).sum()
    if n_dups > 0:
        warnings.append(f"Found {n_dups} duplicate (State, Date) rows – will be removed.")

    # --- Negative sales ---
    n_neg = (df[COL_TOTAL] < 0).sum()
    if n_neg > 0:
        warnings.append(f"Found {n_neg} negative sales values – will be set to 0.")

    # --- Missing target values ---
    n_null_target = df[COL_TOTAL].isnull().sum()
    if n_null_target > 0:
        warnings.append(f"Found {n_null_target} missing target values – will be imputed.")

    # --- Invalid dates ---
    try:
        pd.to_datetime(df[COL_DATE])
    except Exception:
        raise ValidationError("Date column contains unparseable values.")

    for w in warnings:
        logger.warning("Validation: %s", w)

    return warnings


def validate_state_series(
    state: str,
    df: pd.DataFrame,
    min_weeks: int,
) -> Tuple[bool, str]:
    """
    Check whether a single state has enough data for reliable modelling.

    Returns:
        ``(is_valid, message)``
    """
    n_weeks = len(df)
    if n_weeks < min_weeks:
        msg = (
            f"State '{state}' has only {n_weeks} weekly records "
            f"(minimum {min_weeks} required) – skipping."
        )
        logger.warning(msg)
        return False, msg
    return True, "OK"


def validate_state_name(state: str, valid_states: List[str]) -> None:
    """Raise ``ValidationError`` if *state* is not in the known list."""
    if state not in valid_states:
        raise ValidationError(
            f"Unknown state '{state}'. Valid states: {valid_states}"
        )
