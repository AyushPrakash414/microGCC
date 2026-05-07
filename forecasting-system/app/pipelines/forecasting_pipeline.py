"""
Forecasting Pipeline.

Loads the best trained model for a given state and produces
an 8-week (configurable) sales forecast with confidence intervals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from app.core.config import get_settings, get_yaml_config
from app.core.constants import MODEL_FILE_EXTENSIONS, TARGET
from app.core.logger import get_logger
from app.models.arima_model import ARIMAForecaster
from app.models.lstm_model import LSTMForecaster
from app.models.prophet_model import ProphetForecaster
from app.models.xgboost_model import XGBoostForecaster
from app.models.base_model import BaseForecaster
from app.utils.helpers import load_json

logger = get_logger(__name__)
settings = get_settings()
cfg = get_yaml_config()


_MODEL_CLASSES = {
    "arima": ARIMAForecaster,
    "prophet": ProphetForecaster,
    "xgboost": XGBoostForecaster,
    "lstm": LSTMForecaster,
}


def _load_best_model(state: str) -> tuple[BaseForecaster, str, str]:
    """
    Load the best model for a state.

    Returns:
        ``(model_instance, model_name, version)``
    """
    latest_path = settings.metadata_dir / state / "latest.json"
    if not latest_path.exists():
        raise FileNotFoundError(
            f"No trained model found for state '{state}'. "
            "Please run POST /train first."
        )

    latest = load_json(latest_path)
    best_model_name: str = latest["best_model"]
    version: str = latest["version"]

    ext = MODEL_FILE_EXTENSIONS[best_model_name]
    model_path = settings.saved_models_dir / state / version / f"{best_model_name}{ext}"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_cls = _MODEL_CLASSES[best_model_name]
    model = model_cls()
    model.load_model(model_path)

    logger.info(
        "Loaded best model for %s: %s (%s) from %s",
        state, best_model_name, version, model_path,
    )
    return model, best_model_name, version


def run_forecasting_pipeline(state: str) -> Dict[str, Any]:
    """
    Produce a forecast for *state*.

    Returns:
        A dict matching ``ForecastResponse`` schema.
    """
    model, model_name, version = _load_best_model(state)
    horizon = cfg.horizon

    logger.info("Forecasting %d weeks for %s using %s", horizon, state, model_name)

    forecast_df = model.predict(steps=horizon)

    forecast_list = []
    for _, row in forecast_df.iterrows():
        forecast_list.append({
            "date": row["date"],
            "predicted_sales": round(float(row["predicted_sales"]), 2),
            "lower_bound": round(float(row.get("lower_bound", 0)), 2),
            "upper_bound": round(float(row.get("upper_bound", 0)), 2),
        })

    return {
        "state": state,
        "best_model": model_name,
        "forecast_horizon_weeks": horizon,
        "forecast": forecast_list,
    }
