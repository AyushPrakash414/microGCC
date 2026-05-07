"""
Model Selector.

Orchestrates training, evaluation, and automatic selection of the
best forecasting model per state based on the primary metric (RMSE).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.core.config import get_settings, get_yaml_config
from app.core.constants import MODEL_FILE_EXTENSIONS, MODEL_NAMES, TARGET
from app.core.logger import get_logger
from app.features.feature_engineering import engineer_features
from app.models.arima_model import ARIMAForecaster
from app.models.base_model import BaseForecaster
from app.models.lstm_model import LSTMForecaster
from app.models.prophet_model import ProphetForecaster
from app.models.xgboost_model import XGBoostForecaster
from app.utils.helpers import (
    ensure_dir,
    get_next_version,
    save_json,
    timer,
    timestamp_now,
    append_metrics_csv,
)
from app.utils.metrics import compute_all_metrics

logger = get_logger(__name__)
settings = get_settings()
cfg = get_yaml_config()


def _create_model(name: str) -> BaseForecaster:
    """Factory: instantiate a model by name."""
    mapping = {
        "arima": ARIMAForecaster,
        "prophet": ProphetForecaster,
        "xgboost": XGBoostForecaster,
        "lstm": LSTMForecaster,
    }
    cls = mapping.get(name)
    if cls is None:
        raise ValueError(f"Unknown model: {name}")
    return cls()


def chronological_split(
    df: pd.DataFrame, val_weeks: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train/validation chronologically (no data leakage)."""
    split_idx = len(df) - val_weeks
    return df.iloc[:split_idx], df.iloc[split_idx:]


def train_and_evaluate_state(
    state: str,
    state_data: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Train all models for a single state, evaluate, and select the best.

    Returns a result dict containing per-model metrics and best model info.
    """
    logger.info("=" * 60)
    logger.info("Processing state: %s (%d weekly records)", state, len(state_data))
    logger.info("=" * 60)

    # Feature engineering for ML models
    featured_data = engineer_features(state_data.copy())

    val_weeks = cfg.validation_weeks
    train_df, val_df = chronological_split(featured_data, val_weeks)

    # Also keep raw series split for ARIMA / Prophet (no engineered features)
    raw_train, raw_val = chronological_split(state_data, val_weeks)

    results: Dict[str, Dict[str, Any]] = {}
    version = get_next_version(settings.saved_models_dir / state)
    model_dir = settings.saved_models_dir / state / version
    ensure_dir(model_dir)

    for model_name in MODEL_NAMES:
        logger.info("--- Training %s for %s ---", model_name, state)
        model = _create_model(model_name)

        try:
            with timer(f"{model_name} training") as t:
                if model_name in ("arima", "prophet"):
                    model.train(raw_train, raw_val)
                else:
                    model.train(train_df, val_df)

            # Evaluate on validation set
            if model_name in ("arima", "prophet"):
                preds_df = model.predict(steps=len(raw_val))
                y_true = raw_val[TARGET].values
            else:
                preds_df = model.predict(steps=len(val_df))
                y_true = val_df[TARGET].values

            y_pred = preds_df["predicted_sales"].values[: len(y_true)]
            metrics = compute_all_metrics(y_true, y_pred)

            # Save model
            ext = MODEL_FILE_EXTENSIONS[model_name]
            model_path = model_dir / f"{model_name}{ext}"
            model.save_model(model_path)

            results[model_name] = {
                "metrics": metrics,
                "training_time_seconds": t["elapsed"],
                "model_path": str(model_path),
            }

            logger.info(
                "%s | %s -> RMSE=%.2f  MAE=%.2f  MAPE=%.2f%%  (%.1fs)",
                state, model_name,
                metrics["rmse"], metrics["mae"], metrics["mape"],
                t["elapsed"],
            )

        except Exception as e:
            logger.error("Failed to train %s for %s: %s", model_name, state, e)
            results[model_name] = {
                "metrics": {"rmse": float("inf"), "mae": float("inf"), "mape": float("inf")},
                "training_time_seconds": 0.0,
                "error": str(e),
            }

    # Select best model
    best_model = min(
        results,
        key=lambda m: results[m]["metrics"]["rmse"],
    )

    # Build metadata
    metadata = {
        "state": state,
        "best_model": best_model,
        "version": version,
        "timestamp": timestamp_now(),
        "models": {
            name: {
                "rmse": r["metrics"]["rmse"],
                "mae": r["metrics"]["mae"],
                "mape": r["metrics"]["mape"],
                "training_time_seconds": r.get("training_time_seconds", 0),
                "model_path": r.get("model_path", ""),
            }
            for name, r in results.items()
        },
        "features_used": list(train_df.columns),
    }

    # Save metadata
    meta_dir = settings.metadata_dir / state
    ensure_dir(meta_dir)
    save_json(metadata, meta_dir / f"{version}_metadata.json")

    # Also save a "latest" pointer
    save_json(
        {"best_model": best_model, "version": version, "rmse": results[best_model]["metrics"]["rmse"]},
        settings.metadata_dir / state / "latest.json",
    )

    # Append to centralized metrics CSV
    csv_records = []
    for name, r in results.items():
        csv_records.append({
            "state": state,
            "model": name,
            "rmse": r["metrics"]["rmse"],
            "mae": r["metrics"]["mae"],
            "mape": r["metrics"]["mape"],
            "training_time_seconds": r.get("training_time_seconds", 0),
            "version": version,
            "timestamp": timestamp_now(),
        })
    append_metrics_csv(settings.metrics_dir / "all_metrics.csv", csv_records)

    logger.info("[OK] Best model for %s: %s (RMSE=%.2f)", state, best_model, results[best_model]["metrics"]["rmse"])
    return metadata
