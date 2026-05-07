"""
XGBoost Forecasting Model.

Uses gradient-boosted trees with engineered features and
**recursive (autoregressive) forecasting** for multi-step prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from app.core.config import get_yaml_config
from app.core.constants import TARGET
from app.core.logger import get_logger
from app.features.feature_engineering import get_feature_columns
from app.models.base_model import BaseForecaster
from app.utils.helpers import ensure_dir, save_artifact, load_artifact

logger = get_logger(__name__)
cfg = get_yaml_config()


class XGBoostForecaster(BaseForecaster):
    """XGBoost-based forecaster with recursive multi-step prediction."""

    name = "xgboost"

    def __init__(self) -> None:
        super().__init__()
        params = cfg.model_params("xgboost")
        self.n_estimators = params.get("n_estimators", 500)
        self.max_depth = params.get("max_depth", 6)
        self.learning_rate = params.get("learning_rate", 0.05)
        self.subsample = params.get("subsample", 0.8)
        self.colsample_bytree = params.get("colsample_bytree", 0.8)
        self.early_stopping_rounds = params.get("early_stopping_rounds", 50)
        self.random_state = params.get("random_state", 42)

        self.feature_cols: List[str] = []
        self._train_tail: pd.DataFrame | None = None

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> None:
        """Fit XGBRegressor on engineered features."""
        self.feature_cols = get_feature_columns()
        X_train = train_data[self.feature_cols].values
        y_train = train_data[TARGET].values

        model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            verbosity=0,
        )

        fit_kwargs: Dict[str, Any] = {}
        if val_data is not None and len(val_data) > 0:
            X_val = val_data[self.feature_cols].values
            y_val = val_data[TARGET].values
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False

        logger.info("Training XGBoost on %d samples, %d features", len(X_train), len(self.feature_cols))
        model.fit(X_train, y_train, **fit_kwargs)

        self.model = model
        # Keep tail of training data for recursive forecasting
        self._train_tail = train_data.copy()
        self.is_fitted = True
        logger.info("XGBoost training complete")

    def predict(self, steps: int, **kwargs: Any) -> pd.DataFrame:
        """
        Recursive (autoregressive) forecasting.

        For each step:
        1. Build the feature row from the most recent known values.
        2. Predict the next value.
        3. Append prediction as a new "known" value.
        4. Repeat.
        """
        if not self.is_fitted or self.model is None or self._train_tail is None:
            raise RuntimeError("XGBoost model is not fitted.")

        import holidays as holidays_lib

        history = self._train_tail[[TARGET]].copy()
        last_date = history.index[-1]
        predictions = []
        future_dates = []

        us_holidays = holidays_lib.country_holidays(
            cfg.holiday_country,
            years=[last_date.year, last_date.year + 1, last_date.year + 2],
        )

        for step in range(1, steps + 1):
            next_date = last_date + pd.Timedelta(weeks=step)
            future_dates.append(next_date)

            # Build feature vector from history
            features = self._build_recursive_features(history, next_date, us_holidays)
            pred = float(self.model.predict(np.array([features]))[0])
            predictions.append(pred)

            # Append prediction to history for next iteration
            new_row = pd.DataFrame(
                {TARGET: [pred]}, index=pd.DatetimeIndex([next_date])
            )
            history = pd.concat([history, new_row])

        # Simple confidence intervals (±10% for XGBoost since it doesn't natively support CI)
        preds = np.array(predictions)
        margin = np.abs(preds) * 0.10

        return pd.DataFrame({
            "date": [d.strftime("%Y-%m-%d") for d in future_dates],
            "predicted_sales": predictions,
            "lower_bound": (preds - margin).tolist(),
            "upper_bound": (preds + margin).tolist(),
        })

    def _build_recursive_features(
        self,
        history: pd.DataFrame,
        target_date: pd.Timestamp,
        us_holidays: Any,
    ) -> List[float]:
        """Build a single feature row for *target_date* from *history*."""
        series = history[TARGET]

        features: List[float] = []

        # Lag features
        for lag in cfg.lag_periods:
            idx = len(series) - lag
            features.append(float(series.iloc[idx]) if idx >= 0 else 0.0)

        # Rolling features
        for window in cfg.rolling_windows:
            tail = series.iloc[-(window + 1):-1] if len(series) > window else series
            features.append(float(tail.mean()))
            features.append(float(tail.std()) if len(tail) > 1 else 0.0)

        # Date features
        features.append(float(target_date.dayofweek))
        features.append(float(target_date.isocalendar()[1]))
        features.append(float(target_date.month))
        features.append(float(target_date.quarter))
        features.append(float(target_date.year))
        features.append(1.0 if target_date.dayofweek >= 5 else 0.0)

        # Holiday
        week_start = target_date - pd.Timedelta(days=6)
        week_dates = pd.date_range(week_start, target_date)
        is_hol = 1.0 if any(d in us_holidays for d in week_dates) else 0.0
        features.append(is_hol)

        return features

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance scores."""
        if not self.is_fitted or self.model is None:
            return {}
        importances = self.model.feature_importances_
        return dict(zip(self.feature_cols, importances.tolist()))

    def save_model(self, path: Path) -> None:
        ensure_dir(path.parent)
        save_artifact(
            {
                "model": self.model,
                "feature_cols": self.feature_cols,
                "train_tail": self._train_tail,
            },
            path,
        )

    def load_model(self, path: Path) -> None:
        data = load_artifact(path)
        self.model = data["model"]
        self.feature_cols = data["feature_cols"]
        self._train_tail = data.get("train_tail")
        self.is_fitted = True
        logger.info("XGBoost model loaded from %s", path)
