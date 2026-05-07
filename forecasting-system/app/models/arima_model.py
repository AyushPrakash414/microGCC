"""
ARIMA / SARIMAX Forecasting Model.

Uses statsmodels SARIMAX for trend + seasonal time-series forecasting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from app.core.config import get_yaml_config
from app.core.logger import get_logger
from app.models.base_model import BaseForecaster
from app.utils.helpers import ensure_dir, save_artifact, load_artifact

logger = get_logger(__name__)
cfg = get_yaml_config()


class ARIMAForecaster(BaseForecaster):
    """SARIMAX-based forecaster with trend and seasonal components."""

    name = "arima"

    def __init__(self) -> None:
        super().__init__()
        params = cfg.model_params("arima")
        self.order = tuple(params.get("order", [1, 1, 1]))
        self.seasonal_order = tuple(params.get("seasonal_order", [1, 1, 1, 52]))
        self.enforce_stationarity = params.get("enforce_stationarity", False)
        self.enforce_invertibility = params.get("enforce_invertibility", False)
        self._train_series: pd.Series | None = None

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> None:
        """Fit SARIMAX on the target column of *train_data*."""
        from app.core.constants import TARGET

        y = train_data[TARGET].astype(float)
        self._train_series = y

        logger.info(
            "Training SARIMAX order=%s seasonal=%s on %d obs",
            self.order, self.seasonal_order, len(y),
        )

        try:
            model = SARIMAX(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility,
            )
            self.model = model.fit(disp=False, maxiter=200)
            self.is_fitted = True
            logger.info("SARIMAX training complete (AIC=%.2f)", self.model.aic)
        except Exception as e:
            logger.error("SARIMAX training failed: %s", e)
            # Fallback to simpler order
            logger.info("Attempting fallback with order=(1,1,0), no seasonal")
            model = SARIMAX(
                y,
                order=(1, 1, 0),
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.model = model.fit(disp=False, maxiter=200)
            self.is_fitted = True
            logger.info("SARIMAX fallback training complete")

    def predict(self, steps: int, **kwargs: Any) -> pd.DataFrame:
        """Forecast *steps* weeks ahead with confidence intervals."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("ARIMA model is not fitted.")

        forecast = self.model.get_forecast(steps=steps)
        pred_mean = forecast.predicted_mean
        conf = forecast.conf_int(alpha=0.05)

        last_date = self._train_series.index[-1] if self._train_series is not None else pd.Timestamp.now()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(weeks=1),
            periods=steps,
            freq="W",
        )

        return pd.DataFrame({
            "date": future_dates.strftime("%Y-%m-%d"),
            "predicted_sales": pred_mean.values,
            "lower_bound": conf.iloc[:, 0].values,
            "upper_bound": conf.iloc[:, 1].values,
        })

    def save_model(self, path: Path) -> None:
        ensure_dir(path.parent)
        save_artifact(
            {"model": self.model, "train_series": self._train_series},
            path,
        )

    def load_model(self, path: Path) -> None:
        data = load_artifact(path)
        self.model = data["model"]
        self._train_series = data.get("train_series")
        self.is_fitted = True
        logger.info("ARIMA model loaded from %s", path)
