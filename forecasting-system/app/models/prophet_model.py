"""
Facebook Prophet Forecasting Model.

Handles seasonality, trend, and US holidays natively.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import get_yaml_config
from app.core.constants import PROPHET_DS, PROPHET_Y, TARGET
from app.core.logger import get_logger
from app.models.base_model import BaseForecaster
from app.utils.helpers import ensure_dir, save_artifact, load_artifact

logger = get_logger(__name__)
cfg = get_yaml_config()


class ProphetForecaster(BaseForecaster):
    """Facebook Prophet-based forecaster."""

    name = "prophet"

    def __init__(self) -> None:
        super().__init__()
        params = cfg.model_params("prophet")
        self.yearly_seasonality = params.get("yearly_seasonality", True)
        self.weekly_seasonality = params.get("weekly_seasonality", True)
        self.daily_seasonality = params.get("daily_seasonality", False)
        self.changepoint_prior_scale = params.get("changepoint_prior_scale", 0.05)
        self._last_date: pd.Timestamp | None = None

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert to Prophet expected format (ds, y)."""
        pdf = pd.DataFrame({
            PROPHET_DS: df.index,
            PROPHET_Y: df[TARGET].values,
        })
        return pdf

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> None:
        from prophet import Prophet

        logger.info("Training Prophet on %d observations", len(train_data))

        m = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
        )
        m.add_country_holidays(country_name=cfg.holiday_country)

        pdf = self._prepare_df(train_data)
        m.fit(pdf)

        self.model = m
        self._last_date = train_data.index[-1]
        self.is_fitted = True
        logger.info("Prophet training complete")

    def predict(self, steps: int, **kwargs: Any) -> pd.DataFrame:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Prophet model is not fitted.")

        future = self.model.make_future_dataframe(periods=steps, freq="W")
        forecast = self.model.predict(future)

        # Take only the forecasted rows
        forecast = forecast.tail(steps)

        return pd.DataFrame({
            "date": forecast[PROPHET_DS].dt.strftime("%Y-%m-%d").values,
            "predicted_sales": forecast["yhat"].values,
            "lower_bound": forecast["yhat_lower"].values,
            "upper_bound": forecast["yhat_upper"].values,
        })

    def save_model(self, path: Path) -> None:
        ensure_dir(path.parent)
        save_artifact(
            {"model": self.model, "last_date": self._last_date},
            path,
        )

    def load_model(self, path: Path) -> None:
        data = load_artifact(path)
        self.model = data["model"]
        self._last_date = data.get("last_date")
        self.is_fitted = True
        logger.info("Prophet model loaded from %s", path)
