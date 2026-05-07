"""
Core Configuration Module.

Loads settings from .env, config.yaml, and environment variables
using Pydantic Settings for type-safe configuration management.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import ConfigDict

# ---------------------------------------------------------------------------
# Resolve project root (forecasting-system/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Pydantic Settings
# ---------------------------------------------------------------------------
class AppSettings(BaseSettings):
    """Application-level settings sourced from environment variables."""

    # App
    APP_NAME: str = "Time Series Forecasting System"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    # Paths (relative to PROJECT_ROOT)
    DATA_PATH: str = "app/data/dataset.xlsx"
    SAVED_MODELS_PATH: str = "app/saved_models"
    METADATA_PATH: str = "metadata"
    METRICS_PATH: str = "metrics"
    REPORTS_PATH: str = "reports"

    # Forecasting
    FORECAST_HORIZON: int = 8
    VALIDATION_WEEKS: int = 8
    MIN_DATA_WEEKS: int = 52

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    model_config = ConfigDict(env_file=".env", extra="ignore")

    # ------------------------------------------------------------------
    # Convenience helpers that return absolute Path objects
    # ------------------------------------------------------------------
    @property
    def data_filepath(self) -> Path:
        return PROJECT_ROOT / self.DATA_PATH

    @property
    def saved_models_dir(self) -> Path:
        return PROJECT_ROOT / self.SAVED_MODELS_PATH

    @property
    def metadata_dir(self) -> Path:
        return PROJECT_ROOT / self.METADATA_PATH

    @property
    def metrics_dir(self) -> Path:
        return PROJECT_ROOT / self.METRICS_PATH

    @property
    def reports_dir(self) -> Path:
        return PROJECT_ROOT / self.REPORTS_PATH


# ---------------------------------------------------------------------------
# YAML Configuration Loader
# ---------------------------------------------------------------------------
def _load_yaml_config() -> Dict[str, Any]:
    """Load config.yaml from project root."""
    config_path = PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


class YAMLConfig:
    """Typed access to hierarchical YAML configuration."""

    def __init__(self, raw: Dict[str, Any]) -> None:
        self._raw = raw

    # --- Forecasting ---
    @property
    def horizon(self) -> int:
        return self._raw.get("forecasting", {}).get("horizon", 8)

    @property
    def validation_weeks(self) -> int:
        return self._raw.get("forecasting", {}).get("validation_weeks", 8)

    @property
    def min_data_weeks(self) -> int:
        return self._raw.get("forecasting", {}).get("min_data_weeks", 52)

    @property
    def frequency(self) -> str:
        return self._raw.get("forecasting", {}).get("frequency", "W")

    # --- Features ---
    @property
    def lag_periods(self) -> List[int]:
        return self._raw.get("features", {}).get("lag_periods", [1, 7, 30])

    @property
    def rolling_windows(self) -> List[int]:
        return self._raw.get("features", {}).get("rolling_windows", [7, 30])

    @property
    def date_features(self) -> List[str]:
        return self._raw.get("features", {}).get("date_features", [])

    @property
    def holiday_country(self) -> str:
        return self._raw.get("features", {}).get("holiday_country", "US")

    # --- Model Hyper-parameters ---
    def model_params(self, model_name: str) -> Dict[str, Any]:
        return self._raw.get("models", {}).get(model_name, {})

    # --- Evaluation ---
    @property
    def primary_metric(self) -> str:
        return self._raw.get("evaluation", {}).get("primary_metric", "rmse")

    @property
    def eval_metrics(self) -> List[str]:
        return self._raw.get("evaluation", {}).get("metrics", ["rmse", "mae", "mape"])

    # --- Logging ---
    @property
    def log_level(self) -> str:
        return self._raw.get("logging", {}).get("level", "INFO")

    @property
    def log_format(self) -> str:
        return self._raw.get("logging", {}).get(
            "format", "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )

    @property
    def log_file(self) -> str:
        return self._raw.get("logging", {}).get("file", "logs/app.log")


# ---------------------------------------------------------------------------
# Singleton Accessors
# ---------------------------------------------------------------------------
@lru_cache()
def get_settings() -> AppSettings:
    """Return cached AppSettings instance."""
    return AppSettings()


@lru_cache()
def get_yaml_config() -> YAMLConfig:
    """Return cached YAMLConfig instance."""
    return YAMLConfig(_load_yaml_config())
