"""
Test Suite for the Time Series Forecasting System.

Covers:
- API endpoints
- Preprocessing
- Feature engineering
- Metrics utilities
- Validators
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================================
# Metrics Tests
# =========================================================================
class TestMetrics:
    """Test evaluation metrics."""

    def test_rmse(self):
        from app.utils.metrics import rmse

        y_true = [3, -0.5, 2, 7]
        y_pred = [2.5, 0.0, 2, 8]
        result = rmse(y_true, y_pred)
        assert isinstance(result, float)
        assert result > 0

    def test_mae(self):
        from app.utils.metrics import mae

        y_true = [3, -0.5, 2, 7]
        y_pred = [2.5, 0.0, 2, 8]
        result = mae(y_true, y_pred)
        assert isinstance(result, float)
        assert result >= 0

    def test_mape(self):
        from app.utils.metrics import mape

        y_true = [100, 200, 300]
        y_pred = [110, 190, 310]
        result = mape(y_true, y_pred)
        assert isinstance(result, float)
        assert result > 0
        assert result < 100

    def test_mape_zero_safe(self):
        from app.utils.metrics import mape

        y_true = [0, 0, 0]
        y_pred = [1, 2, 3]
        result = mape(y_true, y_pred)
        assert result == 0.0

    def test_compute_all_metrics(self):
        from app.utils.metrics import compute_all_metrics

        y_true = [100, 200, 300, 400]
        y_pred = [110, 190, 310, 420]
        result = compute_all_metrics(y_true, y_pred)
        assert "rmse" in result
        assert "mae" in result
        assert "mape" in result


# =========================================================================
# Validators Tests
# =========================================================================
class TestValidators:
    """Test data validation utilities."""

    def test_validate_raw_dataframe_valid(self):
        from app.utils.validators import validate_raw_dataframe

        df = pd.DataFrame({
            "State": ["CA", "CA"],
            "Date": ["2023-01-01", "2023-01-08"],
            "Total": [100.0, 200.0],
        })
        warnings = validate_raw_dataframe(df)
        assert isinstance(warnings, list)

    def test_validate_raw_dataframe_missing_columns(self):
        from app.utils.validators import validate_raw_dataframe, ValidationError

        df = pd.DataFrame({"Foo": [1, 2]})
        with pytest.raises(ValidationError, match="Missing required columns"):
            validate_raw_dataframe(df)

    def test_validate_raw_dataframe_empty(self):
        from app.utils.validators import validate_raw_dataframe, ValidationError

        df = pd.DataFrame(columns=["State", "Date", "Total"])
        with pytest.raises(ValidationError, match="empty"):
            validate_raw_dataframe(df)

    def test_validate_state_series_insufficient(self):
        from app.utils.validators import validate_state_series

        df = pd.DataFrame({"Total": [1, 2, 3]})
        valid, msg = validate_state_series("TestState", df, min_weeks=52)
        assert valid is False

    def test_validate_state_name(self):
        from app.utils.validators import validate_state_name, ValidationError

        validate_state_name("California", ["California", "Texas"])
        with pytest.raises(ValidationError):
            validate_state_name("FooBar", ["California", "Texas"])


# =========================================================================
# Helpers Tests
# =========================================================================
class TestHelpers:
    """Test helper utilities."""

    def test_get_next_version_new(self, tmp_path):
        from app.utils.helpers import get_next_version

        result = get_next_version(tmp_path / "nonexistent")
        assert result == "v1"

    def test_get_next_version_existing(self, tmp_path):
        from app.utils.helpers import get_next_version

        (tmp_path / "v1").mkdir()
        (tmp_path / "v2").mkdir()
        result = get_next_version(tmp_path)
        assert result == "v3"

    def test_timer(self):
        from app.utils.helpers import timer
        import time

        with timer("test") as t:
            time.sleep(0.05)
        assert "elapsed" in t
        assert t["elapsed"] >= 0.04


# =========================================================================
# Feature Engineering Tests
# =========================================================================
class TestFeatureEngineering:
    """Test feature engineering pipeline."""

    def _make_sample_df(self, n=100):
        dates = pd.date_range("2020-01-05", periods=n, freq="W")
        return pd.DataFrame(
            {"Total": np.random.randint(1000, 5000, size=n)},
            index=dates,
        )

    def test_engineer_features(self):
        from app.features.feature_engineering import engineer_features

        df = self._make_sample_df(100)
        result = engineer_features(df)
        assert "lag_1" in result.columns
        assert "rolling_mean_7" in result.columns
        assert "month" in result.columns
        assert "is_holiday" in result.columns
        assert len(result) < 100  # burn-in rows dropped

    def test_get_feature_columns(self):
        from app.features.feature_engineering import get_feature_columns

        cols = get_feature_columns()
        assert isinstance(cols, list)
        assert "lag_1" in cols
        assert "is_holiday" in cols


# =========================================================================
# API Tests (requires FastAPI TestClient)
# =========================================================================
class TestAPI:
    """Test FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app.api.main import app

        return TestClient(app)

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert "version" in data
        assert "supported_states" in data

    def test_metrics_empty(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_forecast_no_model(self, client):
        response = client.get("/forecast/NonExistentState")
        assert response.status_code in (404, 500)
