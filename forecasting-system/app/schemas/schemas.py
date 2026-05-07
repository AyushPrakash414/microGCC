"""
Pydantic Request / Response Schemas.

Defines typed API contracts for all FastAPI endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "running"})
    version: str = Field(..., json_schema_extra={"example": "1.0.0"})
    models_loaded: int = Field(..., json_schema_extra={"example": 4})
    supported_states: List[str] = Field(
        ..., json_schema_extra={"example": ["California", "Texas"]}
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
class TrainResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "training_started"})
    message: str = Field(..., json_schema_extra={"example": "Background training initiated for 43 states."})
    job_id: Optional[str] = None


class TrainStatusResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "completed"})
    states_trained: int = Field(..., json_schema_extra={"example": 43})
    states_skipped: int = Field(..., json_schema_extra={"example": 0})
    duration_seconds: Optional[float] = None


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------
class ForecastPoint(BaseModel):
    date: str = Field(..., json_schema_extra={"example": "2024-01-07"})
    predicted_sales: float = Field(..., json_schema_extra={"example": 345678.0})
    lower_bound: Optional[float] = Field(None, json_schema_extra={"example": 310000.0})
    upper_bound: Optional[float] = Field(None, json_schema_extra={"example": 380000.0})


class ForecastResponse(BaseModel):
    state: str = Field(..., json_schema_extra={"example": "California"})
    best_model: str = Field(..., json_schema_extra={"example": "xgboost"})
    forecast_horizon_weeks: int = Field(..., json_schema_extra={"example": 8})
    forecast: List[ForecastPoint]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class ModelMetric(BaseModel):
    model: str
    rmse: float
    mae: float
    mape: float
    training_time_seconds: Optional[float] = None


class StateMetrics(BaseModel):
    state: str
    best_model: str
    models: List[ModelMetric]


class MetricsResponse(BaseModel):
    total_states: int
    results: List[StateMetrics]


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------
class ErrorResponse(BaseModel):
    detail: str = Field(..., json_schema_extra={"example": "State 'FooBar' not found."})
