"""
Forecast API Route.

GET /forecast/{state}  — Returns 8-week sales forecast for a state.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.schemas.schemas import ForecastResponse, ErrorResponse
from app.services.forecast_service import get_forecast
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/forecast", tags=["Forecasting"])


@router.get(
    "/{state}",
    response_model=ForecastResponse,
    responses={
        404: {"model": ErrorResponse, "description": "State not found or no model trained."},
        500: {"model": ErrorResponse, "description": "Prediction error."},
    },
    summary="Get sales forecast for a state",
    description=(
        "Load the best trained model for the specified state and return "
        "the next 8 weeks of predicted sales with confidence intervals."
    ),
)
async def forecast_state(state: str):
    """Return 8-week forecast for *state*."""
    try:
        result = get_forecast(state)
        return ForecastResponse(**result)
    except FileNotFoundError as e:
        logger.warning("Forecast 404: %s", e)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error("Forecast error for %s: %s", state, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {e}",
        )
