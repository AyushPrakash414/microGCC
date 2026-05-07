"""
Metrics API Route.

GET /metrics       — Returns model comparison metrics for all states.
GET /metrics/{state} — Returns metrics for a single state.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.schemas.schemas import MetricsResponse, StateMetrics, ErrorResponse
from app.services.metrics_service import get_all_metrics, get_state_metrics
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.get(
    "",
    response_model=MetricsResponse,
    summary="Get all model comparison metrics",
    description=(
        "Returns RMSE, MAE, MAPE, and training time for every model "
        "across all states, including the best model selection."
    ),
)
async def metrics_all():
    """Return global metrics."""
    try:
        data = get_all_metrics()
        return MetricsResponse(**data)
    except Exception as e:
        logger.error("Metrics error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get(
    "/{state}",
    response_model=StateMetrics,
    responses={404: {"model": ErrorResponse}},
    summary="Get metrics for a single state",
)
async def metrics_state(state: str):
    """Return metrics for *state*."""
    result = get_state_metrics(state)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No metrics found for state '{state}'.",
        )
    return StateMetrics(**result)
