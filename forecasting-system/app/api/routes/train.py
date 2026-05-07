"""
Training API Route.

POST /train  — Triggers background training for all states.
GET  /train/status — Returns current training status.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, status
from fastapi.responses import JSONResponse

from app.schemas.schemas import TrainResponse, TrainStatusResponse
from app.services.train_service import start_training, training_status

router = APIRouter(prefix="/train", tags=["Training"])


@router.post(
    "",
    response_model=TrainResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Train all forecasting models",
    description=(
        "Triggers a background training pipeline that preprocesses data, "
        "trains ARIMA, Prophet, XGBoost, and LSTM models for every state, "
        "evaluates them, selects the best, and persists artifacts."
    ),
)
async def train_models(background_tasks: BackgroundTasks):
    """Kick off the training pipeline in the background."""
    background_tasks.add_task(start_training, parallel=True)
    return TrainResponse(
        status="training_started",
        message="Background training initiated. Use GET /train/status to monitor.",
    )


@router.get(
    "/status",
    response_model=TrainStatusResponse,
    summary="Check training status",
    description="Returns the current status of the training pipeline.",
)
async def get_train_status():
    """Return training pipeline status."""
    s = training_status()
    return TrainStatusResponse(
        status=s.get("status", "idle"),
        states_trained=s.get("states_trained", 0),
        states_skipped=s.get("states_skipped", 0),
        duration_seconds=s.get("duration_seconds"),
    )
