"""
Training Service.

Bridges the API layer with the training pipeline.
"""

from __future__ import annotations

from typing import Any, Dict

from app.core.logger import get_logger
from app.pipelines.training_pipeline import run_training_pipeline, get_training_status

logger = get_logger(__name__)


def start_training(parallel: bool = True) -> Dict[str, Any]:
    """
    Trigger a full training run.

    This is designed to be called from a BackgroundTask so it
    does not block the HTTP response.
    """
    logger.info("Training service: starting full pipeline …")
    try:
        result = run_training_pipeline(parallel=parallel, max_workers=4)
        return result
    except Exception as e:
        logger.error("Training service error: %s", e)
        return {"status": "failed", "error": str(e)}


def training_status() -> Dict[str, Any]:
    """Return the current status of the training pipeline."""
    return get_training_status()
