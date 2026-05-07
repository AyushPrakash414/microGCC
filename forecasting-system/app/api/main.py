"""
FastAPI Application Entry Point.

Registers all routers, middleware, and the enhanced health endpoint.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import forecast, metrics, train
from app.core.config import get_settings, PROJECT_ROOT
from app.core.constants import API_DESCRIPTION, API_TITLE, API_VERSION
from app.core.logger import get_logger
from app.schemas.schemas import HealthResponse

logger = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup and shutdown hooks."""
    logger.info(
        "%s v%s started (env=%s)",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.APP_ENV,
    )
    yield
    logger.info("Application shutting down.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(train.router)
app.include_router(forecast.router)
app.include_router(metrics.router)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------
def _count_trained_models() -> int:
    """Count the number of state directories that contain at least one version."""
    models_dir = settings.saved_models_dir
    if not models_dir.exists():
        return 0
    return sum(
        1 for d in models_dir.iterdir()
        if d.is_dir() and any(v.is_dir() and v.name.startswith("v") for v in d.iterdir())
    )


def _get_supported_states() -> List[str]:
    """List states that have trained models available."""
    models_dir = settings.saved_models_dir
    if not models_dir.exists():
        return []
    return sorted([
        d.name for d in models_dir.iterdir()
        if d.is_dir() and any(v.is_dir() and v.name.startswith("v") for v in d.iterdir())
    ])


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
    description="Returns application status, version, loaded models count, and supported states.",
)
async def health_check():
    """Enhanced health endpoint with production diagnostics."""
    return HealthResponse(
        status="running",
        version=settings.APP_VERSION,
        models_loaded=_count_trained_models(),
        supported_states=_get_supported_states(),
    )
