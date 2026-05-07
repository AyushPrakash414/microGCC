"""
Forecast Service.

Provides forecast results with a caching layer for repeated requests.
"""

from __future__ import annotations

from typing import Any, Dict

from cachetools import TTLCache

from app.core.logger import get_logger
from app.pipelines.forecasting_pipeline import run_forecasting_pipeline

logger = get_logger(__name__)

# Cache forecasts for 10 minutes (keyed by state)
_forecast_cache: TTLCache = TTLCache(maxsize=100, ttl=600)


def get_forecast(state: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    Return forecast for *state*, using cache when available.

    Args:
        state: US state name.
        use_cache: Whether to use cached results.

    Returns:
        Dict matching ``ForecastResponse`` schema.
    """
    if use_cache and state in _forecast_cache:
        logger.info("Cache hit for state: %s", state)
        return _forecast_cache[state]

    logger.info("Generating forecast for state: %s", state)
    result = run_forecasting_pipeline(state)
    _forecast_cache[state] = result
    return result


def clear_cache() -> None:
    """Flush the forecast cache."""
    _forecast_cache.clear()
    logger.info("Forecast cache cleared.")
