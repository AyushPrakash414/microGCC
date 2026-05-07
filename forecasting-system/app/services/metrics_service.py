"""
Metrics Service.

Reads and returns model comparison metrics from the centralized CSV registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def get_all_metrics() -> Dict[str, Any]:
    """
    Load the centralized metrics CSV and return structured results.

    Returns:
        Dict matching ``MetricsResponse`` schema.
    """
    csv_path = settings.metrics_dir / "all_metrics.csv"
    if not csv_path.exists():
        return {"total_states": 0, "results": []}

    df = pd.read_csv(csv_path)

    # Keep only the latest version per (state, model)
    df = df.sort_values("timestamp").drop_duplicates(
        subset=["state", "model"], keep="last"
    )

    results: List[Dict[str, Any]] = []
    for state, group in df.groupby("state"):
        best_row = group.loc[group["rmse"].idxmin()]
        models = []
        for _, row in group.iterrows():
            models.append({
                "model": row["model"],
                "rmse": round(float(row["rmse"]), 4),
                "mae": round(float(row["mae"]), 4),
                "mape": round(float(row["mape"]), 4),
                "training_time_seconds": round(float(row.get("training_time_seconds", 0)), 3),
            })
        results.append({
            "state": str(state),
            "best_model": str(best_row["model"]),
            "models": models,
        })

    return {
        "total_states": len(results),
        "results": results,
    }


def get_state_metrics(state: str) -> Dict[str, Any] | None:
    """Return metrics for a single state."""
    all_metrics = get_all_metrics()
    for entry in all_metrics["results"]:
        if entry["state"] == state:
            return entry
    return None
