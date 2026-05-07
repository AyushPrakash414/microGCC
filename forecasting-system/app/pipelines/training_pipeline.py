"""
Training Pipeline.

End-to-end orchestration:
1. Load & preprocess data
2. Train all models per state (with optional parallelism)
3. Evaluate, select best, save models & metadata
4. Generate reports
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from app.core.config import get_settings, get_yaml_config
from app.core.logger import get_logger
from app.models.model_selector import train_and_evaluate_state
from app.pipelines.preprocessing import run_preprocessing_pipeline
from app.utils.helpers import ensure_dir, save_json, timer, timestamp_now

logger = get_logger(__name__)
settings = get_settings()
cfg = get_yaml_config()

# Global training status
_training_status: Dict[str, Any] = {"status": "idle"}


def get_training_status() -> Dict[str, Any]:
    """Return current training status."""
    return _training_status


def _train_single_state(args: tuple) -> Dict[str, Any]:
    """Worker function for parallel training."""
    state, state_df = args
    try:
        return train_and_evaluate_state(state, state_df)
    except Exception as e:
        logger.error("Training failed for state %s: %s", state, e)
        return {"state": state, "error": str(e)}


def run_training_pipeline(
    parallel: bool = True,
    max_workers: int = 4,
) -> Dict[str, Any]:
    """
    Execute the full training pipeline.

    Args:
        parallel: If True, train states concurrently.
        max_workers: Max threads for parallel training.

    Returns:
        Summary dict with per-state results.
    """
    global _training_status

    _training_status = {"status": "running", "started_at": timestamp_now()}

    with timer("Full training pipeline") as t:
        # 1. Preprocess
        logger.info("=" * 70)
        logger.info("STAGE 1: Data Preprocessing")
        logger.info("=" * 70)
        state_datasets = run_preprocessing_pipeline()

        states_trained = 0
        states_skipped = 0
        all_results: Dict[str, Any] = {}

        # 2. Train
        logger.info("=" * 70)
        logger.info("STAGE 2: Model Training (%d states)", len(state_datasets))
        logger.info("=" * 70)

        if parallel and len(state_datasets) > 1:
            logger.info("Using parallel training with %d workers", max_workers)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_train_single_state, (state, df)): state
                    for state, df in state_datasets.items()
                }
                for future in concurrent.futures.as_completed(futures):
                    state = futures[future]
                    try:
                        result = future.result()
                        all_results[state] = result
                        if "error" not in result:
                            states_trained += 1
                        else:
                            states_skipped += 1
                    except Exception as e:
                        logger.error("State %s failed: %s", state, e)
                        all_results[state] = {"error": str(e)}
                        states_skipped += 1
        else:
            for state, df in state_datasets.items():
                try:
                    result = train_and_evaluate_state(state, df)
                    all_results[state] = result
                    states_trained += 1
                except Exception as e:
                    logger.error("Training failed for %s: %s", state, e)
                    all_results[state] = {"error": str(e)}
                    states_skipped += 1

        # 3. Generate reports
        logger.info("=" * 70)
        logger.info("STAGE 3: Report Generation")
        logger.info("=" * 70)
        _generate_reports(all_results)

    summary = {
        "status": "completed",
        "states_trained": states_trained,
        "states_skipped": states_skipped,
        "total_states": len(state_datasets),
        "duration_seconds": t["elapsed"],
        "timestamp": timestamp_now(),
    }

    # Save pipeline summary
    save_json(summary, settings.metadata_dir / "pipeline_summary.json")
    save_json(
        {s: r.get("best_model", "N/A") for s, r in all_results.items() if "error" not in r},
        settings.metadata_dir / "best_models.json",
    )

    _training_status = summary
    logger.info("Training pipeline complete: %s", summary)
    return summary


def _generate_reports(results: Dict[str, Any]) -> None:
    """Generate comparison charts and summary reports."""
    reports_dir = settings.reports_dir
    ensure_dir(reports_dir)

    # Model comparison bar chart
    model_wins: Dict[str, int] = {}
    states: List[str] = []
    rmses: Dict[str, List[float]] = {m: [] for m in ["arima", "prophet", "xgboost", "lstm"]}

    for state, result in results.items():
        if "error" in result or "models" not in result:
            continue
        states.append(state)
        best = result.get("best_model", "")
        model_wins[best] = model_wins.get(best, 0) + 1
        for model_name in rmses:
            val = result["models"].get(model_name, {}).get("rmse", float("nan"))
            rmses[model_name].append(val if val != float("inf") else float("nan"))

    if not states:
        logger.warning("No valid results to generate reports.")
        return

    # --- Best model distribution pie chart ---
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(
            model_wins.values(),
            labels=model_wins.keys(),
            autopct="%1.1f%%",
            startangle=140,
            colors=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"],
        )
        ax.set_title("Best Model Distribution Across States")
        fig.tight_layout()
        fig.savefig(reports_dir / "model_distribution.png", dpi=150)
        plt.close(fig)
        logger.info("Report saved: model_distribution.png")
    except Exception as e:
        logger.error("Failed to generate pie chart: %s", e)

    # --- RMSE comparison chart (first 10 states) ---
    try:
        display_states = states[:10]
        x = range(len(display_states))
        fig, ax = plt.subplots(figsize=(14, 7))
        width = 0.2
        for i, (model_name, values) in enumerate(rmses.items()):
            display_vals = values[:10]
            ax.bar(
                [xi + i * width for xi in x],
                display_vals,
                width=width,
                label=model_name.upper(),
            )
        ax.set_xlabel("State")
        ax.set_ylabel("RMSE")
        ax.set_title("Model RMSE Comparison by State")
        ax.set_xticks([xi + 1.5 * width for xi in x])
        ax.set_xticklabels(display_states, rotation=45, ha="right")
        ax.legend()
        fig.tight_layout()
        fig.savefig(reports_dir / "rmse_comparison.png", dpi=150)
        plt.close(fig)
        logger.info("Report saved: rmse_comparison.png")
    except Exception as e:
        logger.error("Failed to generate RMSE chart: %s", e)

    # --- Training time comparison ---
    try:
        training_times: Dict[str, float] = {}
        for state, result in results.items():
            if "error" in result or "models" not in result:
                continue
            for model_name, info in result["models"].items():
                training_times.setdefault(model_name, 0)
                training_times[model_name] += info.get("training_time_seconds", 0)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(
            list(training_times.keys()),
            list(training_times.values()),
            color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"],
        )
        ax.set_xlabel("Total Training Time (seconds)")
        ax.set_title("Total Training Time by Model")
        fig.tight_layout()
        fig.savefig(reports_dir / "training_time.png", dpi=150)
        plt.close(fig)
        logger.info("Report saved: training_time.png")
    except Exception as e:
        logger.error("Failed to generate training time chart: %s", e)
