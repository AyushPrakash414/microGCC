"""
General Helper Utilities.

File I/O helpers, directory management, and miscellaneous utilities.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List

import joblib
import pandas as pd

from app.core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------
def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Model persistence helpers
# ---------------------------------------------------------------------------
def save_artifact(obj: Any, filepath: Path) -> None:
    """Save a Python object using joblib."""
    ensure_dir(filepath.parent)
    joblib.dump(obj, filepath)
    logger.info("Artifact saved -> %s", filepath)


def load_artifact(filepath: Path) -> Any:
    """Load a Python object using joblib."""
    if not filepath.exists():
        raise FileNotFoundError(f"Artifact not found: {filepath}")
    return joblib.load(filepath)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def save_json(data: Any, filepath: Path) -> None:
    """Write data to a JSON file."""
    ensure_dir(filepath.parent)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("JSON saved -> %s", filepath)


def load_json(filepath: Path) -> Any:
    """Read data from a JSON file."""
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def append_metrics_csv(
    filepath: Path,
    records: List[Dict[str, Any]],
) -> None:
    """Append metric records to a CSV file, creating it if needed."""
    ensure_dir(filepath.parent)
    df = pd.DataFrame(records)
    write_header = not filepath.exists()
    df.to_csv(filepath, mode="a", header=write_header, index=False)
    logger.info("Metrics appended -> %s (%d records)", filepath, len(records))


# ---------------------------------------------------------------------------
# Timing context manager
# ---------------------------------------------------------------------------
@contextmanager
def timer(label: str = "Operation") -> Generator[Dict[str, float], None, None]:
    """
    Context manager that measures elapsed wall-clock time.

    Usage::

        with timer("Training XGBoost") as t:
            model.fit(X, y)
        print(t["elapsed"])
    """
    result: Dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = round(time.perf_counter() - start, 3)
        logger.info("%s completed in %.3fs", label, result["elapsed"])


# ---------------------------------------------------------------------------
# Model version management
# ---------------------------------------------------------------------------
def get_next_version(state_dir: Path) -> str:
    """
    Determine the next version folder (v1, v2, …) inside a state directory.
    """
    if not state_dir.exists():
        return "v1"
    existing = sorted(
        [d.name for d in state_dir.iterdir() if d.is_dir() and d.name.startswith("v")],
        key=lambda x: int(x[1:]) if x[1:].isdigit() else 0,
    )
    if not existing:
        return "v1"
    last = int(existing[-1][1:])
    return f"v{last + 1}"


def get_latest_version(state_dir: Path) -> str | None:
    """Return the latest version folder name, or None."""
    if not state_dir.exists():
        return None
    existing = sorted(
        [d.name for d in state_dir.iterdir() if d.is_dir() and d.name.startswith("v")],
        key=lambda x: int(x[1:]) if x[1:].isdigit() else 0,
    )
    return existing[-1] if existing else None


def timestamp_now() -> str:
    """ISO-formatted timestamp."""
    return datetime.utcnow().isoformat() + "Z"
