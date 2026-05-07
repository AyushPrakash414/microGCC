"""
Abstract Base Model.

Every forecasting model inherits from this class and implements:
- train()
- predict()
- evaluate()
- save_model()
- load_model()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from app.utils.metrics import compute_all_metrics


class BaseForecaster(ABC):
    """Abstract base class for all forecasting models."""

    name: str = "base"

    def __init__(self) -> None:
        self.model: Any = None
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @abstractmethod
    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> None:
        """Fit the model on training data."""
        ...

    @abstractmethod
    def predict(self, steps: int, **kwargs: Any) -> pd.DataFrame:
        """
        Produce a forecast for *steps* periods ahead.

        Returns a DataFrame with columns:
        ``date``, ``predicted_sales``, ``lower_bound``, ``upper_bound``
        """
        ...

    @abstractmethod
    def save_model(self, path: Path) -> None:
        """Persist the trained model to disk."""
        ...

    @abstractmethod
    def load_model(self, path: Path) -> None:
        """Load a previously saved model from disk."""
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def evaluate(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
    ) -> Dict[str, float]:
        """Compute standard evaluation metrics."""
        return compute_all_metrics(y_true, y_pred)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} fitted={self.is_fitted}>"
