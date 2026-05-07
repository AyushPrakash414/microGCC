"""
LSTM Forecasting Model.

Deep-learning sequence model using TensorFlow / Keras with:
- MinMaxScaler normalisation
- Sliding-window sequence generation
- Recursive multi-step forecasting
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from app.core.config import get_yaml_config
from app.core.constants import TARGET
from app.core.logger import get_logger
from app.models.base_model import BaseForecaster
from app.utils.helpers import ensure_dir, save_artifact, load_artifact

logger = get_logger(__name__)
cfg = get_yaml_config()

# Suppress TF info logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


class LSTMForecaster(BaseForecaster):
    """LSTM-based forecaster with MinMaxScaler and recursive prediction."""

    name = "lstm"

    def __init__(self) -> None:
        super().__init__()
        params = cfg.model_params("lstm")
        self.units = params.get("units", 64)
        self.dropout = params.get("dropout", 0.2)
        self.epochs = params.get("epochs", 100)
        self.batch_size = params.get("batch_size", 16)
        self.sequence_length = params.get("sequence_length", 12)
        self.patience = params.get("patience", 10)

        self.scaler = MinMaxScaler()
        self._last_sequence: np.ndarray | None = None
        self._last_date: pd.Timestamp | None = None

    # ------------------------------------------------------------------
    # Sequence helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _create_sequences(
        data: np.ndarray, seq_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding-window (X, y) sequences."""
        X, y = [], []
        for i in range(seq_len, len(data)):
            X.append(data[i - seq_len : i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    def _build_model(self, input_shape: Tuple[int, int]) -> Any:
        """Build and compile a Keras LSTM model."""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout

        model = Sequential([
            LSTM(self.units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout),
            LSTM(self.units // 2, return_sequences=False),
            Dropout(self.dropout),
            Dense(32, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------
    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> None:
        from tensorflow.keras.callbacks import EarlyStopping

        logger.info("Training LSTM on %d observations", len(train_data))

        values = train_data[TARGET].values.reshape(-1, 1)
        scaled = self.scaler.fit_transform(values)

        X, y = self._create_sequences(scaled, self.sequence_length)
        if len(X) == 0:
            logger.warning("Not enough data for LSTM sequences; skipping.")
            return

        # Reshape: (samples, timesteps, features)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Validation split
        callbacks = []
        val_split = 0.0
        if val_data is not None and len(val_data) > self.sequence_length:
            val_values = np.concatenate([values[-self.sequence_length:], val_data[TARGET].values.reshape(-1, 1)])
            val_scaled = self.scaler.transform(val_values)
            X_val, y_val = self._create_sequences(val_scaled, self.sequence_length)
            if len(X_val) > 0:
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
                callbacks.append(
                    EarlyStopping(
                        monitor="val_loss",
                        patience=self.patience,
                        restore_best_weights=True,
                    )
                )
        else:
            X_val, y_val = None, None
            val_split = 0.1
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.patience,
                    restore_best_weights=True,
                )
            )

        model = self._build_model((self.sequence_length, 1))

        fit_kwargs: Dict[str, Any] = dict(
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
        )
        if X_val is not None and y_val is not None and len(X_val) > 0:
            fit_kwargs["validation_data"] = (X_val, y_val)
        else:
            fit_kwargs["validation_split"] = val_split

        model.fit(X, y, **fit_kwargs)

        self.model = model
        self._last_sequence = scaled[-self.sequence_length:]
        self._last_date = train_data.index[-1]
        self.is_fitted = True
        logger.info("LSTM training complete")

    def predict(self, steps: int, **kwargs: Any) -> pd.DataFrame:
        """Recursive multi-step LSTM forecasting."""
        if not self.is_fitted or self.model is None or self._last_sequence is None:
            raise RuntimeError("LSTM model is not fitted.")

        current_seq = self._last_sequence.copy()
        predictions_scaled: List[float] = []

        for _ in range(steps):
            x_input = current_seq.reshape(1, self.sequence_length, 1)
            pred_scaled = float(self.model.predict(x_input, verbose=0)[0, 0])
            predictions_scaled.append(pred_scaled)
            # Shift window
            current_seq = np.append(current_seq[1:], [[pred_scaled]], axis=0)

        # Inverse transform
        preds = self.scaler.inverse_transform(
            np.array(predictions_scaled).reshape(-1, 1)
        ).flatten()

        last_date = self._last_date or pd.Timestamp.now()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(weeks=1),
            periods=steps,
            freq="W",
        )

        # Confidence intervals (±15% heuristic for LSTM)
        margin = np.abs(preds) * 0.15

        return pd.DataFrame({
            "date": future_dates.strftime("%Y-%m-%d"),
            "predicted_sales": preds,
            "lower_bound": (preds - margin).tolist(),
            "upper_bound": (preds + margin).tolist(),
        })

    def save_model(self, path: Path) -> None:
        ensure_dir(path.parent)
        # Save Keras model
        self.model.save(str(path))
        # Save scaler & metadata alongside
        meta_path = path.parent / f"{path.stem}_meta.pkl"
        save_artifact(
            {
                "scaler": self.scaler,
                "last_sequence": self._last_sequence,
                "last_date": self._last_date,
                "sequence_length": self.sequence_length,
            },
            meta_path,
        )

    def load_model(self, path: Path) -> None:
        from tensorflow.keras.models import load_model as keras_load

        self.model = keras_load(str(path))
        meta_path = path.parent / f"{path.stem}_meta.pkl"
        if meta_path.exists():
            meta = load_artifact(meta_path)
            self.scaler = meta["scaler"]
            self._last_sequence = meta["last_sequence"]
            self._last_date = meta.get("last_date")
            self.sequence_length = meta.get("sequence_length", self.sequence_length)
        self.is_fitted = True
        logger.info("LSTM model loaded from %s", path)
