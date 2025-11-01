from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from backend import config
from backend.utils.files import ensure_directory

logger = logging.getLogger(__name__)
LOOKBACK_HOURS = 7 * 24


@dataclass
class ForecastResult:
    path: Path
    history_start: datetime
    history_end: datetime


def run_forecast(
    data_path: Optional[Path] = None,
    output_dir: Path | str = config.PROCESSED_DATA_DIR,
    horizon_hours: int = config.DEFAULT_FORECAST_HORIZON_HOURS,
) -> ForecastResult:
    """Run the demand forecast pipeline and persist the horizon forecast."""
    output_dir = ensure_directory(Path(output_dir))
    demand_history = _load_demand_history(data_path)

    if demand_history.empty:
        raise ValueError("No demand data available to forecast.")

    history_end = demand_history["timestamp"].max()
    cutoff = history_end - timedelta(hours=LOOKBACK_HOURS - 1)
    trimmed_history = demand_history[demand_history["timestamp"] >= cutoff]
    if len(trimmed_history) < len(demand_history) and len(trimmed_history) < LOOKBACK_HOURS:
        logger.warning(
            "Limited data available in last 7 days (%d points); using full history of %d points.",
            len(trimmed_history),
            len(demand_history),
        )
        trimmed_history = demand_history

    history_start = trimmed_history["timestamp"].min()

    series = trimmed_history["demand_mw"].astype(float)
    timestamps = trimmed_history["timestamp"]

    model = LSTMForecaster()
    logger.info(
        "Training LSTM forecaster with %d historical points to predict %d hours.",
        len(series),
        horizon_hours,
    )

    forecast_values = model.forecast(series, horizon_hours)
    last_timestamp = timestamps.max()
    forecast_index = pd.date_range(
        start=last_timestamp + timedelta(hours=1),
        periods=horizon_hours,
        freq="H",
        tz=last_timestamp.tzinfo or timezone.utc,
    )

    forecast_df = pd.DataFrame(
        {
            "timestamp": forecast_index,
            "forecast_mw": forecast_values,
        }
    )
    output_path = output_dir / "forecast_next24.csv"
    forecast_df.to_csv(output_path, index=False)
    logger.info("Forecast saved to %s.", output_path)
    return ForecastResult(
        path=output_path,
        history_start=history_start,
        history_end=history_end,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_demand_history(data_path: Optional[Path]) -> pd.DataFrame:
    candidates: list[Path] = []
    if data_path and data_path.exists():
        candidates.append(data_path)
    else:
        raw_candidate = config.PROCESSED_DATA_DIR / "raw_latest.csv"
        demand_candidate = config.PROCESSED_DATA_DIR / "demand_latest.csv"
        for candidate in (raw_candidate, demand_candidate):
            if candidate.exists():
                candidates.append(candidate)
                break

    if candidates:
        path = candidates[0]
        logger.debug("Loading demand history from %s.", path)
        df = pd.read_csv(path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        else:
            raise ValueError(f"'timestamp' column not found in {path}")
        if "demand_mw" not in df.columns:
            raise ValueError(f"'demand_mw' column missing in {path}")
        df = df.sort_values("timestamp").dropna(subset=["demand_mw"])
        return df

    logger.warning(
        "Falling back to synthetic demand history because no dataset was provided."
    )
    return _generate_synthetic_history()


def _generate_synthetic_history(hours: int = 24 * 365) -> pd.DataFrame:
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    index = pd.date_range(end=now, periods=hours, freq="H", tz=timezone.utc)
    phase = np.linspace(0, np.pi * 4, len(index))
    base_load = 40000 + 5000 * np.sin(phase)
    noise = np.random.normal(scale=1500, size=len(index))
    demand = np.clip(base_load + noise, a_min=20000, a_max=None)
    return pd.DataFrame({"timestamp": index, "demand_mw": demand})


# ---------------------------------------------------------------------------
# LSTM forecaster with graceful fallback
# ---------------------------------------------------------------------------


@dataclass
class _TorchArtifacts:
    torch: Any
    nn: Any
    optim: Any
    utils: Any


class LSTMForecaster:
    """Minimal LSTM forecaster that gracefully degrades to a naive baseline."""

    def __init__(
        self,
        sequence_length: int = 24,
        hidden_size: int = 32,
        learning_rate: float = 5e-3,
        epochs: int = 20,
    ) -> None:
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._torch = self._load_torch()

    def forecast(self, series: pd.Series, horizon: int) -> list[float]:
        if series.empty:
            raise ValueError("Input series for forecasting is empty.")

        if self._torch is None or len(series) < self.sequence_length + 1:
            logger.warning(
                "Torch not available or insufficient data for LSTM; using naive forecast."
            )
            return self._naive_forecast(series, horizon)

        artifacts = self._torch
        torch = artifacts.torch
        nn = artifacts.nn
        optim = artifacts.optim

        normalized_series, mean, std = self._normalize(series.to_numpy(dtype="float32"))
        data = torch.tensor(normalized_series, dtype=torch.float32).view(-1, 1)

        seq_len = min(self.sequence_length, len(data) - 1)
        model = _LSTMModel(input_size=1, hidden_size=self.hidden_size)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        total_sequences = max(len(data) - seq_len - 1, 0)
        if total_sequences == 0:
            logger.warning("Insufficient history for LSTM; falling back to naive forecast.")
            return self._naive_forecast(series, horizon)

        max_sequences = 256
        stride = max(1, total_sequences // max_sequences)

        for epoch in range(self.epochs):
            loss_accumulator = 0.0
            steps = 0
            for start_idx in range(0, total_sequences, stride):
                optimizer.zero_grad()
                sequence = data[start_idx : start_idx + seq_len].unsqueeze(0)
                target = data[start_idx + seq_len].view(1)
                prediction = model(sequence).view(1)
                loss = criterion(prediction, target)
                loss.backward()
                optimizer.step()
                loss_accumulator += loss.item()
                steps += 1

            if steps and epoch % 10 == 0:
                logger.debug(
                    "Epoch %d/%d - training loss: %.5f",
                    epoch + 1,
                    self.epochs,
                    loss_accumulator / steps,
                )

        model.eval()
        predictions: list[float] = []
        history = data.clone()
        with torch.no_grad():
            for _ in range(horizon):
                input_seq = history[-seq_len:].unsqueeze(0)
                next_pred = model(input_seq).squeeze()
                history = torch.cat([history, next_pred.view(1, 1)], dim=0)
                denormalized = self._denormalize(next_pred.item(), mean, std)
                predictions.append(float(max(denormalized, 0.0)))

        return predictions

    @staticmethod
    def _naive_forecast(series: pd.Series, horizon: int) -> list[float]:
        recent = series.tail(24)
        baseline = recent.mean()
        return [float(baseline)] * horizon

    @staticmethod
    def _normalize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
        mean = float(np.mean(values))
        std = float(np.std(values) or 1.0)
        normalized = (values - mean) / std
        return normalized, mean, std

    @staticmethod
    def _denormalize(value: float, mean: float, std: float) -> float:
        return value * std + mean

    @staticmethod
    def _load_torch() -> Optional[_TorchArtifacts]:
        try:
            import torch
            from torch import nn, optim
            from torch.utils import data as torch_data  # noqa: F401

            torch.manual_seed(42)
            return _TorchArtifacts(torch=torch, nn=nn, optim=optim, utils=torch_data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("PyTorch not available for LSTM forecasting: %s", exc)
            return None


class _LSTMModel:
    def __init__(self, input_size: int, hidden_size: int) -> None:
        import torch
        from torch import nn

        class _Net(nn.Module):
            def __init__(self, input_size: int, hidden_size: int) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    batch_first=True,
                )
                self.linear = nn.Linear(hidden_size, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                output, _ = self.lstm(x)
                last_hidden = output[:, -1, :]
                return self.linear(last_hidden)

        self._torch = torch
        self.model = _Net(input_size, hidden_size)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def parameters(self):
        return self.model.parameters()

    def eval(self):
        self.model.eval()
