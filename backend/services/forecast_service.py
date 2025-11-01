from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from backend import config
from backend.forecasting.lstm_forecast import ForecastResult, run_forecast

logger = logging.getLogger(__name__)


def run_demand_forecast(
    data_path: Optional[Path] = None, horizon_hours: int = config.DEFAULT_FORECAST_HORIZON_HOURS
) -> ForecastResult:
    """Run the demand forecast workflow and return the generated forecast artefact."""
    processed_dir = config.ensure_processed_dir()
    resolved_data_path = data_path

    if resolved_data_path is None:
        candidate = processed_dir / "raw_latest.csv"
        if candidate.exists():
            resolved_data_path = candidate
            logger.info("Using latest processed dataset at %s for forecasting.", resolved_data_path)
        else:
            logger.warning(
                "No existing energy dataset found. Forecast will rely on synthetic data."
            )

    result = run_forecast(
        data_path=resolved_data_path,
        output_dir=processed_dir,
        horizon_hours=horizon_hours,
    )
    logger.info(
        "Forecast complete using history window %s -> %s. Saved forecast to %s.",
        result.history_start.isoformat(),
        result.history_end.isoformat(),
        result.path,
    )
    return result
