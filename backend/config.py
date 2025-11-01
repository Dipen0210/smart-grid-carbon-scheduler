from __future__ import annotations

import os
from pathlib import Path

# Base directories
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Forecasting defaults
DEFAULT_FORECAST_HORIZON_HOURS = 24

# External service defaults / environment
EIA_DEFAULT_API_KEY = "eAyqVC9eUsf9kOim4wJ4228vYJBSgoSqKTl4Ac1p"
EIA_API_KEY = os.getenv("EIA_API_KEY", EIA_DEFAULT_API_KEY)
EIA_DEFAULT_SERIES_ID = os.getenv("EIA_SERIES_ID", "EBA.NYIS-ALL.D.H")

OPEN_METEO_LATITUDE = float(os.getenv("OPEN_METEO_LATITUDE", "40.7128"))
OPEN_METEO_LONGITUDE = float(os.getenv("OPEN_METEO_LONGITUDE", "-74.0060"))

ELECTRICITY_MAPS_TOKEN = os.getenv("ELECTRICITY_MAPS_TOKEN")
ELECTRICITY_MAPS_ZONE = os.getenv("ELECTRICITY_MAPS_ZONE", "US-NY")


def ensure_processed_dir() -> Path:
    """Ensure the processed data directory exists and return it."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return PROCESSED_DATA_DIR
