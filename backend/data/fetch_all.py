from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests

from backend import config
from backend.utils.files import ensure_directory

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_HOURS = int(os.getenv("FETCH_LOOKBACK_HOURS", str(24 * 365)))
REQUEST_TIMEOUT = 30


def fetch_all_data(
    output_dir: Path | str = config.PROCESSED_DATA_DIR,
    lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
    *,
    label: str = "latest",
) -> tuple[Path, dict[str, object]]:
    """Fetch demand, weather, and carbon intensity data and persist merged CSV."""
    output_dir = ensure_directory(Path(output_dir))

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(hours=lookback_hours)

    logger.info(
        "Fetching datasets for window %s -> %s (%s hours).",
        start.isoformat(),
        now.isoformat(),
        lookback_hours,
    )

    demand_df = _fetch_eia_demand(start, now)
    weather_df = _fetch_open_meteo_weather(start, now)
    carbon_df = _fetch_electricity_maps_carbon(start, now)

    merged = (
        demand_df.merge(weather_df, on="timestamp", how="outer")
        .merge(carbon_df, on="timestamp", how="outer")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    suffix = f"_{label}" if label else ""
    merged_path = output_dir / f"raw{suffix}.csv"
    merged.to_csv(merged_path, index=False)
    logger.info("Merged dataset persisted to %s.", merged_path)

    # Persist companion datasets for downstream consumers.
    carbon_path = output_dir / f"carbon{suffix}.csv"
    carbon_df.to_csv(carbon_path, index=False)
    logger.debug("Carbon intensity dataset persisted to %s.", carbon_path)

    demand_path = output_dir / f"demand{suffix}.csv"
    demand_df.to_csv(demand_path, index=False)
    logger.debug("Demand history dataset persisted to %s.", demand_path)

    weather_path = output_dir / f"weather{suffix}.csv"
    weather_df.to_csv(weather_path, index=False)
    logger.debug("Weather dataset persisted to %s.", weather_path)

    metadata: dict[str, object] = {
        "label": label,
        "lookback_hours": lookback_hours,
        "window_start": start.isoformat(),
        "window_end": now.isoformat(),
        "merged_path": str(merged_path),
        "demand_path": str(demand_path),
        "carbon_path": str(carbon_path),
        "weather_path": str(weather_path),
    }
    legacy_demand = output_dir / f"demand_history{suffix}.csv"
    if legacy_demand.exists():
        legacy_demand.unlink(missing_ok=True)
    return merged_path, metadata


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------


def _fetch_eia_demand(start: datetime, end: datetime) -> pd.DataFrame:
    api_key = config.EIA_API_KEY
    series_id = config.EIA_DEFAULT_SERIES_ID

    if not api_key:
        logger.warning(
            "EIA_API_KEY not configured; generating synthetic demand data instead."
        )
        return _generate_synthetic_series(
            start, end, base=41000, amplitude=6000, column="demand_mw"
        )

    params = {
        "api_key": api_key,
        "series_id": series_id,
        "start": start.strftime("%Y%m%dT%HZ"),
        "end": end.strftime("%Y%m%dT%HZ"),
    }
    url = "https://api.eia.gov/series/"

    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        series = payload["series"][0]["data"]
        records = [
            {
                "timestamp": _parse_eia_timestamp(item[0]),
                "demand_mw": float(item[1]),
            }
            for item in series
        ]
        df = pd.DataFrame(records)
        df = df[df["timestamp"].between(start, end)]
        logger.info("Fetched %d hourly demand records from EIA.", len(df))
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to fetch EIA demand data, falling back to synthetic. %s", exc)
        return _generate_synthetic_series(
            start, end, base=41000, amplitude=6000, column="demand_mw"
        )


def _fetch_open_meteo_weather(start: datetime, end: datetime) -> pd.DataFrame:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": config.OPEN_METEO_LATITUDE,
        "longitude": config.OPEN_METEO_LONGITUDE,
        "hourly": "temperature_2m,relativehumidity_2m,wind_speed_10m",
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "timezone": "UTC",
    }

    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        hours = payload["hourly"]["time"]
        temp = payload["hourly"]["temperature_2m"]
        humidity = payload["hourly"]["relativehumidity_2m"]
        wind = payload["hourly"]["wind_speed_10m"]

        data = [
            {
                "timestamp": datetime.fromisoformat(ts).replace(tzinfo=timezone.utc),
                "temperature_c": float(t),
                "relative_humidity_pct": float(h),
                "wind_speed_mps": float(w),
            }
            for ts, t, h, w in zip(hours, temp, humidity, wind)
        ]
        df = pd.DataFrame(data)
        df = df[df["timestamp"].between(start, end)]
        logger.info("Fetched %d hourly weather records from Open-Meteo.", len(df))
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Failed to fetch Open-Meteo weather data, using synthetic fallback. %s", exc
        )
        return _generate_weather_series(start, end)


def _fetch_electricity_maps_carbon(start: datetime, end: datetime) -> pd.DataFrame:
    token = config.ELECTRICITY_MAPS_TOKEN
    zone = config.ELECTRICITY_MAPS_ZONE

    if not token:
        logger.warning(
            "ELECTRICITY_MAPS_TOKEN missing; generating synthetic carbon intensity."
        )
        return _generate_carbon_series(start, end)

    headers = {"auth-token": token}
    params = {
        "zone": zone,
        "start": start.isoformat().replace("+00:00", "Z"),
        "end": end.isoformat().replace("+00:00", "Z"),
    }
    url = "https://api.electricitymap.org/v3/carbon-intensity/history"

    try:
        response = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        history = payload.get("history") or payload.get("data") or []
        records = [
            {
                "timestamp": datetime.fromisoformat(entry["datetime"].replace("Z", "+00:00")),
                "carbon_intensity_gco2_per_kwh": float(entry["carbonIntensity"]),
            }
            for entry in history
        ]
        df = pd.DataFrame(records)
        df = df[df["timestamp"].between(start, end)]
        logger.info(
            "Fetched %d hourly carbon intensity records from Electricity Maps.", len(df)
        )
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Failed to fetch Electricity Maps data, using synthetic fallback. %s", exc
        )
        return _generate_carbon_series(start, end)


# ---------------------------------------------------------------------------
# Synthetic fallbacks
# ---------------------------------------------------------------------------


def _generate_synthetic_series(
    start: datetime, end: datetime, base: float, amplitude: float, column: str
) -> pd.DataFrame:
    rng = _hourly_range(start, end)
    phase = np.linspace(0, math.pi * 2, len(rng))
    noise = np.random.normal(scale=amplitude * 0.05, size=len(rng))
    values = base + amplitude * np.sin(phase) + noise
    df = pd.DataFrame({"timestamp": rng, column: values})
    return df


def _generate_weather_series(start: datetime, end: datetime) -> pd.DataFrame:
    rng = _hourly_range(start, end)
    phase = np.linspace(0, math.pi * 2, len(rng))
    temperature = 25 + 10 * np.sin(phase) + np.random.normal(scale=1.5, size=len(rng))
    humidity = 50 + 20 * np.cos(phase) + np.random.normal(scale=3, size=len(rng))
    wind = 5 + 2 * np.sin(phase / 2) + np.random.normal(scale=0.5, size=len(rng))
    return pd.DataFrame(
        {
            "timestamp": rng,
            "temperature_c": temperature,
            "relative_humidity_pct": humidity.clip(0, 100),
            "wind_speed_mps": wind.clip(min=0),
        }
    )


def _generate_carbon_series(start: datetime, end: datetime) -> pd.DataFrame:
    rng = _hourly_range(start, end)
    phase = np.linspace(0, math.pi * 4, len(rng))
    values = 400 + 100 * np.sin(phase) + np.random.normal(scale=20, size=len(rng))
    values = np.clip(values, 150, 700)
    return pd.DataFrame(
        {
            "timestamp": rng,
            "carbon_intensity_gco2_per_kwh": values,
        }
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _parse_eia_timestamp(value: str) -> datetime:
    # EIA timestamps look like "2024-01-01T00:00:00Z" or "20240101T00Z"
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return datetime.strptime(value, "%Y%m%dT%HZ").replace(tzinfo=timezone.utc)


def _hourly_range(start: datetime, end: datetime) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="H", tz=timezone.utc)
