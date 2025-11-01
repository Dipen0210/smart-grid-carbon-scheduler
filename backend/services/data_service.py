from __future__ import annotations

import logging
from pathlib import Path
import shutil

from backend import config
from backend.data.fetch_all import fetch_all_data
from backend.data.fetch_all import DEFAULT_LOOKBACK_HOURS

logger = logging.getLogger(__name__)

WEEK_LOOKBACK_HOURS = 7 * 24
BASELINE_LABEL = "year"
LATEST_LABEL = "latest"


def _ensure_alias(src: Path, alias_name: str) -> None:
    alias_path = src.parent / alias_name
    if src.exists():
        shutil.copyfile(src, alias_path)


def fetch_fresh_data() -> tuple[Path, dict[str, object]]:
    """Fetch demand, weather, and carbon datasets and persist them to processed storage."""
    output_dir = config.ensure_processed_dir()
    logger.info("Fetching fresh datasets into %s.", output_dir)

    baseline_path = output_dir / f"raw_{BASELINE_LABEL}.csv"
    baseline_exists = baseline_path.exists()

    if baseline_exists:
        lookback = WEEK_LOOKBACK_HOURS
        label = LATEST_LABEL
    else:
        lookback = DEFAULT_LOOKBACK_HOURS
        label = BASELINE_LABEL

    path, metadata = fetch_all_data(
        output_dir=output_dir,
        lookback_hours=lookback,
        label=label,
    )

    if label == BASELINE_LABEL:
        _ensure_alias(path, "raw_latest.csv")
        _ensure_alias(output_dir / f"demand_{BASELINE_LABEL}.csv", "demand_latest.csv")
        _ensure_alias(output_dir / f"carbon_{BASELINE_LABEL}.csv", "carbon_latest.csv")
        _ensure_alias(output_dir / f"weather_{BASELINE_LABEL}.csv", "weather_latest.csv")
        metadata["aliases"] = {
            "raw_latest": str(output_dir / "raw_latest.csv"),
            "demand_latest": str(output_dir / "demand_latest.csv"),
            "carbon_latest": str(output_dir / "carbon_latest.csv"),
            "weather_latest": str(output_dir / "weather_latest.csv"),
        }

    metadata["baseline_exists"] = baseline_exists
    metadata["label"] = label
    metadata["lookback_hours"] = lookback
    logger.info(
        "Data fetch complete for label '%s' (lookback %d hours). Saved to %s.",
        label,
        lookback,
        path,
    )
    return path, metadata
