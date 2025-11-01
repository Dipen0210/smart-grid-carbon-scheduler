from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

from backend import config
from backend.rl.train_ppo import train_and_schedule

logger = logging.getLogger(__name__)


def run_rl_scheduler(
    energy_kwh: float,
    max_power_kw: float,
    hours: int,
    carbon_intensity_path: Optional[Path] = None,
    episodes: int = 200,
) -> Tuple[Path, list[dict], dict]:
    """Train the PPO scheduler and persist the resulting schedule."""
    processed_dir = config.ensure_processed_dir()
    resolved_carbon_path = carbon_intensity_path

    if resolved_carbon_path is None:
        candidate = processed_dir / "carbon_latest.csv"
        if candidate.exists():
            resolved_carbon_path = candidate
        else:
            # Fallback to merged dataset if carbon export not present.
            merged = processed_dir / "raw_latest.csv"
            if merged.exists():
                resolved_carbon_path = merged

    if resolved_carbon_path:
        logger.info("Using carbon data at %s for RL scheduling.", resolved_carbon_path)
    else:
        logger.warning(
            "No carbon intensity dataset found. Scheduler will rely on synthetic data."
        )

    schedule_path, schedule_records, metadata = train_and_schedule(
        energy_kwh=energy_kwh,
        max_power_kw=max_power_kw,
        hours=hours,
        carbon_intensity_path=resolved_carbon_path,
        output_dir=processed_dir,
        episodes=episodes,
    )
    logger.info("RL scheduling complete. Saved to %s.", schedule_path)
    return schedule_path, schedule_records, metadata
