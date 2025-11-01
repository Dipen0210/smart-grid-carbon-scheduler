from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def latest_file(directory: Path, pattern: str = "*.csv") -> Optional[Path]:
    if not directory.exists():
        logger.debug("Directory %s does not exist while searching for files.", directory)
        return None
    candidates = sorted(directory.glob(pattern))
    if not candidates:
        logger.debug(
            "No files found in %s matching pattern %s.", directory, pattern
        )
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    logger.debug("Latest file resolved to %s.", latest)
    return latest
