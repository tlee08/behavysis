"""Clean logging system for behavysis following the cellcounter pattern."""

import logging
import logging.handlers
from pathlib import Path

from behavysis.constants import CACHE_DIR

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> None:
    """Configure logging once at application startup.

    After calling this, use: logger = logging.getLogger(__name__)

    Parameters
    ----------
    level : int
        Minimum log level for console output
    log_file : Path | None
        Optional custom log file path. If None and project_name is provided,
        uses ~/.behavysis/{project_name}/debug.log

    Returns:
    -------
    None
    """
    root = logging.getLogger("behavysis")
    if root.handlers:  # if already configured
        return

    # Set logging levels and formatter
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # File handler (with rotation)
    log_file = log_file or CACHE_DIR / "debug.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10_000_000, backupCount=3
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(fh)
