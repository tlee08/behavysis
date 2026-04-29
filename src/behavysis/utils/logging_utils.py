"""Clean logging system for behavysis following the cellcounter pattern."""

import logging
import logging.handlers
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from behavysis.constants import CACHE_DIR

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class LogLevel(Enum):
    """Log level enumeration."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class ProcessResult:
    """Structured result container for process function diagnostics.

    Replaces the previous StringIO-based logging approach with a cleaner,
    structured result object.
    """

    process_name: str
    success: bool = True
    error_message: str = ""
    logs: list[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None

    def add_log(self, level: LogLevel, message: str) -> None:
        """Add a log entry to the result."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logs.append(f"{timestamp} - {level.name} - {message}")

    def mark_complete(self, success: bool = True, error_message: str = "") -> None:
        """Mark the process as complete."""
        self.end_time = datetime.now()
        self.success = success
        self.error_message = error_message

    @property
    def duration(self) -> float:
        """Get process duration in seconds."""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "process_name": self.process_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "success": self.success,
            "error_message": self.error_message,
            "logs": "\n".join(self.logs) if self.logs else "",
            "duration": self.duration,
        }


def setup_logging(
    level: LogLevel = LogLevel.INFO,
    log_file: Path | str | None = None,
) -> None:
    """Configure logging once at application startup.

    After calling this, use: logger = logging.getLogger(__name__)

    Parameters
    ----------
    level : LogLevel
        Minimum log level for console output
    log_file : Path | str | None
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
    ch.setLevel(level.value)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # File handler (with rotation)
    log_file = Path(log_file) if log_file else CACHE_DIR / "debug.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10_000_000, backupCount=3
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(fh)
