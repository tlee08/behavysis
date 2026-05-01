import logging
from datetime import UTC, datetime

from pydantic import BaseModel, Field


class ProcessResult(BaseModel):
    """Structured result container for process function diagnostics.

    Replaces the previous StringIO-based logging approach with a cleaner,
    structured result object.
    """

    process_name: str
    success: bool = True
    error_message: str = ""
    logs: list[str] = []
    start_time: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    end_time: datetime | None = None

    def add_log(self, level: int, message: str) -> None:
        """Add a log entry to the result."""
        timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S")
        self.logs.append(f"{timestamp} - {logging.getLevelName(level)} - {message}")

    def mark_complete(self, *, success: bool = True, error_message: str = "") -> None:
        """Mark the process as complete."""
        self.end_time = datetime.now(tz=UTC)
        self.success = success
        self.error_message = error_message

    @property
    def duration(self) -> float:
        """Get process duration in seconds."""
        end = self.end_time or datetime.now(tz=UTC)
        return (end - self.start_time).total_seconds()


class ProcessResultCollection(BaseModel):
    """Structured collection of process results for an experiment."""

    experiment: str
    results: dict[str, ProcessResult] = {}
