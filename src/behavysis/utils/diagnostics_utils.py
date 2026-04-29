import logging
from datetime import UTC, datetime

import numpy as np
from pydantic import BaseModel, Field

DIAGNOSTICS_SUCCESS_MESSAGES = (
    "Success! Success! Success!!",
    "Done and DONE!!",
    "Yay! Completed!",
    "This process was completed. Good on you :)",
    "Thumbs up!",
    "Woohoo!!!",
    "Phenomenal!",
    ":) :) :) :) :)",
    "Go you!",
    "You are doing awesome!",
    "You got this!",
    "You're doing great!",
    "Sending good vibes.",
    "I believe in you!",
    "You're a champion!",
    "No task too tall :) :)",
    "A job done well, and a well done job!",
    "Top job!",
)


def success_msg() -> str:
    """Return a random positive message!"""
    return f"SUCCESS: {np.random.choice(DIAGNOSTICS_SUCCESS_MESSAGES)}\n"


def file_exists_msg(fp: str | None = None) -> str:
    """Return a warning message."""
    fp_str = f", {fp}, " if fp else " "
    return f"Output file{fp_str}already exists - not overwriting file.To overwrite, specify `overwrite=True`."


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

    def mark_complete(self, success: bool = True, error_message: str = "") -> None:
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
