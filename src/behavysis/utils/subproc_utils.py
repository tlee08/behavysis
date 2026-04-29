"""Utility functions."""

import logging
import subprocess

logger = logging.getLogger(__name__)


def run_subproc_console(cmd: list[str], **kwargs) -> None:
    """Run a subprocess and stream the output to a file."""
    # Starting the subprocess
    with subprocess.Popen(cmd, **kwargs) as p:
        # Wait for the subprocess to finish
        p.wait()
        # Error handling (returncode is not 0)
        if p.returncode:
            msg = "Subprocess failed to run."
            raise ValueError(msg)
