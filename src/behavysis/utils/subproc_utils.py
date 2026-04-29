"""Utility functions."""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def run_subproc_fstream(cmd: list[str], fp: Path, **kwargs) -> None:
    """Run a subprocess and stream the output to a file."""
    # Making a file to store the output
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("w") as f:
        # Starting the subprocess
        with subprocess.Popen(cmd, stdout=f, stderr=f, **kwargs) as p:
            # Wait for the subprocess to finish
            p.wait()
            # Error handling (returncode is not 0)
            if p.returncode:
                f.seek(0)
                raise ValueError(f.read())


def run_subproc_str(cmd: list[str], **kwargs) -> str:
    """Run a subprocess and return the output as a string."""
    # Running the subprocess
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs
    ) as p:
        # Wait for the subprocess to finish
        out, err = p.communicate()
        # Error handling (returncode is not 0)
        if p.returncode:
            raise ValueError(err)
        return out


def run_subproc_logger(cmd: list[str], **kwargs) -> None:
    """Run a subprocess and stream the output to a logger."""
    # Starting the subprocess
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs
    ) as p:
        # Wait for the subprocess to finish
        while True:
            out = p.stdout.readline() if p.stdout else b""
            err = p.stderr.readline() if p.stderr else b""
            if out == b"" and p.poll() is not None:
                break
            if out:
                logger.info(out.strip())
            if err:
                logger.error(err.strip())
        # Error handling (returncode is not 0)
        if p.returncode:
            raise ValueError(err)


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


def run_subproc_simple(cmd_str) -> None:
    try:
        subprocess.run(cmd_str, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
