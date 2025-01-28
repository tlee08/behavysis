"""
Utility functions.
"""

import os
from subprocess import PIPE, Popen

ENCODING = "utf-8"

# TODO: Add logging (e.g. run_subproc_logger)


def run_subproc_fstream(cmd: list[str], fp: str, **kwargs) -> None:
    """Run a subprocess and stream the output to a file."""
    # Making a file to store the output
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, "w", encoding=ENCODING) as f:
        # Starting the subprocess
        with Popen(cmd, stdout=f, stderr=f, **kwargs) as p:
            # Wait for the subprocess to finish
            p.wait()
            # Error handling (returncode is not 0)
            if p.returncode:
                f.seek(0)
                raise ValueError(f.read())


def run_subproc_str(cmd: list[str], **kwargs) -> str:
    """Run a subprocess and return the output as a string."""
    # Running the subprocess
    with Popen(cmd, stdout=PIPE, stderr=PIPE, **kwargs) as p:
        # Wait for the subprocess to finish
        out, err = p.communicate()
        # Error handling (returncode is not 0)
        if p.returncode:
            raise ValueError(err)
        return out


def run_subproc_console(cmd: list[str], **kwargs) -> None:
    """Run a subprocess and stream the output to a file."""
    # Starting the subprocess
    with Popen(cmd, **kwargs) as p:
        # Wait for the subprocess to finish
        p.wait()
        # Error handling (returncode is not 0)
        if p.returncode:
            raise ValueError("ERROR: Subprocess failed to run.")
