"""
Utility functions.
"""

from __future__ import annotations

import os
from subprocess import PIPE, Popen


class SubprocMixin:
    """__summary__"""

    @staticmethod
    def run_subproc_fstream(cmd: list[str], fp: str, **kwargs) -> None:
        """Run a subprocess and stream the output to a file."""
        # Making a file to store the output
        os.makedirs(os.path.split(fp)[0], exist_ok=True)
        with open(fp, "w", encoding="utf-8") as f:
            # Starting the subprocess
            with Popen(cmd, stdout=f, stderr=f, **kwargs) as p:
                # Wait for the subprocess to finish
                p.wait()
                # Error handling
                if p.returncode:
                    f.seek(0)
                    raise ValueError(f.read().decode())

    @staticmethod
    def run_subproc_str(cmd: list[str], **kwargs) -> str:
        """Run a subprocess and return the output as a string."""
        # Running the subprocess
        with Popen(cmd, stdout=PIPE, stderr=PIPE, **kwargs) as p:
            # Wait for the subprocess to finish
            out, err = p.communicate()
            # Error handling
            if p.returncode:
                raise ValueError(err.decode("utf-8"))
            return out.decode("utf-8")

    @staticmethod
    def run_subproc_console(cmd: list[str], **kwargs) -> None:
        """Run a subprocess and stream the output to a file."""
        # Starting the subprocess
        with Popen(cmd, **kwargs) as p:
            # Wait for the subprocess to finish
            p.wait()
            # Error handling
            if p.returncode:
                raise ValueError("ERROR: Subprocess failed to run.")
