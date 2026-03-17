"""
Utility functions.
"""

import os
import subprocess

ENCODING = "utf-8"


def run_subproc_fstream(cmd: list[str], fp: str, **kwargs) -> None:
    """Run a subprocess and stream the output to a file."""
    # Making a file to store the output
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, "w", encoding=ENCODING) as f:
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
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs) as p:
        # Wait for the subprocess to finish
        out, err = p.communicate()
        # Error handling (returncode is not 0)
        if p.returncode:
            raise ValueError(err)
        return out


def run_subproc_logger(cmd: list[str], logger, **kwargs) -> None:
    """Run a subprocess and stream the output to a logger."""
    # Starting the subprocess
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs) as p:
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
            raise ValueError("ERROR: Subprocess failed to run.")


def run_subproc_simple(cmd_str) -> None:
    try:
        subprocess.run(cmd_str, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
