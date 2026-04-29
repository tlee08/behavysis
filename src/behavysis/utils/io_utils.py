"""Utility functions."""

import asyncio
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def silent_remove(fp: Path) -> None:
    """Removes the given file or dir if it exists.

    Does nothing if not.
    Does not throw any errors,
    """
    try:
        if fp.is_file():
            fp.unlink()
        elif fp.is_dir():
            shutil.rmtree(fp)
    except (OSError, FileNotFoundError):
        pass


def get_name(fp: Path | str) -> str:
    """Given the filepath, returns the name of the file.

    The name is:
    ```
    <path_to_file>/<name>.<ext>
    ```
    """
    return Path(fp).stem


def check_files_exist(*args: Path) -> bool:
    """Args is dst_fp_ls."""
    return any(dst_fp.exists() for dst_fp in args)


async def async_read(
    fp: Path, executor: ThreadPoolExecutor, read_func: Callable
) -> list:
    """Asynchronously read a single file."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, read_func, fp)


async def async_read_files(fp_ls: list[Path], read_func: Callable) -> list:
    """Asynchronously read a list of files and return a list of numpy arrays."""
    with ThreadPoolExecutor() as executor:
        tasks = [async_read(fp, executor, read_func) for fp in fp_ls]
        return await asyncio.gather(*tasks)


def async_read_files_run(fp_ls: list[Path], read_func: Callable) -> list:
    """Asynchronously read a list of files and return a list of numpy arrays."""
    return asyncio.run(async_read_files(fp_ls, read_func))
