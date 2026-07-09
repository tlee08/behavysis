"""Utility functions."""

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


async def async_read(
    fp: Path,
    executor: ThreadPoolExecutor,
    read_func: Callable,
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
