"""Utility functions."""

import asyncio
import json
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib

# def clear_dir_junk(my_dir: str) -> None:
#     """
#     Removes all hidden junk files in given directory.
#     Hidden files begin with ".".
#     """
#     for i in os.listdir(my_dir):
#         path = os.path.join(my_dir, i)
#         # If the file has a "." at the start, remove it
#         if re.search(r"^\.", i):
#             silent_remove(path)


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


def read_json(fp: Path) -> dict:
    """Reads the json file at the given filepath."""
    return json.loads(fp.read_text())


def write_json(fp: Path, data: dict) -> None:
    """Writes the given data to the json file at the given filepath."""
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(data, indent=4))


def joblib_load(fp: Path) -> object:
    """Load a joblib file."""
    return joblib.load(fp)


def joblib_dump(data: object, fp: Path) -> None:
    """Dump a joblib file."""
    fp.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, fp)


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
