"""
Utility functions.
"""

import asyncio
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

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


def silent_remove(fp: str) -> None:
    """
    Removes the given file or dir if it exists.
    Does nothing if not.
    Does not throw any errors,
    """
    try:
        if os.path.isfile(fp):
            os.remove(fp)
        elif os.path.isdir(fp):
            shutil.rmtree(fp)
    except (OSError, FileNotFoundError):
        pass


def get_name(fp: str) -> str:
    """
    Given the filepath, returns the name of the file.
    The name is:
    ```
    <path_to_file>/<name>.<ext>
    ```
    """
    return os.path.splitext(os.path.basename(fp))[0]


def check_files_exist(*args: str):
    """
    args is dst_fp_ls
    """
    for dst_fp in args:
        if os.path.exists(dst_fp):
            return True
    return False


def read_json(fp: str) -> dict:
    """
    Reads the json file at the given filepath.
    """
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(fp: str, data: dict) -> None:
    """
    Writes the given data to the json file at the given filepath.
    """
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def joblib_load(fp: str):
    """
    Load a joblib file.
    """
    return joblib.load(fp)


def joblib_dump(data, fp: str):
    """
    Dump a joblib file.
    """
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    joblib.dump(data, fp)


async def async_read(fp: str, executor: ThreadPoolExecutor, read_func: Callable) -> list:
    """Asynchronously read a single file."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, read_func, fp)


async def async_read_files(fp_ls, read_func: Callable) -> list:
    """Asynchronously read a list of files and return a list of numpy arrays."""
    with ThreadPoolExecutor() as executor:
        tasks = [async_read(fp, executor, read_func) for fp in fp_ls]
        return await asyncio.gather(*tasks)


def async_read_files_run(fp_ls, read_func: Callable) -> list:
    """Asynchronously read a list of files and return a list of numpy arrays."""
    return asyncio.run(async_read_files(fp_ls, read_func))
