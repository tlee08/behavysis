"""
Utility functions.
"""

import os
import re
import shutil

from behavysis_pipeline.utils.logging_utils import init_logger

logger = init_logger(__name__)


def clear_dir_junk(my_dir: str) -> None:
    """
    Removes all hidden junk files in given directory.
    Hidden files begin with ".".
    """
    for i in os.listdir(dir):
        path = os.path.join(my_dir, i)
        # If the file has a "." at the start, remove it
        if re.search(r"^\.", i):
            silent_remove(path)


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


def check_files_exist(*args: tuple[str, ...]):
    """
    args is dst_fp_ls
    """
    logger.debug(f"Checking if the following files exist already: {args}")
    for dst_fp in args:
        if os.path.exists(dst_fp):
            logger.debug(f"{dst_fp} already exists.")
            logger.debug("Returning True.")
            return True
    logger.debug("None of the filepaths exist.")
    logger.debug("Returning False.")
    return False
