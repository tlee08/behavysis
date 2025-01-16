"""
Utility functions.
"""

from __future__ import annotations

import os
import re
import shutil
from typing import Any

from jinja2 import Environment, PackageLoader

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
    logger.debug("None of the filepaths in `pfm_fp_ls` exist.")
    logger.debug("Returning False.")
    return False


def render_template(tmpl_name: str, pkg_name: str, pkg_subdir: str, **kwargs: Any) -> str:
    """
    Renders the given template with the given arguments.
    """
    # Loading the Jinja2 environment
    env = Environment(loader=PackageLoader(pkg_name, pkg_subdir))
    # Getting the template
    template = env.get_template(tmpl_name)
    # Rendering the template
    return template.render(**kwargs)


def save_template(tmpl_name: str, pkg_name: str, pkg_subdir: str, dst_fp: str, **kwargs: Any) -> None:
    """
    Renders the given template with the given arguments and saves it to the out_fp.
    """
    # Rendering the template
    rendered = render_template(tmpl_name, pkg_name, pkg_subdir, **kwargs)
    # Making the directory if it doesn't exist
    os.makedirs(os.path.dirname(dst_fp), exist_ok=True)
    # Saving the rendered template
    with open(dst_fp, "w") as f:
        f.write(rendered)
