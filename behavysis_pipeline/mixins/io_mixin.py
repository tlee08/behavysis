"""
Utility functions.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
from typing import Any

from jinja2 import Environment, PackageLoader

from behavysis_pipeline.mixins.diagnostics_mixin import DiagnosticsMixin


class IOMixin:
    """__summary__"""

    @staticmethod
    def clear_dir_junk(my_dir: str) -> None:
        """
        Removes all hidden junk files in given directory.
        Hidden files begin with ".".
        """
        for i in os.listdir(dir):
            path = os.path.join(my_dir, i)
            # If the file has a "." at the start, remove it
            if re.search(r"^\.", i):
                IOMixin.silent_rm(path)

    @staticmethod
    def silent_rm(fp: str) -> None:
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

    @staticmethod
    def get_name(fp: str) -> str:
        """
        Given the filepath, returns the name of the file.
        The name is:
        ```
        <path_to_file>/<name>.<ext>
        ```
        """
        return os.path.splitext(os.path.basename(fp))[0]

    @staticmethod
    def overwrite_check(out_fp_var: str = "out_fp", overwrite_var: str = "overwrite"):
        """
        Decorator to check if we should skip processing (i.e. not overwrite the file).
        Returns the function early if we should skip.
        """

        def decorator(func):
            """__summary__"""

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                """__summary__"""
                # Asserting that the variables are in the kwargs
                assert out_fp_var in kwargs, f"Missing {out_fp_var} in kwargs"
                assert overwrite_var in kwargs, f"Missing {overwrite_var} in kwargs"
                # Getting the out_fp and overwrite variables
                out_fp = kwargs[out_fp_var]
                overwrite = kwargs[overwrite_var]
                # If overwrite is False, checking if we should skip processing
                if not overwrite and os.path.exists(out_fp):
                    return DiagnosticsMixin.warning_msg(func)
                # Running the function and returning
                return func(*args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def render_template(
        tmpl_name: str,
        pkg_name: str,
        pkg_subdir: str,
        **kwargs: Any,
    ) -> str:
        """
        Renders the given template with the given arguments.
        """
        # Loading the Jinja2 environment
        env = Environment(loader=PackageLoader(pkg_name, pkg_subdir))
        # Getting the template
        template = env.get_template(tmpl_name)
        # Rendering the template
        return template.render(**kwargs)

    @staticmethod
    def save_template(
        tmpl_name: str,
        pkg_name: str,
        pkg_subdir: str,
        dst_fp: str,
        **kwargs: Any,
    ) -> None:
        """
        Renders the given template with the given arguments and saves it to the out_fp.
        """
        # Rendering the template
        rendered = IOMixin.render_template(tmpl_name, pkg_name, pkg_subdir, **kwargs)
        # Making the directory if it doesn't exist
        os.makedirs(os.path.dirname(dst_fp), exist_ok=True)
        # Saving the rendered template
        with open(dst_fp, "w") as f:
            f.write(rendered)
