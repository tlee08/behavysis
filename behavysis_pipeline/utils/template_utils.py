"""
Utility functions.
"""

from __future__ import annotations

import os
from typing import Any

from jinja2 import Environment, PackageLoader

from behavysis_pipeline.utils.diagnostics_utils import file_exists_msg
from behavysis_pipeline.utils.logging_utils import init_logger

logger = init_logger(__name__)


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


# TODO: move back to scripts. A bit redundant here.
def import_template(src_fp: str, dst_fp: str, overwrite: bool):
    """
    Imports the template file to the project folder.
    """
    if not overwrite and os.path.exists(dst_fp):
        print(file_exists_msg(dst_fp))
        return
    # Saving the template to the file
    save_template(
        src_fp,
        "behavysis_pipeline",
        "templates",
        dst_fp,
    )
