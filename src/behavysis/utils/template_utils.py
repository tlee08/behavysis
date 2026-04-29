"""Utility functions."""

from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader

from behavysis.utils.diagnostics_utils import file_exists_msg
from behavysis.utils.io_utils import check_files_exist


def render_template(
    tmpl_name: str, pkg_name: str, pkg_subdir: str, **kwargs: Any
) -> str:
    """Renders the given template with the given arguments."""
    # Loading the Jinja2 environment
    env = Environment(loader=PackageLoader(pkg_name, pkg_subdir))
    # Getting the template
    template = env.get_template(tmpl_name)
    # Rendering the template
    return template.render(**kwargs)


def save_template(
    tmpl_name: str, pkg_name: str, pkg_subdir: str, dst_fp: Path, **kwargs: Any
) -> None:
    """Renders the given template with the given arguments and saves it to the dst_fp."""
    # Rendering the template
    rendered = render_template(tmpl_name, pkg_name, pkg_subdir, **kwargs)
    # Making the directory if it doesn't exist
    dst_fp.parent.mkdir(parents=True, exist_ok=True)
    # Saving the rendered template
    with open(dst_fp, "w") as f:
        f.write(rendered)


def import_static_templates_script(
    description: str,
    templates_ls: list[str],
    pkg_name: str,
    pkg_subdir: str,
    root_dir: Path,
    to_overwrite: bool = False,
    dialogue: bool = True,
) -> tuple[bool, bool]:
    """A function to import static templates to a folder.
    Useful for calling scripts from the command line.
    """
    to_continue = False
    if dialogue:
        # Dialogue to check if the user wants to make the files
        to_continue = (
            input(
                f"Running {description} in current directory. Continue? [y/N]: "
            ).lower()
            + " "
        )[0] == "y"
        if not to_continue:
            print("Exiting.")
            return to_continue, to_overwrite
        # Dialogue to check if the user wants to overwrite the files
        to_overwrite = (input("Overwrite existing files? [y/N]: ").lower() + " ")[
            0
        ] == "y"
    # Making the root folder
    root_dir.mkdir(parents=True, exist_ok=True)
    # Copying the Python files to the project folder
    for template_fp in templates_ls:
        dst_fp = root_dir / template_fp
        if not to_overwrite and check_files_exist(dst_fp):
            # Check if we should skip importing (i.e. overwrite is False and file exists)
            print(file_exists_msg(dst_fp))
            continue
        # Writing the template to the destination file
        save_template(template_fp, pkg_name, pkg_subdir, dst_fp)
    # Returning the overwrite and dialogue values
    return to_continue, to_overwrite
