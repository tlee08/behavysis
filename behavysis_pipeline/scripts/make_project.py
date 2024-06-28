import os

from behavysis_core.constants import Folders
from behavysis_core.mixins.io_mixin import IOMixin


def main(root_dir: str = ".", overwrite: bool = False) -> None:
    """
    Makes a script to run a behavysis analysis project.

    Copies the `run_project.py` script and `default_configs.json` to `root_dir`.
    """
    # Making the root folder
    os.makedirs(root_dir, exist_ok=True)
    # Making each subfolder
    for f in Folders:
        os.makedirs(os.path.join(root_dir, f.value), exist_ok=True)
    # Copying the files to the project folder
    for i in ["default_configs.json", "run_project.py"]:
        # Getting the file path
        dst_fp = os.path.join(root_dir, i)
        # If not overwrite and file exists, then don't overwrite
        if not overwrite and os.path.exists(dst_fp):
            continue
        # Saving the template to the file
        IOMixin.save_template(
            i,
            "behavysis_pipeline",
            "templates",
            dst_fp,
        )


if __name__ == "__main__":
    main()
