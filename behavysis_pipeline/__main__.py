import os

from behavysis_core.constants import Folders
from behavysis_core.mixins.io_mixin import IOMixin


def make_project(root_dir: str = ".", overwrite: bool = False) -> None:
    """
    Makes a script to run a behavysis analysis project.

    Copies the `run.py` script and `default_configs.json` to `root_dir`.
    """
    # Making the root folder
    os.makedirs(root_dir, exist_ok=True)
    # Making each subfolder
    for f in Folders:
        os.makedirs(os.path.join(root_dir, f.value), exist_ok=True)
    # Copying the default_configs.json and run.py files to the project folder
    for i in ["default_configs.json", "run.py"]:
        # Getting the file path
        dst_fp = os.path.join(root_dir, i)
        # If not overwrite and file exists, then don't overwrite
        if not overwrite and os.path.exists(dst_fp):
            continue
        # Saving the template to the file
        IOMixin.save_template(
            i,
            "behavysis_pipeline",
            "script_templates",
            dst_fp,
        )


def make_behav_classifier(root_dir: str = ".", overwrite: bool = False) -> None:
    """
    Makes a script to build a BehavClassifier.

    Copies the `train_behav_classifier.py` script to `root_dir/behav_models`.
    """
    # Making the project root folder
    os.makedirs(os.path.join(root_dir, "behav_models"), exist_ok=True)
    # Copying the default_configs.json and run.py files to the project folder
    for i in ["train_behav_classifier.py"]:
        # Getting the file path
        dst_fp = os.path.join(root_dir, i)
        # If not overwrite and file exists, then don't overwrite
        if not overwrite and os.path.exists(dst_fp):
            continue
        # Saving the template to the file
        IOMixin.save_template(
            i,
            "behavysis_pipeline",
            "script_templates",
            dst_fp,
        )
