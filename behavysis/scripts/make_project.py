import os

from behavysis.constants import Folders
from behavysis.pydantic_models.experiment_configs import get_default_configs
from behavysis.utils.diagnostics_utils import file_exists_msg
from behavysis.utils.template_utils import import_static_templates_script


def main() -> None:
    """
    Makes a script to run a behavysis analysis project.
    """
    root_dir = "."
    # Adding the script to run the pipeline
    to_continue, to_overwrite = import_static_templates_script(
        description="Make Behavysis Pipeline Project",
        templates_ls=["run_pipeline.py"],
        pkg_name="behavysis",
        pkg_subdir="templates",
        root_dir=root_dir,
        to_overwrite=False,
        dialogue=True,
    )
    if not to_continue:
        return
    # Adding the default configs
    if not to_overwrite and os.path.exists("default_configs.json"):
        print(file_exists_msg("default_configs.json"))
    else:
        default_configs = get_default_configs()
        default_configs.write_json("default_configs.json")
    # Adding the folders
    for folder in Folders:
        os.makedirs(os.path.join(root_dir, folder.value), exist_ok=True)


if __name__ == "__main__":
    main()
