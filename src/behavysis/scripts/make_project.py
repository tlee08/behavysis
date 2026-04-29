from pathlib import Path

from behavysis.constants import Folders
from behavysis.models.experiment_configs import get_default_configs
from behavysis.utils.diagnostics_utils import file_exists_msg
from behavysis.utils.template_utils import import_static_templates_script


def main() -> None:
    """Makes a script to run a behavysis analysis project."""
    root_dir = Path.cwd()
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
    if not to_overwrite and (Path.cwd() / "default_configs.json").exists():
        print(file_exists_msg("default_configs.json"))
    else:
        default_configs = get_default_configs()
        (Path.cwd() / "default_configs.json").write_text(
            default_configs.model_dump_json(indent=2)
        )
    # Adding the folders
    for folder in Folders:
        (Path.cwd() / folder.value).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
