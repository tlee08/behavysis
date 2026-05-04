from pathlib import Path

from behavysis.constants import Folders
from behavysis.models.experiment_configs import get_default_configs
from behavysis.utils.template_utils import confirm, save_template


def main() -> None:
    if not confirm("Create behavysis project in current directory?"):
        return

    overwrite = confirm("Overwrite existing files?")

    if overwrite or not Path("run_pipeline.py").exists():
        save_template("run_pipeline.py", Path("run_pipeline.py"))

    if overwrite or not Path("default_configs.json").exists():
        Path("default_configs.json").write_text(
            get_default_configs().model_dump_json(indent=2)
        )

    for folder in Folders:
        Path(folder.value).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
