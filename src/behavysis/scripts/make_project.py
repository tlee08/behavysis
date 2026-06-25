from pathlib import Path

from behavysis.constants import DEFAULT_CONFIG_FP, Folders
from behavysis.models import get_default_config
from behavysis.utils.template_utils import confirm, save_template


def main() -> None:
    if not confirm("Create behavysis project in current directory?"):
        return

    overwrite = confirm("Overwrite existing files?")

    if overwrite or not Path("run_pipeline.py").exists():
        save_template("run_pipeline.py", Path("run_pipeline.py"))

    if overwrite or not Path(DEFAULT_CONFIG_FP).exists():
        Path(DEFAULT_CONFIG_FP).write_text(
            get_default_config().model_dump_json(indent=2)
        )

    for folder in Folders:
        Path(folder.value).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
