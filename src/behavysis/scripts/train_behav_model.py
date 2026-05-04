from pathlib import Path

from behavysis.utils.template_utils import confirm, save_template


def main() -> None:
    if not confirm("Create training script in current directory?"):
        return
    if Path("train_behav_model.py").exists() and not confirm(
        "Overwrite existing file?"
    ):
        return
    save_template("train_behav_model.py", Path("train_behav_model.py"))


if __name__ == "__main__":
    main()
