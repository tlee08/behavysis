"""Make Behavysis model builder."""

from pathlib import Path

from behavysis.utils.template_utils import confirm, save_template


def main() -> None:
    """Make Behavysis model builder."""
    if not confirm("Create training script in current directory?"):
        return
    if Path("train_behaviour_classifier.py").exists() and not confirm(
        "Overwrite existing file?",
    ):
        return
    save_template(
        "train_behaviour_classifier.py", Path("train_behaviour_classifier.py")
    )


if __name__ == "__main__":
    main()
