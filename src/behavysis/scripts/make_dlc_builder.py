"""Make DLC builder."""

from pathlib import Path

from behavysis.utils.template_utils import confirm, save_template


def main() -> None:
    """Make DLC builder."""
    if not confirm("Create DLC builder script in current directory?"):
        return
    if Path("dlc_builder.py").exists() and not confirm("Overwrite existing file?"):
        return
    save_template("dlc_builder/dlc_builder.py", Path("dlc_builder.py"))


if __name__ == "__main__":
    main()
