from pathlib import Path

from behavysis.utils.template_utils import confirm, save_template


def main() -> None:
    if not confirm("Create DLC builder script in current directory?"):
        return
    if Path("dlc_builder.ipynb").exists() and not confirm("Overwrite existing file?"):
        return
    save_template("dlc_builder.ipynb", Path("dlc_builder.ipynb"))


if __name__ == "__main__":
    main()
