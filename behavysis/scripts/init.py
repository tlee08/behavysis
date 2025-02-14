import os

from behavysis.utils.misc_utils import get_module_dir
from behavysis.utils.subproc_utils import run_subproc_simple


def main() -> None:
    """
    Sets up the behavysis environment.
    - Installs DEEPLABCUT conda env
    - Installs SimBA conda env
    """
    # Getting the templates directory
    behavysis_dir = get_module_dir("behavysis")
    templates_dir = os.path.join(behavysis_dir, "templates")
    templates_dir = os.path.abspath(templates_dir)
    # Checking if the templates directory exists
    assert os.path.isdir(templates_dir), f"Templates directory not found: {templates_dir}"
    # Running
    for cmd_str in [
        f"cd {templates_dir} && conda env create -f DEEPLABCUT.yaml",
        f"cd {templates_dir} && conda env create -f simba.yaml",
    ]:
        run_subproc_simple(cmd_str)


if __name__ == "__main__":
    main()
