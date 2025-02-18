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
    # Running
    for cmd_str in [
        f"conda env create -f {os.path.join(templates_dir, 'DEEPLABCUT.yaml')}",
        f"conda env create -f {os.path.join(templates_dir, 'simba.yaml')}",
    ]:
        run_subproc_simple(cmd_str)


if __name__ == "__main__":
    main()
