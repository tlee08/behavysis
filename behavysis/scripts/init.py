import os
import subprocess

from behavysis.utils.misc_utils import get_module_dir


def main() -> None:
    """
    Sets up the behavysis environment.
    - Installs DEEPLABCUT conda env
    - Installs SimBA conda env
    """
    behavysis_dir = get_module_dir("behavysis")
    templates_dir = os.path.join(behavysis_dir, "templates")
    # Installing DEEPLABCUT env
    subprocess.call(["cd", templates_dir, "&&" "conda", "env", "create", "-f", "DEEPLABCUT.yaml"])
    # Installing simba env
    subprocess.call(["cd", templates_dir, "&&", "conda", "env", "create", "-f", "simba_env.yaml"])


if __name__ == "__main__":
    main()
