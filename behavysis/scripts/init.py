import os
import subprocess

from behavysis.utils.misc_utils import get_module_dir


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
    # Determine whether OS is Windows or Unix
    shell = True if os.name == "nt" else False
    # Running install DEEPLABCUT env
    for cmd_str in [
        f"cd {templates_dir} && conda env create -f DEEPLABCUT.yaml",
        f"cd {templates_dir} && conda env create -f simba_env.yaml",
    ]:
        try:
            subprocess.run(
                f"cd {templates_dir} && conda env create -f DEEPLABCUT.yaml",
                shell=shell,
                check=True,
                executable="/bin/bash" if not shell else None,
            )
            print("DEEPLABCUT environment installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
