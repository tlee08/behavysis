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
    # Running
    for cmd_str in [
        (
            f"conda env create -f {os.path.join(templates_dir, 'DEEPLABCUT.yaml')} \n "
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 \n "
            'pip install "git+https://github.com/DeepLabCut/DeepLabCut.git@pytorch_dlc#egg=deeplabcut[gui,modelzoo,wandb]"'
        ),
        f"conda env create -f {os.path.join(templates_dir, 'simba.yaml')}",
    ]:
        subprocess.run(cmd_str, shell=True, check=True)


if __name__ == "__main__":
    main()
