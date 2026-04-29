import subprocess
from importlib.resources import files
from pathlib import Path


def main() -> None:
    """Sets up the behavysis environment.
    - Installs DEEPLABCUT conda env
    - Installs SimBA conda env.
    """
    templates_dir = Path(str(files("behavysis"))) / "templates"
    # Running
    for cmd_str in [
        (
            f"conda env create -f {templates_dir / 'DEEPLABCUT.yaml'} \n "
            # "conda activate DEEPLABCUT \n "
            # "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 \n "
            # 'pip install "git+https://github.com/DeepLabCut/DeepLabCut.git@pytorch_dlc#egg=deeplabcut[gui,modelzoo,wandb]"'
        ),
        f"conda env create -f {templates_dir / 'simba.yaml'}",
    ]:
        subprocess.run(cmd_str, shell=True)


if __name__ == "__main__":
    main()
