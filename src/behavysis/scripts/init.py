"""Functions for uv package scripts.

Overall deeplabcut install command:
```
conda env create -f DEEPLABCUT.yaml
conda activate DEEPLABCUT
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install "git+https://github.com/DeepLabCut/DeepLabCut.git@pytorch_dlc#egg=deeplabcut[gui,modelzoo,wandb]"
```
"""

import os
import subprocess
from importlib.resources import as_file, files


def main() -> None:
    """Sets up the behavysis environment.

    - Installs DEEPLABCUT conda env
    """
    conda_exe = os.environ.get("CONDA_EXE", "conda")
    yaml_res = files("behavysis").joinpath("templates", "dlc", "DEEPLABCUT.yaml")
    # Running
    with as_file(yaml_res) as yaml_fp:
        subprocess.run([conda_exe, "env", "create", "-f", str(yaml_fp)], check=True)  # noqa: S603


if __name__ == "__main__":
    main()
