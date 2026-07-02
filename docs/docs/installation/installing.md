# Installing

**Step 1: Install conda**

Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

```zsh
conda --version   # verify installation
```

**Step 2: Speed up conda (recommended)**

```zsh
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

**Step 3: Install the behavysis environment**

Download [`conda_env.yaml`](https://github.com/tlee08/behavysis/blob/main/conda_env.yaml) and run:

```zsh
conda env create -f path/to/conda_env.yaml
```

**Step 4: Install the DeepLabCut environment**

```zsh
conda activate behavysis
behavysis-init
```

This creates the `DEEPLABCUT` conda environment needed for keypoint tracking.

**Step 5: Verify**

```zsh
conda activate behavysis
python -c "import behavysis"
```

## Developer installation

For contributors — uses uv in addition to conda:

```zsh
conda env create -f conda_env.yaml
conda activate behavysis
uv pip install -e ".[dev]"
```
