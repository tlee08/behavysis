# Conda

**Step 1:**
Install conda by visiting the [Miniconda downloads page](https://docs.conda.io/en/latest/miniconda.html) and following the prompts to install on your system.

TODO: IMAGE

<!-- <p align="center">
    <img src="../../figures/mac_conda_page.png" style="width:90%">
</p> -->

**Step 2:**
Open the downloaded miniconda file and follow the installation prompts.

**Step 3:**
Open a terminal window. An image of this application is shown below.

<p align="center">
    <img src="../../figures/mac_terminal.png" alt="mac_terminal" title="mac_terminal" style="width:40%">
</p>

**Step 4:**
Verify that conda has been installed with the following command.

```zsh
conda --version
```

A response like `conda xx.xx.xx` indicates that it has been correctly installed.

**Step 5:**
Update conda and use the libmamba solver (makes downloading conda programs [MUCH faster](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community)):

```zsh
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

**Step 6:**
Install packages that help Jupyter notebooks read conda environments:

```zsh
conda install nb_conda nb_conda_kernels
```

**Step 5:**
Download the `behavysis_pipeline` source code from [here](https://github.com/tlee08/behavysis_pipeline)

**Step 6:**
In the terminal, navigate to the `behavysis_pipeline` folder:

```zsh
cd <path/to/behavysis_pipeline>
```

**Step 7:**
Run the following commands to install `behavysis_pipeline` to your computer:

```zsh
conda env create -f conda_env.yaml
```

This will create a conda virtual environment named `behavysis_pipeline_env`, with the `behavysis_pipeline` program.

**Step 8:**
Verify that `behavysis_pipeline` has been correctly installed by running:

```zsh
conda env list
```

You should see `behavysis_pipeline_env` listed in the terminal.
