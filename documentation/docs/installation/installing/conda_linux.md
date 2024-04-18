# For Linux

**Step 1:**
Install conda by visiting the [Miniconda downloads page](https://docs.conda.io/en/latest/miniconda.html) and following the prompts to install on Mac.
<p align="center">
    <img src="../../figures/mac_conda_page.png" alt="mac_conda_page" title="mac_conda_page" style="width:90%">
</p>

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
Connect to the RDS

**Step 6:**
Navigate to the following folder named "source" in the terminal.
You can do this by going to the folder using Finder and dragging this folder into the terminal. Make sure to type `cd ` in the terminal app before dragging the folder into the terminal.
<p align="center">
    <img src="../../figures/mac_source_location.png" alt="mac_source_location" title="mac_source_location" style="width:80%">
</p>

```zsh
cd <\path\to\the\source\folder>
```

**Step 7:**
Create a conda virtual environment, where the program can run.
The virtual environment will be called "ba_env".
```zsh
conda install nb_conda nb_conda_kernels
conda env create -f ba_package.yaml
```

**Step 8:**
Verify that `ba_package` has been correctly installed by running:
```zsh
conda env list
```
You should see `ba_env` listed in the terminal.
