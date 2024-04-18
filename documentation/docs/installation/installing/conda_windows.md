# For Windows

**Step 1:**
Install conda by visiting the [Miniconda downloads page](https://docs.conda.io/en/latest/miniconda.html) and following the prompts to install on Windows.

<p align="center">
    <img src="../../figures/windows_conda_page.png" alt="windows_conda_page" title="windows_conda_page" style="width:90%">
</p>

**Step 2:**
Open the downloaded miniconda file and follow the installation prompts.

**Step 3:**
Open an Anaconda PowerShell Prompt window (make sure this is specifically this application - some other command prompt windows make look similar but they won't work). An image of this application is shown below. Ensure you use "Run as Administrator" mode.

<p align="center">
    <img src="../../figures/windows_conda_powershell.png" alt="windows_conda_powershell" title="windows_conda_powershell" style="width:40%">
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
You can do this by navigating to the folder using File Explorer and dragging this folder into the terminal. Make sure to type `cd ` in the terminal app before dragging the folder into the terminal.

<p align="center">
    <img src="../../figures/windows_source_location.png" alt="windows_source_location" title="windows_source_location" style="width:80%">
</p>

```zsh
cd <\path\to\the\source\folder>
```

**Step 7:**
Create a conda virtual environment, where the program can run.
The virtual environment will be called "ba_env".

```zsh
conda env create -f ba_pipeline.yaml
```

**Step 8:**
Verify that `ba_pipeline` has been correctly installed by running:

```zsh
conda env list
```

You should see `ba_env` listed in the terminal.
