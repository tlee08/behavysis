# For Windows

For more information about how to uninstall conda, see [here](https://docs.anaconda.com/free/anaconda/install/uninstall/).

**Step 1:**
Open an Anaconda PowerShell Prompt window (make sure this is specifically this application - some other command prompt windows make look similar but they won't work). An image of this application is shown below.
<p align="center">
    <img src="../../figures/windows_conda_powershell.png" alt="windows_conda_powershell" title="windows_conda_powershell" style="width:40%">
</p>

**Step 2:**
To uninstall `ba_env`, run the following command:
```zsh
conda env remove -n ba_env
```

**Step 3:**
Enter the following commands in the terminal to remove all associated conda files and programs.
```zsh
conda install anaconda-clean
anaconda-clean --yes
```

**Step 4:**
Open the File Explorer and delete your environment (`anaconda3\envs`) and package (`anaconda3\pkgs`) folders in your user folder.
