# For Linux

For more information about how to uninstall conda, see [here](https://docs.anaconda.com/free/anaconda/install/uninstall/).

**Step 1:**
Open a terminal window. An image of this application is shown below.

<p align="center">
    <img src="../../figures/mac_terminal.png" alt="mac_terminal" title="mac_terminal" style="width:40%">
</p>

**Step 2:**
To uninstall `behavysis_pipeline_env`, run the following command:

```zsh
conda env remove -n behavysis_pipeline_env
```

**Step 3:**
Enter the following commands in the terminal to remove all associated conda files and programs.

```zsh
conda install anaconda-clean
anaconda-clean --yes
```

**Step 4:**
Enter the following commands to delete conda itself.

```zsh
rm -rf ~/anaconda3
rm -rf ~/opt/anaconda3
rm -rf ~/.anaconda_backup
```

**Step 5:**
Edit your bash or zsh profile so conda it does not look for conda anymore.
Open each of these files (note that not all of them may exist on your computer), `~/.zshrc`, `~/.zprofile`, or `~/.bash_profile`, with the following command.

```zsh
open ~/.zshrc
open ~/.zprofile
open ~/.bash_profile
```

And delete the lines between the block of text shown in the image below.

<p align="center">
    <img src="../../figures/mac_conda_uninstall_profile.png" alt="mac_conda_uninstall_profile" title="mac_conda_uninstall_profile" style="width:60%">
</p>
