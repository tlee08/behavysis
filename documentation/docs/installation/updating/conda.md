# Updating BA_Package

**Step 1:**
Navigate to the following folder named "source" in the terminal.
You can do this by navigating to the folder using File Explorer and dragging this folder into the terminal. Make sure to type `cd ` in the terminal app before dragging the folder into the terminal.
<p align="center">
    <img src="../../figures/mac_source_location.png" alt="mac_source_location" title="mac_source_location" style="width:80%">
</p>

```zsh
cd </path/to/the/source/folder>
```

**Step 2:**
Run the following commands to open the conda virtual environment where the program is stored and update the program.
```zsh
conda env update -n ba_env --file BA_Package.yaml
```
