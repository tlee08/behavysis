# Updating All Config Files in a Project

## Loading in all relevant packages

```python
from ba_pipeline import *
```

## Making the project and importing all experiments

The directory path of the project must be specified and must contain the experiment files you wish to analyse in a particular folder structure.

```python
proj_dir = "path/to/project" # Specify your project folder path here
proj = BAProject(proj_dir)

proj.importExperiments()
```

## Updating the configurations for the experiment

To update the config files of experiments, a "default" config JSON file is used like a template to modify the individual experiment's config file.
There are three settings for how the experiment config files are updated:

- `overwrite="add"`: Add only parameters from the default file to each experiment configs file, where the parameter does not already exist in the experiment configs.
- `overwrite="set"`: Add/overwrite all experiment parameters from the default file into the each experiment config file.
- `overwrite="reset"`: Completely reset each experiment config file to be a copy of the default file.

```python
configs_fp = "path/to/default_configs.json"
proj.updateConfigFiles(configs_fp, overwrite="add") # overwrite can also be "set" or "reset"
```
