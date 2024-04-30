# Analysing a Single Experiment

## Loading in all relevant packages

```python
from behavysis_pipeline import *
```

## Making the Experiment

The directory path of the project must be specified and must contain the experiment files you wish to analyse in a particular folder structure.

For more information on how to structure a project directory, please see [setup](../setup.md).

For more information on how a `Experiment` works, please see [Experiment](../../reference/experiment.md).

```python
proj_dir = "./project"
exp_name = "experiment1"
exp = Experiment(exp_name, proj_dir)
```

### ALTERNATIVE: Selecting an experiment from a Project

Alternatively, you can make a project and then specify the experiment to analyse from the project.
The name of an experiment is given by the name of its corresponding files (e.g., "<experiment_name>.mp4").

```python
# Making a project and importing experiments into the project
proj = Project(proj_dir)
proj.importExperiments()

# Showing the names of all experiments that have been imported to the project
for exp_name in proj.experiments:
    print(exp_name)

# Selecting the single experiment from the project to analyse
exp_name = "experiment1"
exp = proj.experiments["experiment1"]
```

## Updating the configurations for the experiment

If you would like the configurations (which are stored in config files) to be updated new parameters, define the JSON style of configuration parameters you would like to add and run the following lines.

For more information about how a configurations file works, please see the configs file.

```python
configs_fp = "path/to/default_configs.json"
exp.updateConfigFile(configs_fp, overwrite="add")
```

## Analysing a single experiment

Each of the following processes will be explained prior to running the process.

`formatVideo` formats the raw mp4 video so it can be fed through the DLC pose estimation algorithm.

```python
exp.formatVideo()
```

`runDLC`: Runs the DLC pose estimation algorithm on the formatted mp4 file.

```python
exp.runDLC()
```

`calculateParams` calculates relevant parameters to store in the `auto` section of the config file. The calculations performed are:

- `calcStartFrame`: calculates the start frame of the video based on when the subject entered the video.
- `calcEndFrame`: calculates the end frame of the video based on the start frame and user-defined duration.
- `calcPXPerMM`: calculates the pixels per mm conversion for the video based on the user-defined height and width of the arena.

```python
exp.calculateParams(getVideoMetadata, calcStartFrame, calcEndFrame, calcPXPerMM)
```

`preprocess` preprocesses the DLC csv data and output the preprocessed data to a `preprocessed_csv.<exp_name>.csv` file. The preprocessings performed are:

- `trimToStartEnd`: Trims the csv data between the experiment start and end frames, which are either defined by the user or from `calculateParams`.
- `interpolatePoints`: Linearly interpolates subject points that have an accuracy below a user-defined pcutoff.
- `mapTimestamps`: adds a column to the dataframe which is the time (in seconds) from the beginning of the video for each frame.

```python
exp.preprocess(trimToStartEnd, interpolatePoints, mapTimestamps)
```

`analyse` analyses the preprocessed csv data to extract useful analysis and results. The analyses performed are:

- `analyseDistance`: Analyses the distance that the subject travels over time.
- `analyseSpeed`: Analyses the speed of the subject over time.
- `analyseThigmotaxis`: Analyses the position and time spent by the subject in thigmotaxis.

```python
exp.analyse(analyseDistance, analyseSpeed, analyseThigmotaxis)
```
