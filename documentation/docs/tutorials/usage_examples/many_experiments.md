# Analysing Many Experiments

## Loading in all relevant packages

```python
from ba_pipeline import *
```

## Making the Experiments

The directory path of the project must be specified and must contain the experiment files you wish to analyse in a particular folder structure.

The process of analysing multiple experiments is just analysing each experiment and performing this multiple times (with a loop).

For more information on how to structure a project directory, please see [setup](../setup.md).

For more information on how a `BAExperiment` works, please see [BAExperiment](../../reference/BAExperiment.md).

```python
proj_dir = "./project"
experiment_names = exp_name = ["experiment1", "experiment2", "experiment3"]
experiments = list()

for name in experiment_names:
    exp = BAExperiment(name, proj_dir)
    experiments.append(exp)
```

## Updating the configurations for the experiment

If you would like the configurations (which are stored in config files) to be updated new parameters, define the JSON style of configuration parameters you would like to add and run the following lines.

For more information about how a configurations file works, please see [the configs file](../configs_setup/configs_setup.md).

```python
configs_fp = "path/to/default_configs.json"
for exp in experiments:
    exp.updateConfigFile(configs_fp, overwrite="add")
```

## Analysing many experiments

Each of the following processes will be explained prior to running the process.

`formatVideo` formats the raw mp4 video so it can be fed through the DLC pose estimation algorithm.

```python
for exp in experiments:
    exp.formatVideo()
```

`runDLC`: Runs the DLC pose estimation algorithm on the formatted mp4 file.

```python
for exp in experiments:
    exp.runDLC()
```

`calculateParams` calculates relevant parameters to store in the `auto` section of the config file. The calculations performed are:

- `calcStartFrame`: calculates the start frame of the video based on when the subject entered the video.
- `calcEndFrame`: calculates the end frame of the video based on the start frame and user-defined duration.
- `calcPXPerMM`: calculates the pixels per mm conversion for the video based on the user-defined height and width of the arena.

```python
for exp in experiments:
    exp.calculateParams(getVideoMetadata, calcStartFrame, calcEndFrame, calcPXPerMM)
```

`preprocess` preprocesses the DLC csv data and output the preprocessed data to a `preprocessed_csv.<exp_name>.csv` file. The preprocessings performed are:

- `trimToStartEnd`: Trims the csv data between the experiment start and end frames, which are either defined by the user or from `calculateParams`.
- `interpolatePoints`: Linearly interpolates subject points that have an accuracy below a user-defined pcutoff.
- `calcBodyCentre`: Calculate the average "centre" of the subject, based on the user-defined list of body parts and name these $(x,y)$ points with a user-defined name. This $(x,y)$ data is added as a column to the raw data.
- `mapTimestamps`: adds a column to the dataframe which is the time (in seconds) from the beginning of the video for each frame.

```python
for exp in experiments:
    exp.preprocess(trimToStartEnd, interpolatePoints, calcBodyCentre, mapTimestamps)
```

`analyse` analyses the preprocessed csv data to extract useful analysis and results. The analyses performed are:

- `analyseDistance`: Analyses the distance that the subject travels over time.
- `analyseSpeed`: Analyses the speed of the subject over time.
- `analyseThigmotaxis`: Analyses the position and time spent by the subject in thigmotaxis.

```python
for exp in experiments:
    exp.analyse(analyseDistance, analyseSpeed, analyseThigmotaxis)
```
