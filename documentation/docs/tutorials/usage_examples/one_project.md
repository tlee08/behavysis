# Analysing a Single Project

## Loading in all relevant packages

```python
from behavysis_pipeline import *
```

## Making the project and importing all experiments

The directory path of the project must be specified and must contain the experiment files you wish to analyse in a particular folder structure.

For more information on how to structure a project directory, please see [setup](../setup.md).

For more information on how a `BAExperiment` works, please see [BAProject](../../reference/BAProject.md).

```python
proj_dir = "./project"
proj = BAProject(proj_dir)

proj.importExperiments()
```

# Checking all imported experiments

To see all imported experiments, see the `proj_dir/diagnostics/importExperiments.csv` file that has been generated.

## Updating the configurations for all experiments

If you would like the configurations (which are stored in config files) to be updated new parameters, define the JSON style of configuration parameters you would like to add and run the following lines.

For more information about how a configurations file works, please see [the configs file](../configs_setup/configs_setup.md).

```python
configs_fp = "path/to/default_configs.json"
proj.updateConfigFiles(configs_fp, overwrite="add")
```

## Analysing all experiments in a project

The following code processes and analyses all experiments that have been imported into a project. This is similar to analysing a single experiment.

All outcomes for experiment processing is stored in csv files in the `proj_dir/diagnostics` folder. These files store the outcome and process description (i.e. error explanations) of all experiments.

Each of the following processes will be explained prior to running the process.

`formatVideo` formats the raw mp4 videos so it can be fed through the DLC pose estimation algorithm.

```python
proj.formatVideos()
```

`runDLC`: Runs the DLC pose estimation algorithm on the formatted mp4 files.

```python
proj.runDLC()
```

`calculateParams` calculates relevant parameters to store in the `auto` section of the config file. The calculations performed are:

- `calcStartFrame`: calculates the start frame of the video based on when the subject entered the video.
- `calcEndFrame`: calculates the end frame of the video based on the start frame and user-defined duration.
- `calcPXPerMM`: calculates the pixels per mm conversion for the video based on the user-defined height and width of the arena.

```python
proj.calculateParams(getVideoMetadata, calcStartFrame, calcEndFrame, calcPXPerMM)
```

`preprocess` preprocesses the DLC csv data and output the preprocessed data to a `preprocessed_csv.<exp_name>.csv` file. The preprocessings performed are:

- `trimToStartEnd`: Trims the csv data between the experiment start and end frames, which are either defined by the user or from `calculateParams`.
- `interpolatePoints`: Linearly interpolates subject points that have an accuracy below a user-defined pcutoff.
- `calcBodyCentre`: Calculate the average "centre" of the subject, based on the user-defined list of body parts and name these $(x,y)$ points with a user-defined name. This $(x,y)$ data is added as a column to the raw data.
- `mapTimestamps`: adds a column to the dataframe which is the time (in seconds) from the beginning of the video for each frame.

```python
proj.preprocess(trimToStartEnd, interpolatePoints, calcBodyCentre, mapTimestamps)
```

`analyse` analyses the preprocessed csv data to extract useful analysis and results. The analyses performed are:

- `analyseDistance`: Analyses the distance that the subject travels over time.
- `analyseSpeed`: Analyses the speed of the subject over time.
- `analyseThigmotaxis`: Analyses the position and time spent by the subject in thigmotaxis.

```python
proj.analyse(analyseDistance, analyseSpeed, analyseThigmotaxis)
```

`aggregateAnalysis` combines the analysis from all experiments for each different analysis metric. For example, each subject's distance travelled for every **_x_** seconds is combined into a single csv file for comparison.

```python
proj.aggregateAnalysis()
```
