# Analysing a Folder of Experiments

All outcomes for experiment processing is stored in csv files in the `proj_dir/diagnostics` folder. These files store the outcome and process description (i.e. error explanations) of all experiments.

## Loading in all relevant packages

```python
from behavysis import Project
from behavysis.processes import *
```

## Making the project and importing all experiments

The directory path of the project must be specified and must contain the experiment files you wish to analyse in a particular folder structure.

For more information on how to structure a project directory, please see [setup][].

For more information on how a `Experiment` works, please see [behavysis.pipeline.project.Project][].

```python
# Defining the project's folder
proj_dir = "./project"
# Initialising the project
proj = Project(proj_dir)
# Importing all the experiments (from the project folder)
proj.importExperiments()
```

## Checking all imported experiments

To see all imported experiments, see the `proj_dir/diagnostics/importExperiments.csv` file that has been generated.

## Updating the configurations for all experiments

If you would like the configurations (which are stored in config files) to be updated new parameters, define the JSON style of configuration parameters you would like to add and run the following lines.

For more information about how a configurations file works, please see [here][configs-json-file].

```python
# Defining the default configs json path
configs_fp = "path/to/default_configs.json"
# Overwriting the configs
proj.update_configs(
    configs_fp,
    overwrite="user",
)
```

## Get Animal Keypoints in Videos

The following code processes and analyses all experiments that have been imported into a project. This is similar to analysing a single experiment.

### Downsample videos

Formatting the raw mp4 videos so it can be fed through the DLC pose estimation algorithm.

```python
proj.format_vid(
    (
        FormatVid.format_vid,
        FormatVid.get_vid_metadata,
    ),
    overwrite=True,
)
```

### Run Keypoints detection (DeepLabCut)

Running the DLC pose estimation algorithm on the formatted mp4 files.

!!! Note

    Make sure to change the `user.run_dlc.model_fp`
    to the DeepLabCut model's config file you'd like to use.

```python
proj.run_dlc(
    gputouse=None,
    overwrite=True,
)
```

### Calculating Inherent Parameters from Keypoints

Calculating relevant parameters to store in the `auto` section of the config file. The calculations performed are:

```python
proj.calculate_params(
    (
        CalculateParams.start_frame,
        CalculateParams.stop_frame,
        CalculateParams.exp_dur,
        CalculateParams.px_per_mm,
    )
)
```

And see a collation of all experiments' inherent parameters to spot any anomolies before continuing

```python
proj.collate_configs_auto()
```

### Postprocessing

Preprocessing the DLC csv data and output the preprocessed data to a `preprocessed_csv.<exp_name>.csv` file. The preprocessings performed are:

```python
proj.preprocess(
    (
        Preprocess.start_stop_trim,
        Preprocess.interpolate,
        Preprocess.refine_ids,
    ),
    overwrite=overwrite,
)
```

## Make Simple Analysis

Analysing the preprocessed csv data to extract useful analysis and results. The analyses performed are:

```python
proj.analyse(
    (
        Analyse.thigmotaxis,
        Analyse.center_crossing,
        Analyse.in_roi,
        Analyse.speed,
        Analyse.social_distance,
        Analyse.freezing,
    )
)
```

## Automated Behaviour Detection

### Extracting Features

Extracting derivative features from keypoints.
For example - speed, bounding ellipse size, distance between points, etc.

```python
proj.extract_features(overwrite)
```

### Running Behaviour Classifiers

!!! Note

    Make sure to change the `user.classify_behaviours` list
    to the behaviours classifiers you'd like to use.

```python
proj.classify_behaviours(overwrite)
```

### Exporting the Behaviour Detection Results

Exports to such a format, where
a) `behavysis_viewer` can load it and perform semi-automated analysis, and
b) after semi-automated verification, can be used to make a new/improve
a current behaviour classifier
(with [behavysis.behav_classifier.behav_classifier.BehavClassifier][])

```python
proj.export_behaviours(overwrite)
```

### Analyse Behaviours

Similar to [simple analysis][make-simple-analysis], which calculates each experiment's
a) the overall summary, and
b) binned summary.

```python
proj.behav_analyse()
```

## Export any Tables

Tables are stored as `.feather` files.

To export these to csv files, run the following:

```python
proj.export_feather("7_scored_behavs", "/path/to/csv_out")
```

## Evaluate

Evaluates keypoints and behaviours accuracy by making annotated experiment videos.

```python
proj.evaluate(
    (
        Evaluate.eval_vid,
        Evaluate.keypoints_plot,
    ),
    overwrite=overwrite,
)
```
