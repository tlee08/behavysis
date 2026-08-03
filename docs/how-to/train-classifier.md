# Train a Behaviour Classifier

Train a machine learning model to automatically classify behaviours from keypoint
features.

## Prerequisites

- A completed project with scored behaviours in `7_behaviour_scored/`
- Labelled training data (from BORIS or a previous behavysis project)

## Architecture

The classifier extracts features from keypoints (via SimBA), trains a model on
labelled data, then predicts behaviours on new experiments. Pipeline stages 5-7.

See [BehaviourClassifier API](../reference/behaviour_classifier.md) for the full API.

## Quick start

Use the `behaviour_pipeline` preset if you already have a trained classifier:

```bash
behavysis-make-project --preset behaviour_pipeline
```

Edit `classify_behaviour` in the config to point to your trained model:

```yaml
classify_behaviour:
  - proj_dir: /path/to/training/project
    behaviour_name: my_behaviour
    pcutoff: 0.5
    min_empty_window_secs: 0.2
```

## Training a new model

```python
from behavysis.behaviour_classifier import BehaviourClassifier

# From a previous behavysis project with scored behaviours
proj = Project("/path/to/scored_project")
proj.import_experiments(name_ls)
BehaviourClassifier.create_from_project(proj)
```

After training, use the exported model in new projects by pointing
`classify_behaviour.proj_dir` to the training project directory.
