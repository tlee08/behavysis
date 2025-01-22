# Training a Behaviour Classifier

## Loading in all relevant packages

```python
import os

from behavysis_pipeline.mixins.behav_mixin import BehavMixin
from behavysis_pipeline.behav_classifier import BehavClassifier
from behavysis_pipeline.behav_classifier.clf_models.clf_templates import ClfTemplates
from behavysis_pipeline.pipeline import Project

if __name__ == "__main__":
    root_dir = "."
    overwrite = True

    # Option 1: From BORIS
    # Define behaviours in BORIS
    behavs_ls = ["potential huddling", "huddling"]
    # Paths
    configs_dir = os.path.join(root_dir, "0_configs")
    boris_dir = os.path.join(root_dir, "boris")
    dst_dir = os.path.join(root_dir, "7_scored_behavs")
    # Getting names of all files
    names = [os.path.splitext(i)[0] for i in os.listdir(boris_dir)]
    for name in names:
        # Paths
        boris_fp = os.path.join(boris_dir, f"{name}.tsv")
        configs_fp = os.path.join(configs_dir, f"{name}.json")
        dst_fp = os.path.join(dst_dir, f"{name}.feather")
        # Making df from BORIS
        df = BehavMixin.import_boris_tsv(boris_fp, configs_fp, behavs_ls)
        # Saving df
        df.to_feather(dst_fp)
    # Making BehavClassifier objects
    for behav in behavs_ls:
        BehavClassifier.create_new_model(os.path.join(root_dir, "behav_models"), behav)

    # Option 2: From previous behavysis project
    proj = Project(root_dir)
    proj.import_experiments()
    # Making BehavClassifier objects
    BehavClassifier.create_from_project(proj)

    # Loading a BehavModel
    behav = "fight"
    model = BehavClassifier.load(
        os.path.join(root_dir, "behav_models", f"{behav}.json")
    )
    # Testing all different classifiers
    model.clf_eval_compare_all()
    # MANUALLY LOOK AT THE BEST CLASSIFIER AND SELECT
    # Example
    model.pipeline_build(ClfTemplates.dnn_1)
```
