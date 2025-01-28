import os

import pandas as pd

from behavysis_pipeline.behav_classifier.behav_classifier import BehavClassifier
from behavysis_pipeline.behav_classifier.clf_models.clf_templates import DNN1
from behavysis_pipeline.pipeline.project import Project
from behavysis_pipeline.processes.export import Export

if __name__ == "__main__":
    root_dir = "."
    overwrite = True

    # Option 1: From BORIS
    # Define behaviours in BORIS
    behavs_ls = ["potential huddling", "huddling"]
    # Paths
    boris_dir = os.path.join(root_dir, "boris")
    behav_dir = os.path.join(root_dir, "7_scored_behavs")
    config_dir = os.path.join(root_dir, "0_configs")
    for i in os.listdir(boris_dir):
        name = os.path.splitext(i)[0]
        print(name)
        outcome = Export.boris2behav(
            src_fp=os.path.join(boris_dir, f"{name}.tsv"),
            dst_fp=os.path.join(behav_dir, f"{name}.parquet"),
            configs_fp=os.path.join(config_dir, f"{name}.json"),
            behavs_ls=behavs_ls,
            overwrite=overwrite,
        )
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
    model_fp = os.path.join(root_dir, "behav_models", behav)
    model = BehavClassifier.load(model_fp)
    # Testing all different classifiers
    model.clf_eval_all()
    # MANUALLY LOOK AT THE BEST CLASSIFIER AND SELECT
    model.pipeline_build_save(CNN1)

    # Example of evaluating model with novel data
    x = pd.read("path/to/features_extracted")
    y = pd.read("path/to/scored_behavs")
    # Evaluating classifier (results stored in "eval" folder)
    model.clf_eval_save_performance(x, y)

    # Example of using model for inference
    # Loading a BehavModel
    model = BehavClassifier.load(model_fp)
    # Loading classifier
    model.clf_load()
    # Getting data
    x = pd.read_parquet("path/to/features_extracted.parquet")
    # Running inference
    res = model.pipeline_run(x)
