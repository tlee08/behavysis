from pathlib import Path

import pandas as pd

from behavysis import Project
from behavysis.behaviour_classifier import BehaviourClassifier
from behavysis.funcs import boris2behaviour

if __name__ == "__main__":
    root_dir = Path.cwd()
    overwrite = True

    # Option 1: From BORIS
    # Define behaviours in BORIS
    behaviour_ls = ["potential huddling", "huddling"]
    # Paths
    boris_dir = root_dir / "boris"
    behav_dir = root_dir / "7_scored_behaviour"
    config_dir = root_dir / "0_config"
    for i in boris_dir.iterdir():
        name = i.stem
        print(name)
        outcome = boris2behaviour(
            src_fp=boris_dir / f"{name}.tsv",
            dst_fp=behav_dir / f"{name}.parquet",
            config_fp=config_dir / f"{name}.json",
            behaviour_ls=behaviour_ls,
            overwrite=overwrite,
        )
    # Making BehaviourClassifier objects
    for behaviour in behaviour_ls:
        BehaviourClassifier.create_from_project_dir(root_dir)

    # Option 2: From previous behavysis project
    proj = Project(root_dir)
    proj.import_experiments()
    # Making BehaviourClassifier objects
    BehaviourClassifier.create_from_project(proj)

    # Loading a BehavModel
    behaviour = "fight"
    model_fp = root_dir / "behav_models" / behaviour
    model = BehaviourClassifier.load(model_fp, behaviour)
    # Testing all different classifiers
    model.pipeline_training_all()
    # MANUALLY LOOK AT THE BEST CLASSIFIER AND SELECT
    model.clf = "CNN1"

    # Example of evaluating model with novel data
    x = pd.read_parquet("path/to/features_extracted")
    y = pd.read_parquet("path/to/scored_behaviour")
    # Evaluating classifier (results stored in "eval" folder)
    model.clf_eval_save_performance(x, y)

    # Example of using model for inference
    # Loading a BehavModel
    model = BehaviourClassifier.load(model_fp, behaviour)
    # Getting data
    x = pd.read_parquet("path/to/features_extracted.parquet")
    # Running inference
    res = model.pipeline_inference(x)
