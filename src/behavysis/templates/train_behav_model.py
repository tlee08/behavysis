from pathlib import Path

import pandas as pd

from behavysis import Project
from behavysis.behaviour_classifier import BehaviourClassifier, BehaviourClassifierConfig
from behavysis.funcs import boris2behaviour

if __name__ == "__main__":
    root_dir = Path.cwd()
    overwrite = True

    # Option 1: From BORIS
    # Define behaviours in BORIS
    behaviour_ls = ["potential huddling", "huddling"]
    # Paths
    boris_dir = root_dir / "boris"
    behaviour_dir = root_dir / "7_scored_behaviour"
    config_dir = root_dir / "0_config"
    for i in boris_dir.iterdir():
        name = i.stem
        print(name)
        outcome = boris2behaviour(
            src_fp=boris_dir / f"{name}.tsv",
            dst_fp=behaviour_dir / f"{name}.parquet",
            config_fp=config_dir / f"{name}.json",
            behaviour_ls=behaviour_ls,
            overwrite=overwrite,
        )
    # Making BehaviourClassifier objects for all behaviours
    for behaviour in behaviour_ls:
        BehaviourClassifier.create_all_from_project_dir(root_dir)

    # Option 2: From previous behavysis project
    proj = Project(root_dir)
    proj.import_experiments()

    # Create classifiers for all labelled behaviours
    BehaviourClassifier.create_from_project(proj)

    # Train a specific model
    config = BehaviourClassifierConfig(
        behaviour_name="attack",
        model_type="rf",
        individuals=["mouse1marked", "mouse2unmarked"],
        bodyparts=["Nose", "BodyCentre", "LeftEar", "RightEar"],
        pcutoff=0.2,
    )
    clf = BehaviourClassifier.create(root_dir, "attack", config)
    clf.train()

    # Train all model types for comparison
    from behavysis.behaviour_classifier import train_all_models

    train_all_models(
        root_dir,
        "attack",
        config_overrides={
            "individuals": ["mouse1marked", "mouse2unmarked"],
            "bodyparts": ["Nose", "BodyCentre", "LeftEar", "RightEar"],
        },
    )

    # Load a trained classifier
    clf = BehaviourClassifier.load(root_dir, "attack")
    # Or load a specific model type
    clf = BehaviourClassifier.load(root_dir, "attack", model_type="dnn1")

    # Inference
    x = pd.read_parquet("path/to/features_extracted.parquet")
    res = clf.predict(x)
