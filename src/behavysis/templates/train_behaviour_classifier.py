import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")

with app.setup:
    from pathlib import Path

    import marimo as mo

    from behavysis import Project
    from behavysis.funcs import (
        distance,
        dur_frames_from_likelihood,
        in_roi,
        interpolate,
        px_per_mm,
        speed,
        start_frame_from_likelihood,
        start_stop_trim,
        stop_frame_from_dur,
    )
    from behavysis.funcs.analyse import social_distance
    from behavysis.utils import configure_logger
    import marimo as mo
    import shutil
    from pathlib import Path

    from behavysis import Project
    from behavysis.behaviour_classifier import (
        promote_to_production,
        regenerate_leaderboard,
        train_all_models,
    )
    from behavysis.behaviour_classifier import storage as clf_storage
    from behavysis.constants import FEATURES_EXTRACTED_DIR
    from behavysis.transforms import boris_to_behaviour

    configure_logger()


@app.cell
def _():
    mo.md("""
    # Train a behavysis behaviour classifier

    A classifier is **self-contained** in its own directory (`clf_dir`). Its
    name is arbitrary — the behaviour it classifies is declared in each
    model's `config.yaml`. The layout is:

    ```
    {clf_dir}/
        training_data/
            5_features_extracted/   # features (from a processed behavysis project)
            7_behaviour_scored/     # labels  (from BORIS, or a scored project)
        {model_type}/
            config.yaml             # authored TrainingRecipe (behaviour + contract)
        leaderboard.yaml            # cross-model comparison
        production.yaml             # deployed pointer + feature contract
    ```

    This notebook: **assemble training data → train → compare → promote**.
    """)
    return


@app.cell
def _():
    mo.md("""
    ## 1. Configure — edit these
    """)
    return


@app.cell
def _():
    # The classifier directory (arbitrary name; created if missing).
    clf_dir = Path("/absolute/path/to/behaviour_classifier")

    # The behaviour and the feature contract the model is trained on.
    # These become the classifier's public contract in `production.yaml` and are
    # validated against each experiment's `extract_features` config at inference.
    behaviour_name = "attack"
    individuals = ["mouse1marked", "mouse2unmarked"]
    bodyparts = ["Nose", "BodyCentre", "LeftEar", "RightEar"]

    # A behavysis project whose experiments have been processed through the
    # `extract_features` stage — the source of the training features.
    training_project_dir = Path("/absolute/path/to/behavysis_project")
    experiment_names = ["exp1", "exp2"]

    # BORIS exports: one `{experiment}.tsv` per experiment, holding manual scores.
    boris_dir = Path("/absolute/path/to/boris_tsvs")

    overwrite = True
    return (
        behaviour_name,
        bodyparts,
        boris_dir,
        clf_dir,
        experiment_names,
        individuals,
        overwrite,
        training_project_dir,
    )


@app.cell
def _(experiment_names, training_project_dir):
    proj = Project(training_project_dir)
    proj.import_experiments(experiment_names)
    proj.experiments
    return (proj,)


@app.cell
def _():
    mo.md("""
    ## 2. Assemble training data

    The classifier's `training_data/` mirrors the pipeline's stage folders.
    Copy the extracted features from the source project.
    """)
    return


@app.cell
def _(clf_dir, overwrite, proj):
    feats_dst = clf_storage.features_dir(clf_dir)
    feats_dst.mkdir(parents=True, exist_ok=True)
    for _exp in proj.experiments:
        _src = _exp.get_fp(FEATURES_EXTRACTED_DIR)
        _dst = feats_dst / _src.name
        if overwrite or not _dst.exists():
            shutil.copyfile(_src, _dst)
    sorted(p.name for p in feats_dst.iterdir())
    return (feats_dst,)


@app.cell
def _():
    mo.md("""
    ### Labels from BORIS

    Most training labels come from **manually scored videos exported from
    BORIS**. `boris_to_behaviour` converts each `.tsv` into a scored parquet,
    using the experiment's metadata (fps, start/stop frame) to align frames.

    *(Alternatively, if the source project is already scored, copy its
    `7_behaviour_scored/*.parquet` into `clf_storage.labels_dir(clf_dir)`
    instead of running this cell.)*
    """)
    return


@app.cell
def _(behaviour_name, boris_dir, clf_dir, overwrite, proj):
    labels_dst = clf_storage.labels_dir(clf_dir)
    labels_dst.mkdir(parents=True, exist_ok=True)
    for _exp in proj.experiments:
        boris_to_behaviour(
            src_fp=boris_dir / f"{_exp.name}.tsv",
            dst_fp=labels_dst / f"{_exp.name}.parquet",
            metadata=_exp.read_metadata(),
            behaviour_ls=[behaviour_name],
            overwrite=overwrite,
        )
    sorted(p.name for p in labels_dst.iterdir())
    return (labels_dst,)


@app.cell
def _():
    mo.md("""
    ## 3. Train

    `train_all_models` authors a default `TrainingRecipe` (`config.yaml`) for
    every registered model type that lacks one, then trains them all. To tune
    hyperparameters, edit the written `config.yaml` and re-run, or author one
    explicitly and train a single model:

    ```python
    from behavysis.behaviour_classifier import train
    from behavysis.behaviour_classifier.config import TrainingRecipe
    from behavysis.behaviour_classifier.storage import config_fp

    TrainingRecipe(
        model_type="rf",
        behaviour_name=behaviour_name,   # required
        individuals=individuals,          # required feature contract
        bodyparts=bodyparts,              # required feature contract
        epochs=200,
        pcutoff=0.3,
    ).write_yaml(config_fp(clf_dir, "rf"))
    train(clf_dir, "rf")
    ```
    """)
    return


@app.cell
def _(behaviour_name, bodyparts, clf_dir, feats_dst, individuals, labels_dst):
    if not any(feats_dst.iterdir()):
        msg = f"No features in {feats_dst}"
        raise FileNotFoundError(msg)
    if not any(labels_dst.iterdir()):
        msg = f"No labels in {labels_dst}"
        raise FileNotFoundError(msg)

    versions = train_all_models(clf_dir, behaviour_name, individuals, bodyparts)
    versions
    return (versions,)


@app.cell
def _():
    mo.md("""
    ## 4. Compare models and promote the best to production
    """)
    return


@app.cell
def _(clf_dir, versions):
    _ = versions  # rebuild after training
    board = regenerate_leaderboard(clf_dir)
    board.rankings
    return (board,)


@app.cell
def _(board, clf_dir):
    best = board.rankings[0]
    promote_to_production(clf_dir, best.model_type, best.version)
    best
    return


@app.cell
def _():
    mo.md("""
    ## 5. Use the classifier in a pipeline

    Point an experiment's `classify_behaviour` config at the classifier's
    `production.yaml`. The behaviour name and feature contract are read from
    there — nothing is duplicated in the experiment config:

    ```yaml
    classify_behaviour:
      - clf_fp: /absolute/path/to/behaviour_classifier/production.yaml
        pcutoff: 0.5
        min_empty_window_secs: 0.2
        user_defined: []
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
