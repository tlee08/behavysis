import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")

with app.setup:
    import shutil
    from pathlib import Path

    import marimo as mo

    from behavysis import Project
    from behavysis.behaviour_classifier import (
        VersionMetadata,
        promote_to_production,
        regenerate_leaderboard,
        train_all_models,
    )
    from behavysis.behaviour_classifier import storage as clf_storage
    from behavysis.constants import FEATURES_EXTRACTED_DIR
    from behavysis.transforms import boris_to_behaviour
    from behavysis.utils import configure_logger

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
    # `individuals` and `bodyparts` MUST match the source project's
    # `extract_features` config, or inference features will not align.
    behaviour_name = "attack"
    individuals = ["mouse1marked", "mouse2unmarked"]
    bodyparts = ["Nose", "BodyCentre", "LeftEar", "RightEar"]

    # A behavysis project whose experiments have been processed through the
    # `extract_features` stage — the source of the training features.
    training_project_dir = Path("/absolute/path/to/behavysis_project")
    names_ls = [i.stem for i in (training_project_dir / "1_raw_videos").iterdir()]

    # BORIS exports: one `{experiment}.tsv` per experiment, holding manual scores.
    boris_dir = Path("/absolute/path/to/boris_tsvs")

    overwrite = True
    return (
        behaviour_name,
        bodyparts,
        boris_dir,
        clf_dir,
        individuals,
        names_ls,
        overwrite,
        training_project_dir,
    )


@app.cell
def _(names_ls, training_project_dir):
    proj = Project(training_project_dir)
    proj.import_experiments(names_ls)
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
        # Supervised feature selection (fit on the train split only):
        feature_selection=True,           # drop uninformative columns
        variance_threshold=0.0,           # raise to prune near-constant features
        max_features=None,                # cap to top-k by RF importance (None = all)
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
    ## 4. Evaluate

    Training writes rich per-version artifacts to each version's
    `evaluation/` folder — inspect these before trusting a model:

    | File | What it tells you |
    | --- | --- |
    | `{train,val,test}_report.json` | precision / recall / f1 per split |
    | `test_confm.png` | confusion matrix on the held-out test set |
    | `test_pcutoffs.png` | metrics vs probability cutoff — use to pick `pcutoff` |
    | `test_logc.png` | predicted-probability distribution |
    | `feature_importance.png` | top features by importance (sklearn) |
    | `feature_report.json` | `n_features_total` vs used after selection |
    | `history.png` | train/val loss curve (torch only) |

    The cell below rebuilds the leaderboard and prints, per model, the
    test F1, accuracy, and `overfit_ratio` (train F1 − test F1; smaller is
    better) alongside the path to each model's evaluation artifacts.
    """)
    return


@app.cell
def _(clf_dir, versions):
    _ = versions  # rebuild after training
    board = regenerate_leaderboard(clf_dir)

    _rows = []
    for _entry in board.rankings:
        _version = _entry.version
        _eval_dir = clf_storage.eval_dir(clf_dir, _entry.model_type, _version)
        _meta = VersionMetadata.read_yaml(
            clf_storage.metadata_fp(clf_dir, _entry.model_type, _version)
        )
        _rows.append(
            {
                "model_type": _entry.model_type,
                "version": _version,
                "test_f1": _entry.test_f1_behav,
                "test_acc": _entry.test_accuracy,
                "overfit_ratio": _entry.overfit_ratio,
                "n_features": _meta.data.n_features,
                "n_features_selected": _meta.data.n_features_selected,
                "evaluation_dir": str(_eval_dir),
            }
        )
    _rows
    return (board,)


@app.cell
def _():
    mo.md("""
    ## 5. Promote the best model to production

    Auto-promotes the top-ranked model (highest `test_f1_behav`) to
    `production.yaml`. The cell warns if the winner looks weak or overfit —
    review section 4 before relying on it. To override, promote a specific
    model manually:

    ```python
    promote_to_production(clf_dir, "rf", "v003_2025-07-07T120000")
    ```
    """)
    return


@app.cell
def _(board, clf_dir):
    # Tune these gates to your behaviour and dataset.
    _min_test_f1 = 0.7
    _max_overfit = 0.15

    best = board.rankings[0]

    _warnings = []
    if best.test_f1_behav is None or best.test_f1_behav < _min_test_f1:
        _warnings.append(f"test_f1={best.test_f1_behav} < {_min_test_f1}")
    if best.overfit_ratio is not None and best.overfit_ratio > _max_overfit:
        _warnings.append(f"overfit_ratio={best.overfit_ratio} > {_max_overfit}")

    promote_to_production(clf_dir, best.model_type, best.version)

    _status = (
        "⚠️ PROMOTED WITH WARNINGS: " + "; ".join(_warnings)
        if _warnings
        else "✓ Promoted (passed quality gates)"
    )
    mo.md(f"**{_status}**\n\n{best.model_type} {best.version} → production.yaml")
    return


@app.cell
def _():
    mo.md("""
    ## 6. Use the classifier in a pipeline

    Point an experiment's `classify_behaviour` config at the classifier's
    `production.yaml`. The behaviour name and feature contract are read from
    there — nothing is duplicated in the experiment config. Set `pcutoff`
    using `test_pcutoffs.png` from section 4:

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
