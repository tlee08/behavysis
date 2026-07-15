import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")

with app.setup:
    import shutil
    from pathlib import Path

    import marimo as mo

    from behavysis import Project
    from behavysis.behaviour_classifier import (
        ClassifierFp,
        promote_best,
        train_all_models,
        write_contract,
    )
    from behavysis.constants import FEATURES_EXTRACTED_DIR
    from behavysis.transforms import boris_to_behaviour
    from behavysis.utils import configure_logger

    configure_logger()


@app.cell
def _():
    mo.md("""
    # Train a behavysis behaviour classifier

    A classifier is **self-contained** in its own directory (`clf_dir`). Its
    name is arbitrary — the behaviour it classifies is declared in the shared
    `contract.yaml`. The layout is:

    ```
    {clf_dir}/
        contract.yaml               # shared behaviour + feature contract
        active.yaml                 # {name: rf} — which model to use
        training_data/
            5_features_extracted/   # features (from a processed behavysis project)
            7_behaviour_scored/     # labels  (from BORIS, or a scored project)
        classifiers/
            rf-001/
                config.yaml         # TrainingRecipe (hyperparameters)
                model.joblib        # fitted sklearn Pipeline
                evaluation/         # plots, eval parquets
            rf-002/
                ...
            logreg-001/
                ...
    ```

    This notebook: **assemble training data → train → inspect → set active**.
    """)


@app.cell
def _():
    mo.md("""
    ## 1. Configure — edit these
    """)


@app.cell
def _():
    # The classifier directory (arbitrary name; created if missing).
    clf_dir = Path("/absolute/path/to/behaviour_classifier")

    # The behaviour and the feature contract the classifier is trained on.
    # These are written once to `contract.yaml` and validated against each
    # experiment's `extract_features` config at inference.
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
def _(clf_dir):
    clf_proj = ClassifierFp(clf_dir)


@app.cell
def _(clf_proj):
    feats_dir = clf_proj.features_dir()
    labels_dir = clf_proj.labels_dir()
    contract_fp = clf_proj.contract_fp()
    return contract_fp, feats_dir, labels_dir


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


@app.cell
def _(feats_dir, overwrite, proj):
    feats_dir.mkdir(parents=True, exist_ok=True)
    for _exp in proj.experiments:
        _src = _exp.get_fp(FEATURES_EXTRACTED_DIR)
        _dst = feats_dir / _src.name
        if overwrite or not _dst.exists():
            shutil.copyfile(_src, _dst)
    sorted(p.name for p in feats_dir.iterdir())


@app.cell
def _():
    mo.md("""
    ### Labels from BORIS

    Most training labels come from **manually scored videos exported from
    BORIS**. `boris_to_behaviour` converts each `.tsv` into a scored parquet,
    using the experiment's metadata (fps, start/stop frame) to align frames.

    *(Alternatively, if the source project is already scored, copy its
    `7_behaviour_scored/*.parquet` into `clf_proj.labels_dir()`
    instead of running this cell.)*
    """)


@app.cell
def _(behaviour_name, boris_dir, labels_dir, overwrite, proj):
    labels_dir.mkdir(parents=True, exist_ok=True)
    for _exp in proj.experiments:
        boris_to_behaviour(
            src_fp=boris_dir / f"{_exp.name}.tsv",
            dst_fp=labels_dir / f"{_exp.name}.parquet",
            metadata=_exp.read_metadata(),
            behaviour_ls=[behaviour_name],
            overwrite=overwrite,
        )
    sorted(p.name for p in labels_dir.iterdir())


@app.cell
def _():
    mo.md("""
    ## 3. Write contract and train

    `train_all_models` writes the shared `contract.yaml` if missing, then
    trains every registered model type. Each model gets
    a numbered directory with its own `config.yaml`, `model.joblib`,
    and `evaluation/` folder.
    """)


@app.cell
def _(behaviour_name, bodyparts, contract_fp, individuals):
    write_contract(
        contract_fp=contract_fp,
        behaviour_name=behaviour_name,
        individuals=individuals,
        bodyparts=bodyparts,
    )


@app.cell
def _(clf_proj):
    train_all_models(clf_proj.contract_fp())


@app.cell
def _():
    mo.md("""
    ## 4. Inspect evaluation artifacts

    Each model's `evaluation/` folder contains:

    | File | What it tells you |
    | --- | --- |
    | `train_eval.parquet` | Raw eval: experiment, frame, y_true, y_prob, y_pred |
    | `test_eval.parquet` | Same for the held-out test split |
    | `feature_importance.png` | Top features by importance |
    | `feature_report.yaml` | Feature counts before/after selection |

    Pick the best one by inspecting test_eval.parquet
    metrics, or add your own analysis over the raw eval data.
    """)


@app.cell
def _():
    mo.md("""
    ## 5. Set the active model

    Write ``active.yaml`` to point to the best model.  This is the model
    used by ``predict_df`` at inference time:

    ```python
    ClassifierActive(name="rf").write_yaml(clf_proj)
    ```

    Or edit `active.yaml` by hand:

    ```yaml
    name: rf
    ```
    """)


@app.cell
def _(clf_proj):
    promote_best(clf_proj.contract_fp())


@app.cell
def _():
    mo.md("""
    ## 6. Use the classifier in a pipeline

    Point an experiment's `classify_behaviour` config at the classifier root
    (where `active.yaml` and `contract.yaml` live):

    ```yaml
    classify_behaviour:
        - clf_fp: /absolute/path/to/behaviour_classifier
            pcutoff: 0.5
            min_empty_window_secs: 0.2
            sub_behaviour: []
    ```

    The behaviour name, feature contract, and active model are all resolved
    automatically from the classifier directory.
    """)


if __name__ == "__main__":
    app.run()
