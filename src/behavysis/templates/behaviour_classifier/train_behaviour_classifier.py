import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")

with app.setup:
    import shutil
    from pathlib import Path

    import marimo as mo

    from behavysis import Project
    from behavysis.behaviour_classifier import (
        ClassifierContract,
        train_all_models,
        init_classifier,
        make_eval_report_choose_model
    )
    from behavysis.behaviour_classifier.storage import ClassifierFp
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
        active.yaml                 # {name: rf, iteration: 3} — which model to use
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
    return


@app.cell
def _(feats_dir, overwrite, proj):
    feats_dir.mkdir(parents=True, exist_ok=True)
    for _exp in proj.experiments:
        _src = _exp.get_fp(FEATURES_EXTRACTED_DIR)
        _dst = feats_dir / _src.name
        if overwrite or not _dst.exists():
            shutil.copyfile(_src, _dst)
    sorted(p.name for p in feats_dir.iterdir())
    return


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
    return


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
    return


@app.cell
def _():
    mo.md("""
    ## 3. Write contract and train

    `train_all_models` writes the shared `contract.yaml` if missing, then
    trains every registered model type in a new iteration. Each iteration
    gets a numbered directory with its own `config.yaml`, `model.joblib`,
    and `evaluation/` folder.
    """)
    return


@app.cell
def _(behaviour_name, bodyparts, contract_fp, individuals):
    init_classifier(
        contract_fp=contract_fp,
        behaviour_name=behaviour_name,
        individuals=individuals,
        bodyparts=bodyparts,
    )
    return


@app.cell
def _(clf_proj):
    iterations = train_all_models(clf_proj.contract_fp())
    iterations
    return


@app.cell
def _():
    mo.md("""
    ## 4. Inspect evaluation artifacts

    Each iteration's `evaluation/` folder contains:

    | File | What it tells you |
    | --- | --- |
    | `train_eval.parquet` | Raw eval: experiment, frame, y_true, y_prob, y_pred |
    | `test_eval.parquet` | Same for the held-out test split |
    | `feature_importance.png` | Top features by importance |
    | `feature_report.json` | Feature counts before/after selection |

    Iterations are numbered — pick the best one by inspecting test_eval.parquet
    metrics, or add your own analysis over the raw eval data.
    """)
    return


@app.cell
def _(clf_proj):
    eval_res = make_eval_report_choose_model(
        clf_proj.contract_fp(),
        "gxb_v2",
        1
    )
    eval_res

@app.cell
def _():
    mo.md("""
    ## 5. Set the active model

    Write ``active.yaml`` to point to the best iteration.  This is the model
    used by ``predict_df`` at inference time:

    ```python
    ClassifierActive(name="rf", iteration=3).write_yaml(clf_proj)
    ```

    Or edit `active.yaml` by hand:

    ```yaml
    name: rf
    iteration: 3
    ```
    """)
    return


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
    return


if __name__ == "__main__":
    app.run()
