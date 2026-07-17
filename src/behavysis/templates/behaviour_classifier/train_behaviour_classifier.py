import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")

with app.setup:
    import shutil
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    from behavysis import Project
    from behavysis.behaviour_classifier import (
        ClassifierFp,
        list_models,
        make_eval_result_choose_model,
        promote_best,
        train_all_models,
        write_contract,
    )
    from behavysis.constants import FEATURES_EXTRACTED_DIR
    from behavysis.transforms import boris_to_behaviour
    from behavysis.utils import configure_logger

    configure_logger()
    alt.data_transformers.enable("vegafusion")


@app.cell
def _():
    mo.md("""
    # Train & Evaluate a behavysis behaviour classifier

    A classifier is **self-contained** in its own directory (`clf_dir`). Its
    name is arbitrary — the behaviour it classifies is declared in the shared
    `contract.yaml`. The layout is:

    ```
    {clf_dir}/
        contract.yaml               # shared behaviour + feature contract
        active.yaml                 # {model_name: rf} — which model to use
        training_data/
            5_features_extracted/   # features (from a processed behavysis project)
            7_behaviour_scored/     # labels  (from BORIS, or a scored project)
        classifiers/
            rf/
                config.yaml         # TrainingRecipe (hyperparameters)
                model.joblib        # fitted sklearn Pipeline
                evaluation/         # plots, eval parquets
            xgb/
                ...
            logreg/
                ...
    ```

    This notebook: **assemble training data → train → evaluate → set active**.
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
    behaviour_name = "aggression"
    individuals = ["mouse1marked", "mouse2unmarked"]
    bodyparts = [
        "LeftEar",
        "RightEar",
        "Nose",
        "BodyCentre",
        "LeftFlankMid",
        "RightFlankMid",
        "TailBase1",
        "TailTip4",
    ]
    angles = [
        ("Nose", "BodyCentre", "TailBase1"),
    ]

    # A behavysis project whose experiments have been processed through the
    # `extract_features` stage — the source of the training features.
    training_project_dir = Path("/absolute/path/to/behavysis_project")
    names_ls = [i.stem for i in (training_project_dir / "1_raw_videos").iterdir()]

    # BORIS exports: one `{experiment}.tsv` per experiment, holding manual scores.
    boris_dir = Path("/absolute/path/to/boris_tsvs")

    # Metrics to show in the evaluation summary (after training).
    # Available: accuracy, precision, recall, f1, roc_auc, pr_auc,
    #            gini, tp, fp, fn, tn, n, n_positive,
    #            precision_at_recall_099, precision_at_recall_100
    view_metrics = [
        "accuracy", "precision", "recall", "f1",
        "roc_auc", "pr_auc", "gini",
    ]

    overwrite = True
    return (
        angles,
        behaviour_name,
        bodyparts,
        boris_dir,
        clf_dir,
        individuals,
        names_ls,
        overwrite,
        training_project_dir,
        view_metrics,
    )


@app.cell
def _(clf_dir):
    clf_proj = ClassifierFp(clf_dir)
    return (clf_proj,)


@app.cell
def _():
    mo.md("""
    ## 2. Assemble training data

    The classifier's `training_data/` mirrors the pipeline's stage folders.
    Copy the extracted features from the source project.
    """)
    return


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

    Writes `contract.yaml` then trains every registered model type.
    Each model gets a named directory (`classifiers/{model_name}/`)
    with its own `config.yaml`, `model.joblib`, and `evaluation/` folder.
    """)
    return


@app.cell
def _(angles, behaviour_name, bodyparts, contract_fp, individuals):
    write_contract(
        contract_fp=contract_fp,
        behaviour_name=behaviour_name,
        individuals=individuals,
        bodyparts=bodyparts,
        angles=angles,
    )
    return


@app.cell
def _(clf_proj, overwrite):
    train_all_models(clf_proj.contract_fp(), overwrite=overwrite)
    return


@app.cell
def _():
    mo.md("""
    ## 4. Evaluate
    """)
    return


@app.cell
def _(clf_proj):
    trained_models = list_models(clf_proj.contract_fp())
    _msg = (
        f"Found **{len(trained_models)}** trained models: "
        f"{', '.join(trained_models)}"
    )
    mo.md(_msg)
    return (trained_models,)


@app.cell
def _(contract_fp, trained_models):
    # Load eval for every model that has both train and test eval parquets.
    eval_all = {}
    for _model in trained_models:
        _eval_dir = ClassifierFp(contract_fp.parent).eval_dir(_model)
        if (_eval_dir / "train_eval.parquet").exists() and (
            _eval_dir / "test_eval.parquet"
        ).exists():
            eval_all[_model] = make_eval_result_choose_model(contract_fp, _model)
    return (eval_all,)


@app.cell
def _():
    mo.md("""
    ### Metric summary
    """)
    return


@app.cell
def _(eval_all, view_metrics):
    rows = []
    for _model, _res in eval_all.items():
        _report = _res["report"]
        for _split, _metrics in _report.get("frame_report", {}).items():
            for _metric in view_metrics:
                if _metric in _metrics:
                    rows.append({
                        "model": _model,
                        "level": "frame",
                        "split": _split,
                        "metric": _metric,
                        "value": _metrics[_metric],
                    })
        for _split, _metrics in _report.get("bout_report", {}).items():
            for _metric in view_metrics:
                if _metric in _metrics:
                    rows.append({
                        "model": _model,
                        "level": "bout",
                        "split": _split,
                        "metric": _metric,
                        "value": _metrics[_metric],
                    })

    metrics_df = pl.DataFrame(
        rows,
        schema={
            "model": pl.String,
            "level": pl.String,
            "split": pl.String,
            "metric": pl.String,
            "value": pl.Float64,
        },
    ).sort(["level", "split", "metric", "model"])

    mo.ui.table(
        metrics_df.pivot(
            index=["model", "level", "split"],
            columns="metric",
            values="value",
        ),
        page_size=20,
    )
    return (metrics_df,)


@app.cell
def _(metrics_df, view_metrics):
    _df = metrics_df.filter(
        pl.col("level") == "frame",
        pl.col("split") == "test",
        pl.col("metric").is_in(view_metrics),
    )
    bar_chart = (
        alt.Chart(_df)
        .mark_bar()
        .encode(
            alt.X("metric", type="nominal", title=None),
            alt.Y("value", type="quantitative", scale=alt.Scale(domain=[0, 1])),
            alt.Color("model", type="nominal"),
            alt.XOffset("model"),
            alt.Column("metric", type="nominal", title=None)
            .header(labelOrient="bottom"),
        )
        .properties(height=200, width=80)
        .configure_axis(grid=False)
        .resolve_scale(x="independent")
    )
    mo.ui.altair_chart(bar_chart)
    return


@app.cell
def _():
    mo.md("""
    ### ROC & PR curves (test split)
    """)
    return


@app.cell
def _(eval_all):
    roc_parts = []
    pr_parts = []
    for _model, _res in eval_all.items():
        _roc = _res["df"].get("frame_roc_df")
        if _roc is not None:
            roc_parts.append(
                _roc.filter(pl.col("split") == "test")
                .select(["fpr", "tpr"])
                .with_columns(pl.lit(_model).alias("model"))
            )
        _pr = _res["df"].get("frame_pr_df")
        if _pr is not None:
            pr_parts.append(
                _pr.filter(pl.col("split") == "test")
                .select(["recall", "precision"])
                .with_columns(pl.lit(_model).alias("model"))
            )

    roc_df = pl.concat(roc_parts) if roc_parts else None
    pr_df = pl.concat(pr_parts) if pr_parts else None
    return pr_df, roc_df


@app.cell
def _():
    mo.md("""
    #### ROC
    """)
    return


@app.cell
def _(roc_df):
    if roc_df is not None:
        _diag = (
            alt.Chart(pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}))
            .mark_line(strokeDash=[4, 4], color="grey")
            .encode(x="x:Q", y="y:Q")
        )
        roc_chart = (
            alt.Chart(roc_df)
            .mark_line()
            .encode(
                x=alt.X("fpr:Q", title="False Positive Rate",
                        scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("tpr:Q", title="True Positive Rate",
                        scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("model:N"),
            )
        ) + _diag
        mo.ui.altair_chart(roc_chart.properties(width=500, height=400))
    return


@app.cell
def _():
    mo.md("""
    #### PR
    """)
    return


@app.cell
def _(pr_df):
    if pr_df is not None:
        pr_chart = (
            alt.Chart(pr_df)
            .mark_line()
            .encode(
                x=alt.X("recall:Q", title="Recall",
                        scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("precision:Q", title="Precision",
                        scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("model:N"),
            )
        )
        mo.ui.altair_chart(pr_chart.properties(width=500, height=400))
    return


@app.cell
def _():
    mo.md("""
    ### Bout health & review efficiency
    """)
    return


@app.cell
def _(eval_all):
    health_rows = []
    eff_rows = []
    for _model, _res in eval_all.items():
        _health = _res["report"].get("bout_health", {})
        for _split, _metrics in _health.items():
            _row = {"model": _model, "split": _split}
            _row.update(_metrics)
            health_rows.append(_row)
        _eff = _res["report"].get("review_efficiency", {})
        for _split, _metrics in _eff.items():
            eff_rows.append({
                "model": _model,
                "split": _split,
                "efficiency": _metrics.get("efficiency", 0),
            })

    health_df = pl.DataFrame(health_rows) if health_rows else None
    eff_df = pl.DataFrame(eff_rows) if eff_rows else None
    return eff_df, health_df


@app.cell
def _():
    mo.md("""
    **Bout detection quality** (higher is better)
    """)
    return


@app.cell
def _(health_df):
    mo.ui.table(health_df, page_size=20) if health_df is not None else None
    return


@app.cell
def _():
    mo.md("""
    **Review efficiency** = predicted-positive / true-positive. Closer to 1.0 = fewer false positives to review.
    """)
    return


@app.cell
def _(eff_df):
    mo.ui.table(eff_df, page_size=20) if eff_df is not None else None
    return


@app.cell
def _():
    mo.md("""
    ### Per-model detail
    """)
    return


@app.cell
def _(eval_all):
    accordion_items = {}
    for _model, _res in eval_all.items():
        _panels = []
        for _name, _chart in _res["chart"].items():
            _panels.append(mo.vstack([
                mo.md(f"**{_name}**"),
                mo.ui.altair_chart(_chart),
            ]))
        if _panels:
            accordion_items[_model] = mo.vstack(_panels)

    mo.accordion(accordion_items, multiple=True) if accordion_items else mo.md(
        "No per-model detail charts."
    )
    return


@app.cell
def _():
    mo.md("""
    ## 5. Set the active model

    ``promote_best`` (below) picks the best model automatically based on
    the contract's ``eval_metric`` and writes ``active.yaml``.

    Or edit by hand:

    ```yaml
    model_name: rf
    ```
    """)
    return


@app.cell
def _(clf_proj):
    promote_best(clf_proj.contract_fp())
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
