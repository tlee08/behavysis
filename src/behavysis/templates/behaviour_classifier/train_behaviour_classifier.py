import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")

with app.setup:
    import io
    from pathlib import Path

    import altair as alt
    import joblib
    import marimo as mo
    import polars as pl

    from behavysis.behaviour_classifier import (
        ClassifierPaths,
        list_models,
        make_eval_result_choose_model,
        promote_best,
        train_all_models,
        write_contract,
    )
    from behavysis.behaviour_classifier.data import (
        load_all_data,
    )
    from behavysis.behaviour_classifier.evaluation import compute_shap
    from behavysis.transforms import boris_to_behaviour
    from behavysis.utils import configure_logger

    configure_logger()
    alt.data_transformers.enable("vegafusion")


@app.cell
def _():
    mo.md("""
    # Train & Evaluate a behaviour classifier

    A self-contained classifier directory. Layout:

    ```
    {clf_dir}/
        contract.yaml
        active.yaml                 # {model_name: xgb}
        classifiers/
            rf/
                recipe.yaml
                model.joblib
                evaluation/
            xgb/ ...
            logreg/ ...
    
    training_data/                 # elsewhere
        5_features_extracted/
        7_behaviour_scored/
    ```

    **assemble data → train → evaluate → set active**
    """)


@app.cell
def _():
    mo.md("""
    ## 1. Configure — edit these
    """)


@app.cell
def _():
    # Classifier root directory.
    clf_dir = Path("/absolute/path/to/behaviour_classifier")

    # Behaviour to classify — written to contract.yaml.
    behaviour_name = "aggression"

    # Source project (must have completed extract_features stage).
    training_data_dir = Path("/absolute/path/to/behavysis_project")
    names_ls = [i.stem for i in (training_data_dir / "1_raw_videos").iterdir()]

    # Directory of BORIS .tsv exports (one per experiment).
    boris_dir = Path("/absolute/path/to/boris_tsvs")

    # Metrics to show in evaluation summary.
    view_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "gini",
    ]

    overwrite = False
    return (
        behaviour_name,
        boris_dir,
        clf_dir,
        names_ls,
        overwrite,
        training_data_dir,
        view_metrics,
    )


@app.cell
def _(clf_dir):
    clf = ClassifierPaths(clf_dir)
    return (clf,)


@app.cell
def _():
    mo.md("""
    ### Labels from BORIS

    Convert BORIS `.tsv` exports into scored parquets, aligned to each
    experiment's metadata (fps, frame range).

    *(If the source project is already scored, copy
    `7_behaviour_scored/*.parquet` into `clf.labels_dir()` instead.)*
    """)


@app.cell
def _(behaviour_name, boris_dir, labels_dir, overwrite, proj):
    labels_dir.mkdir(parents=True, exist_ok=True)
    for _exp in proj.experiments:
        boris_to_behaviour(
            src_fp=boris_dir / f"{_exp.name}.csv",
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

    Writes `contract.yaml`, then trains every registered model.
    """)


@app.cell
def _(behaviour_name, clf):
    write_contract(
        clf=clf,
        behaviour_name=behaviour_name,
    ).model_dump()


@app.cell
def _(clf, training_data_dir, overwrite):
    train_all_models(clf=clf, training_data_dir=training_data_dir, overwrite=overwrite)


@app.cell
def _():
    mo.md("""
    ## 4. Evaluate
    """)


@app.function
def show_chart_img(chart, width=300):
    _file = io.BytesIO()
    chart.save(_file, format="png")
    return mo.image(_file, width=width)


@app.cell
def _(clf):
    trained_models = list_models(clf)
    _msg = (
        f"Found **{len(trained_models)}** trained models: {', '.join(trained_models)}"
    )
    mo.md(_msg)
    return (trained_models,)


@app.cell
def _(clf, trained_models):
    # Load eval for every model that has both train and test eval parquets.
    eval_all = {}
    for _model in trained_models:
        _eval_dir = clf.eval_dir(_model)
        if (_eval_dir / "train_eval.parquet").exists() and (
            _eval_dir / "test_eval.parquet"
        ).exists():
            eval_all[_model] = make_eval_result_choose_model(clf, _model)
    return (eval_all,)


@app.cell
def _():
    mo.md("""
    ### Metric summary
    """)


@app.cell
def _(eval_all, view_metrics):
    rows = []
    for _model, _res in eval_all.items():
        for _level in ["frame", "bout"]:
            for _split, _metrics in _res["report"].get(f"{_level}_report", {}).items():
                for _metric in view_metrics:
                    if _metric in _metrics:
                        rows.append(
                            {
                                "level": _level,
                                "split": _split,
                                "metric": _metric,
                                "model": _model,
                                "value": _metrics[_metric],
                            }
                        )

    metrics_df = pl.DataFrame(
        rows,
        schema={
            "level": pl.String,
            "split": pl.String,
            "metric": pl.String,
            "model": pl.String,
            "value": pl.Float64,
        },
    ).sort(["level", "split", "metric", "model"])

    mo.ui.table(
        metrics_df.pivot(
            on="metric",
            index=["model", "level", "split"],
            values="value",
        ),
        page_size=20,
    )
    return (metrics_df,)


@app.cell
def _(metrics_df, view_metrics):
    _df = metrics_df.filter(
        pl.col("metric").is_in(view_metrics),
    )
    bar_chart = (
        alt.Chart(_df)
        .mark_bar()
        .encode(
            alt.X("metric:N"),
            alt.Y("value:Q", scale=alt.Scale(domain=[0, 1])),
            alt.Color("model:N"),
            alt.XOffset("model"),
            alt.Column("level:N"),
            alt.Row("split:N"),
        )
        .properties(height=200, width=200)
    )
    mo.ui.altair_chart(bar_chart)


@app.cell
def _():
    mo.md("""
    ### ROC & PR curves
    """)


@app.cell
def _(eval_all):
    roc_parts = []
    pr_parts = []
    for _model, _res in eval_all.items():
        for _level in ["frame", "bout"]:
            _roc = _res["df"].get(f"{_level}_roc_df")
            if _roc is not None:
                roc_parts.append(
                    _roc.with_columns(
                        pl.lit(_model).alias("model"),
                        pl.lit(_level).alias("level"),
                    )
                )
            _pr = _res["df"].get(f"{_level}_pr_df")
            if _pr is not None:
                pr_parts.append(
                    _pr.with_columns(
                        pl.lit(_model).alias("model"),
                        pl.lit(_level).alias("level"),
                    )
                )

    roc_df = pl.concat(roc_parts) if roc_parts else None
    pr_df = pl.concat(pr_parts) if pr_parts else None
    return pr_df, roc_df


@app.cell
def _(roc_df):
    mo.stop(roc_df is None, mo.md("roc_df is None"))

    _diag = (
        alt.Chart(pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}))
        .mark_line(strokeDash=[4, 4], color="grey")
        .encode(x="x:Q", y="y:Q")
    )
    roc_chart = (
        alt.Chart(roc_df)
        .mark_line()
        .encode(
            alt.X("fpr:Q", title="False Positive Rate", scale=alt.Scale(domain=[0, 1])),
            alt.Y("tpr:Q", title="True Positive Rate", scale=alt.Scale(domain=[0, 1])),
            alt.Color("model:N"),
            alt.Column("level:N"),
            alt.Row("split:N"),
        )
    ).properties(width=300, height=300)

    # Must show as image, because too many points
    show_chart_img(roc_chart)


@app.cell
def _(pr_df):
    mo.stop(pr_df is None, mo.md("pr_df is None"))

    pr_chart = (
        alt.Chart(pr_df)
        .mark_line()
        .encode(
            alt.X("recall:Q", title="Recall", scale=alt.Scale(domain=[0, 1])),
            alt.Y("precision:Q", title="Precision", scale=alt.Scale(domain=[0, 1])),
            alt.Color("model:N"),
            alt.Column("level:N"),
            alt.Row("split:N"),
        )
    ).properties(width=300, height=300)

    # Must show as image, because too many points
    show_chart_img(pr_chart)


@app.cell
def _():
    mo.md("""
    ### Bout health
    """)


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
            eff_rows.append(
                {
                    "model": _model,
                    "split": _split,
                    "efficiency": _metrics.get("efficiency", 0),
                }
            )

    health_df = pl.DataFrame(health_rows) if health_rows else None
    eff_df = pl.DataFrame(eff_rows) if eff_rows else None
    return eff_df, health_df


@app.cell
def _(health_df):
    mo.ui.table(health_df, page_size=20) if health_df is not None else None


@app.cell
def _(health_df):
    _df = health_df.unpivot(index=["model", "split"], variable_name="metric")

    health_chart = (
        alt.Chart(_df)
        .mark_bar()
        .encode(
            alt.Y("value:Q"),
            alt.Color("model:N"),
            alt.XOffset("model"),
            alt.Column("metric:N"),
            alt.Row("split:N"),
        )
        .resolve_scale(y="independent")
        .properties(height=200, width=200)
    )
    mo.ui.altair_chart(health_chart)


@app.cell
def _(eff_df):
    mo.md("**Review efficiency** = pred_pos / true_pos")
    mo.ui.table(eff_df, page_size=20) if eff_df is not None else None


@app.cell
def _():
    mo.md("""
    ### Per-model detail
    """)


@app.cell
def _(eval_all):
    accordion_items = {}
    for _model, _res in eval_all.items():
        _panels = []
        for _name, _chart in _res["chart"].items():
            _panels.append(
                mo.vstack(
                    [
                        mo.md(f"**{_name}**"),
                        show_chart_img(_chart),
                    ]
                )
            )
        if _panels:
            accordion_items[_model] = mo.hstack(_panels)

    mo.accordion(accordion_items, multiple=True) if accordion_items else mo.md(
        "No per-model detail charts."
    )


@app.cell
def _():
    mo.md("""
    ### Model Explainability
    """)


@app.cell
def _(clf, behaviour_name):
    # Load model (sklearn)
    model = joblib.load(clf.recipe_fp("rf").with_name("model.joblib"))
    # Load data
    df = load_all_data(
        clf.features_dir(),
        clf.labels_dir(),
        behaviour_name,
    )
    # Get SHAP
    result = compute_shap(model, df)
    result


@app.cell
def _():
    mo.md("""
    ## 5. Set the active model

    ``promote_best`` picks the best model and writes ``active.yaml``.
    Or edit by hand:

    ```yaml
    model_name: rf
    ```
    """)


@app.cell
def _(clf):
    promote_best(clf)


@app.cell
def _():
    mo.md("""
    ## 6. Use in a pipeline

    Add to an experiment's config:

    ```yaml
    classify_behaviour:
        - contract_fp: /absolute/path/to/behaviour_classifier/contract.yaml
          sub_behaviour: []
    ```
    """)


if __name__ == "__main__":
    app.run()
