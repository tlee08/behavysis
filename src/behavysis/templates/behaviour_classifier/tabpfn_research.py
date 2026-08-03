import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")

with app.setup:
    import os
    from pathlib import Path

    import altair as alt
    import numpy as np
    import polars as pl
    from tabpfn import TabPFNClassifier

    from behavysis.behaviour_classifier import (
        make_eval_result,
    )
    from behavysis.behaviour_classifier.data import (
        label_bouts,
        load_all_data,
        stratified_split_by_group,
    )
    from behavysis.behaviour_classifier.storage import ClassifierPaths
    from behavysis.constants import (
        ACTUAL,
        BOUT_ID,
        EXPERIMENT,
        FRAME,
        PRED,
        Array1D,
        Array2D,
    )
    from behavysis.utils import configure_logger

    configure_logger()
    alt.data_transformers.enable("vegafusion")


@app.cell
def _():
    clf_dir = Path.cwd()
    behaviour_name = "aggression"

    clf = ClassifierPaths(clf_dir)
    contract_fp = clf.contract_fp()
    feats_dst = clf.features_dir("generic")
    labels_dst = clf.labels_dir()
    labels_dst.mkdir(parents=True, exist_ok=True)
    return behaviour_name, clf


@app.cell
def _(behaviour_name, clf):
    # Load and align data
    df = load_all_data(
        clf.features_dir("generic"),
        clf.labels_dir(),
        behaviour_name,
    )
    df = label_bouts(df, ACTUAL)
    return (df,)


@app.cell
def _(df):
    # Split into train / test (experiment-level grouping)
    train_idx, test_idx = stratified_split_by_group(df, 0.2, EXPERIMENT, 42)
    train_df = df[train_idx].sort([EXPERIMENT, FRAME])
    test_df = df[test_idx].sort([EXPERIMENT, FRAME])
    return


@app.cell
def _(df):
    # df.group_by(EXPERIMENT).agg(pl.col(ACTUAL).sum())

    exp1_df = df.filter(
        pl.col(EXPERIMENT) == "10_Round2_20230413_AGG-SS_test2-M1_a2_960"
    )
    exp2_df = df.filter(
        pl.col(EXPERIMENT) == "13_Round2_20230413_AGG-SS_test2-M1_a5_960"
    )
    return exp1_df, exp2_df


@app.function
def df_get_features(df: pl.DataFrame) -> Array2D:
    """Given a df, only return features (filters out metadata and label columns)."""
    return (
        df.drop([EXPERIMENT, FRAME, ACTUAL, BOUT_ID], strict=False)
        .to_numpy()
        .astype(np.float32)
    )


@app.function
def df_get_labels(df: pl.DataFrame) -> Array1D:
    """Given a df, return only the labels."""
    return df[ACTUAL].to_numpy()


@app.cell
def _():
    clf = TabPFNClassifier(
        n_estimators=16,
        device="cuda",
        random_state=42,
        fit_mode="fit_with_cache",
        ignore_pretraining_limits=True,
        balance_probabilities=True,
    )
    return (clf,)


@app.cell
def _(clf, exp1_df):
    clf.fit(df_get_features(exp1_df), df_get_labels(exp1_df))
    return


@app.cell
def _(clf, exp2_df):
    predictions = clf.predict(df_get_features(exp2_df))
    return (predictions,)


@app.cell
def _(exp2_df, predictions):
    from behavysis.constants import PROB

    eval_df = label_bouts(
        exp2_df.select([EXPERIMENT, FRAME, ACTUAL]).with_columns(
            pl.lit(predictions).alias(PROB), pl.lit(predictions).alias(PRED)
        ),
        ACTUAL,
    )

    eval_df
    return (eval_df,)


@app.cell
def _(eval_df):
    res = make_eval_result({"test": eval_df})

    res
    return


if __name__ == "__main__":
    app.run()
