"""Polars schema definitions and I/O utilities for behavysis DataFrames.

Each schema is a dict mapping column name → Polars dtype. One row = one atomic
observation. Long-form storage, pivot-on-demand at ML boundaries.
"""

from pathlib import Path

import polars as pl

from behavysis.constants import (
    ACTUAL,
    AGG,
    ANALYSIS,
    BEHAVIOUR,
    BIN_SEC,
    BODYPART,
    EXPERIMENT,
    FRAME,
    INDIVIDUAL,
    LIKELIHOOD,
    MEASURE,
    PRED,
    PROB,
    VALUE,
    X,
    Y,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Core schema dicts
# ═══════════════════════════════════════════════════════════════════════════════

type SchemaDict = dict[str, type[pl.DataType]]

"""One row per (frame, individual, bodypart) coordinate triplet."""
KEYPOINTS_SCHEMA: SchemaDict = {
    FRAME: pl.Int64,
    INDIVIDUAL: pl.Utf8,
    BODYPART: pl.Utf8,
    X: pl.Float64,
    Y: pl.Float64,
    LIKELIHOOD: pl.Float64,
}

"""Features stay wide for ML. ``frame`` + dynamic Float64 feature columns."""
FEATURES_BASE: SchemaDict = {
    FRAME: pl.Int64,
}

"""One row per (frame, behaviour) classifier prediction."""
BEHAVIOUR_PREDICTED_SCHEMA: SchemaDict = {
    FRAME: pl.Int64,
    BEHAVIOUR: pl.Utf8,
    PROB: pl.Float64,
    PRED: pl.Int64,
}

# BehaviourScoredDf base columns; sub_behaviour columns are dynamic and validated
# against BoutStruct at read/write boundaries.
BEHAVIOUR_SCORED_BASE: SchemaDict = {
    FRAME: pl.Int64,
    BEHAVIOUR: pl.Utf8,
    ACTUAL: pl.Int64,
}

"""One row per (frame, individual, measure) value. Frame-by-frame analysis."""
ANALYSIS_SCHEMA: SchemaDict = {
    FRAME: pl.Int64,
    INDIVIDUAL: pl.Utf8,
    MEASURE: pl.Utf8,
    VALUE: pl.Float64,
}

"""One row per (individual, measure, aggregation) statistic."""
SUMMARY_SCHEMA: SchemaDict = {
    INDIVIDUAL: pl.Utf8,
    MEASURE: pl.Utf8,
    AGG: pl.Utf8,
    VALUE: pl.Float64,
}

"""One row per (bin_sec, individual, measure, aggregation) time slice."""
BINNED_SCHEMA: SchemaDict = {
    BIN_SEC: pl.Float64,
    INDIVIDUAL: pl.Utf8,
    MEASURE: pl.Utf8,
    AGG: pl.Utf8,
    VALUE: pl.Float64,
}

"""AnalysisDf concatenated across analysis types with an ``analysis`` column."""
COMBINED_ANALYSIS_SCHEMA: SchemaDict = {
    FRAME: pl.Int64,
    ANALYSIS: pl.Utf8,
    INDIVIDUAL: pl.Utf8,
    MEASURE: pl.Utf8,
    VALUE: pl.Float64,
}

"""SummaryDf collated across experiments with an ``experiment`` column."""
COLLATED_SUMMARY_SCHEMA: SchemaDict = {
    EXPERIMENT: pl.Utf8,
    INDIVIDUAL: pl.Utf8,
    MEASURE: pl.Utf8,
    AGG: pl.Utf8,
    VALUE: pl.Float64,
}

"""BinnedDf collated across experiments with an ``experiment`` column."""
COLLATED_BINNED_SCHEMA: SchemaDict = {
    BIN_SEC: pl.Float64,
    EXPERIMENT: pl.Utf8,
    INDIVIDUAL: pl.Utf8,
    MEASURE: pl.Utf8,
    AGG: pl.Utf8,
    VALUE: pl.Float64,
}

# ═══════════════════════════════════════════════════════════════════════════════
# I/O utilities
# ═══════════════════════════════════════════════════════════════════════════════


def read_df(fp: Path, schema: SchemaDict | None = None) -> pl.DataFrame:
    """Read a parquet file and validate its schema at the I/O boundary."""
    df = pl.read_parquet(fp)
    if schema is not None:
        _check_schema(df, schema)
    return df


def write_df(df: pl.DataFrame, fp: Path, schema: SchemaDict | None = None) -> None:
    """Validate schema and write a parquet file, creating parent directories."""
    if schema is not None:
        _check_schema(df, schema)
    fp.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(fp)


def read_csv(fp: Path, schema: SchemaDict) -> pl.DataFrame:
    """Read a CSV file and validate schema. Used for SimBA boundary."""
    df = pl.read_csv(fp)
    _check_schema(df, schema)
    return df


def write_csv(df: pl.DataFrame, fp: Path, schema: SchemaDict) -> None:
    """Validate schema and write CSV. Used for SimBA boundary."""
    _check_schema(df, schema)
    fp.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(fp)


def init_empty_df(schema: SchemaDict) -> pl.DataFrame:
    """Create an empty Polars DataFrame with the given schema."""
    return pl.DataFrame(schema=schema)


def _check_schema(df: pl.DataFrame, expected: SchemaDict) -> None:
    """Validate that a DataFrame's schema matches the expected schema.

    Raises AssertionError with a descriptive message on mismatch.
    """
    actual = dict(df.schema)
    missing = [k for k in expected if k not in actual]
    extra = [k for k in actual if k not in expected]
    type_mismatches = [
        f"  {k}: expected {expected[k]}, got {actual[k]}"
        for k in expected
        if k in actual and actual[k] != expected[k]
    ]

    errors = []
    if missing:
        errors.append(f"Missing columns: {missing}")
    if extra:
        errors.append(f"Unexpected columns: {extra}")
    if type_mismatches:
        errors.append("Type mismatches:\n" + "\n".join(type_mismatches))

    if errors:
        msg = "Schema mismatch.\n" + "\n".join(errors)
        raise AssertionError(msg)
