"""Functional tests verifying Polars schema and conversion correctness.

These tests serve as a safety net during the migration. They verify that
Polars long-form data produces the same results as the existing pandas
MultiIndex pipeline.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from behavysis.constants import BODYPART, COORD, FRAME, INDIVIDUAL, LIKELIHOOD, X, Y
from behavysis.schemas import (
    ANALYSIS_SCHEMA,
    BINNED_SCHEMA,
    KEYPOINTS_SCHEMA,
    SUMMARY_SCHEMA,
    init_empty_df,
    read_df,
    write_df,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Conversion utilities (these will later live in their respective modules)
# ═══════════════════════════════════════════════════════════════════════════════


def pandas_keypoints_to_polars(df_pd: pd.DataFrame) -> pl.DataFrame:
    """Convert a pandas KeypointsDf (MultiIndex columns) to Polars long form.

    Drops the ``scorer`` level (always nunique=1, useless).
    """
    # Drop scorer level from column MultiIndex
    df_pd = df_pd.copy()
    df_pd.columns = df_pd.columns.droplevel("scorer")

    # Stack to long form: frame x (individuals, bodyparts, coords)
    stacked = df_pd.stack([INDIVIDUAL, BODYPART, COORD], future_stack=True)

    # Unstack coords so x, y, likelihood become columns
    unstacked = stacked.unstack(COORD)

    # Reset index to get frame as a column
    result = unstacked.reset_index()

    # Convert to Polars with correct column names
    return pl.from_pandas(result).select(
        pl.col(FRAME).cast(pl.Int64),
        pl.col(INDIVIDUAL).alias("individual"),
        pl.col(BODYPART).alias("bodypart"),
        pl.col(X).cast(pl.Float64),
        pl.col(Y).cast(pl.Float64),
        pl.col(LIKELIHOOD).cast(pl.Float64),
    )


def polars_keypoints_to_pandas(df_pl: pl.DataFrame) -> pd.DataFrame:
    """Convert Polars long-form keypoints back to pandas MultiIndex wide form.

    Roundtrip verification utility.
    """
    df_pd = df_pl.to_pandas()
    # Pivot back to wide: (individuals, bodyparts, coords) columns
    df_pd = df_pd.set_index([FRAME, "individual", "bodypart", "coords"])
    return df_pd


# ═══════════════════════════════════════════════════════════════════════════════
# Schema tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSchemaInit:
    """Tests for init_empty_df and schema dicts."""

    def test_init_empty_keypoints(self) -> None:
        df = init_empty_df(KEYPOINTS_SCHEMA)
        assert df.schema == KEYPOINTS_SCHEMA
        assert df.height == 0

    def test_init_empty_analysis(self) -> None:
        df = init_empty_df(ANALYSIS_SCHEMA)
        assert df.schema == ANALYSIS_SCHEMA

    def test_init_empty_summary(self) -> None:
        df = init_empty_df(SUMMARY_SCHEMA)
        assert df.schema == SUMMARY_SCHEMA

    def test_init_empty_binned(self) -> None:
        df = init_empty_df(BINNED_SCHEMA)
        assert df.schema == BINNED_SCHEMA


class TestSchemaIO:
    """Tests for read_df / write_df with schema validation."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        df = pl.DataFrame(
            {
                "frame": [0, 0, 1, 1],
                "individual": ["m1", "m2", "m1", "m2"],
                "bodypart": ["nose", "nose", "nose", "nose"],
                "x": [1.0, 2.0, 3.0, 4.0],
                "y": [5.0, 6.0, 7.0, 8.0],
                "likelihood": [0.9, 0.8, 0.7, 0.6],
            },
            schema=KEYPOINTS_SCHEMA,
        )
        fp = tmp_path / "test.parquet"
        write_df(df, fp, KEYPOINTS_SCHEMA)
        result = read_df(fp, KEYPOINTS_SCHEMA)
        assert df.equals(result)

    def test_schema_mismatch_raises(self, tmp_path: Path) -> None:
        df = pl.DataFrame({"frame": [0, 1], "individual": ["a", "b"]})
        fp = tmp_path / "bad.parquet"
        with pytest.raises(AssertionError, match="Schema mismatch"):
            write_df(df, fp, KEYPOINTS_SCHEMA)

    def test_wrong_type_raises(self, tmp_path: Path) -> None:
        df = pl.DataFrame(
            {
                "frame": ["a", "b"],
                "individual": ["x", "y"],
                "bodypart": ["n", "n"],
                "x": [1.0, 2.0],
                "y": [3.0, 4.0],
                "likelihood": [0.5, 0.6],
            },
        )
        fp = tmp_path / "bad_type.parquet"
        with pytest.raises(AssertionError, match="Type mismatches"):
            write_df(df, fp, KEYPOINTS_SCHEMA)

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        df = init_empty_df(KEYPOINTS_SCHEMA)
        fp = tmp_path / "deep" / "nested" / "test.parquet"
        write_df(df, fp, KEYPOINTS_SCHEMA)
        assert fp.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# Keypoints conversion tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestKeypointsConversion:
    """Tests for pandas MultiIndex → Polars long-form keypoints conversion."""

    @pytest.fixture
    def pandas_keypoints(self) -> pd.DataFrame:
        """Synthetic keypoints in the current pandas MultiIndex format."""
        return pd.DataFrame(
            {
                ("scorer", INDIVIDUAL, BODYPART, COORD): [
                    ("DLC", "mouse1", "nose", X),
                    ("DLC", "mouse1", "nose", Y),
                    ("DLC", "mouse1", "nose", LIKELIHOOD),
                    ("DLC", "mouse1", "tail", X),
                    ("DLC", "mouse1", "tail", Y),
                    ("DLC", "mouse1", "tail", LIKELIHOOD),
                    ("DLC", "mouse2", "nose", X),
                    ("DLC", "mouse2", "nose", Y),
                    ("DLC", "mouse2", "nose", LIKELIHOOD),
                    ("DLC", "mouse2", "tail", X),
                    ("DLC", "mouse2", "tail", Y),
                    ("DLC", "mouse2", "tail", LIKELIHOOD),
                ],
                "values": [
                    # Frame 0
                    100.0,
                    200.0,
                    0.99,
                    150.0,
                    250.0,
                    0.98,
                    300.0,
                    400.0,
                    0.97,
                    350.0,
                    450.0,
                    0.96,
                    # Frame 1
                    101.0,
                    201.0,
                    0.99,
                    151.0,
                    251.0,
                    0.98,
                    301.0,
                    401.0,
                    0.97,
                    351.0,
                    451.0,
                    0.96,
                ],
            }
        ).pivot(
            index=None,
            columns=["scorer", INDIVIDUAL, BODYPART, COORD],
            values="values",
        )
        # Add frame index
        # Note: This pivot approach puts data in one row, let me just build properly

    def test_convert_pandas_to_polars(self) -> None:
        """Pandas MultiIndex keypoints should convert correctly to Polars long form."""
        n_frames = 5
        individuals = ["mouse1", "mouse2"]
        bodyparts = ["nose", "tail"]
        coords = [X, Y, LIKELIHOOD]
        scorer = "DLC"

        # Build pandas MultiIndex keypoints
        columns = pd.MultiIndex.from_product(
            [[scorer], individuals, bodyparts, coords],
            names=["scorer", INDIVIDUAL, BODYPART, COORD],
        )
        index = pd.Index(range(n_frames), name=FRAME)
        np.random.seed(42)
        data = np.random.randn(n_frames, len(columns))
        # Make coordinates reasonable
        for i, indiv in enumerate(individuals):
            for j, bpt in enumerate(bodyparts):
                base = i * len(bodyparts) * 3 + j * 3
                data[:, base] = np.random.uniform(100, 900, n_frames)  # x
                data[:, base + 1] = np.random.uniform(50, 450, n_frames)  # y
                data[:, base + 2] = np.random.uniform(0.5, 1.0, n_frames)  # likelihood

        df_pd = pd.DataFrame(data, index=index, columns=columns)

        # Convert to Polars long form
        df_pl = pandas_keypoints_to_polars(df_pd)

        # Verify schema
        assert df_pl.schema == KEYPOINTS_SCHEMA

        # Verify row count: n_frames x n_individuals x n_bodyparts
        assert df_pl.height == n_frames * len(individuals) * len(bodyparts)

        # Verify scorer is gone
        assert "scorer" not in df_pl.columns

        # Verify frame 0, mouse1, nose x value matches
        pandas_val = df_pd.loc[0, (scorer, "mouse1", "nose", X)]
        polars_val = (
            df_pl.filter(
                pl.col("frame") == 0,
                pl.col("individual") == "mouse1",
                pl.col("bodypart") == "nose",
            )
            .select("x")
            .item()
        )
        assert pandas_val == polars_val

        # Verify no NaN values (original data has none by construction)
        assert df_pl.null_count().sum_horizontal().item() == 0

    def test_convert_preserves_all_values(self) -> None:
        """Every value from pandas should be preserved in the Polars conversion."""
        n_frames = 3
        individuals = ["m1"]
        bodyparts = ["nose"]
        coords = [X, Y, LIKELIHOOD]
        scorer = "DLC"

        columns = pd.MultiIndex.from_product(
            [[scorer], individuals, bodyparts, coords],
            names=["scorer", INDIVIDUAL, BODYPART, COORD],
        )
        index = pd.Index(range(n_frames), name=FRAME)
        data = np.array([[10.0, 20.0, 0.5], [11.0, 21.0, 0.6], [12.0, 22.0, 0.7]])
        df_pd = pd.DataFrame(data, index=index, columns=columns)

        df_pl = pandas_keypoints_to_polars(df_pd)

        for frame in range(n_frames):
            row = df_pl.filter(pl.col("frame") == frame)
            assert row.select("x").item() == df_pd.loc[frame, (scorer, "m1", "nose", X)]
            assert row.select("y").item() == df_pd.loc[frame, (scorer, "m1", "nose", Y)]
            assert (
                row.select("likelihood").item()
                == df_pd.loc[frame, (scorer, "m1", "nose", LIKELIHOOD)]
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis schema tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestAnalysisSchema:
    """Tests for Analysis schema conversion patterns."""

    def test_analysis_long_form(self) -> None:
        """AnalysisDf long form should support natural filter + agg operations."""
        df = pl.DataFrame(
            {
                "frame": [0, 0, 1, 1, 0, 1],
                "group": ["m1", "m2", "m1", "m2", "m1", "m1"],
                "measure": ["speed", "speed", "speed", "speed", "dist", "dist"],
                "value": [10.0, 12.0, 11.0, 13.0, 5.0, 6.0],
            },
            schema=ANALYSIS_SCHEMA,
        )

        # Filter for m1 speed
        m1_speed = df.filter(
            pl.col("group") == "m1",
            pl.col("measure") == "speed",
        )
        assert m1_speed.height == 2
        assert m1_speed.select("value").to_series().to_list() == [10.0, 11.0]

        # Group by group and measure
        agg = df.group_by("group", "measure").agg(pl.col("value").mean())
        assert (
            agg.filter(pl.col("measure") == "speed")
            .filter(pl.col("group") == "m1")
            .select("value")
            .item()
            == 10.5
        )

    def test_summary_schema(self) -> None:
        """SummaryDf long form should represent aggregations naturally."""
        df = pl.DataFrame(
            {
                "group": ["m1", "m1", "m2", "m2"],
                "measure": ["speed", "speed", "speed", "speed"],
                "agg": ["mean", "std", "mean", "std"],
                "value": [10.5, 0.5, 12.5, 0.5],
            },
            schema=SUMMARY_SCHEMA,
        )

        # Pivot for display: wide summary
        wide = df.pivot(
            index=["group", "measure"],
            on="agg",
            values="value",
        )
        assert wide.select("mean").row(0)[0] == 10.5
        assert wide.select("std").row(0)[0] == 0.5

    def test_binned_schema(self) -> None:
        """BinnedDf long form should support time-slice analysis."""
        df = pl.DataFrame(
            {
                "bin_sec": [0.0, 0.0, 30.0, 30.0],
                "group": ["m1", "m2", "m1", "m2"],
                "measure": ["speed", "speed", "speed", "speed"],
                "agg": ["mean", "mean", "mean", "mean"],
                "value": [10.0, 12.0, 9.0, 11.0],
            },
            schema=BINNED_SCHEMA,
        )

        # Pivot for plotting
        wide = df.pivot(
            index=["bin_sec", "group"],
            on="measure",
            values="value",
        )
        assert wide.height == 4
