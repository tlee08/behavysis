"""Tests for generic feature extraction against real parquet data.

Validates that the generic Polars+NumPy+SciPy feature extraction produces
output for the classic 2-animal, 8-bodypart configuration and for
alternative bodypoint configs.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from behavysis.schemas import KEYPOINTS_SCHEMA, read_df

RESOURCES_DIR = Path(__file__).parent.parent.parent / "resources"
TEST_PARQUET = RESOURCES_DIR / "VIDEO_001_short.parquet"

INDIVS_2 = ["mouse1marked", "mouse2unmarked"]
BPTS_8 = [
    "LeftEar",
    "RightEar",
    "Nose",
    "BodyCentre",
    "LeftFlankMid",
    "RightFlankMid",
    "TailBase1",
    "TailTip4",
]


@pytest.fixture(scope="module")
def keypoints_df():
    """Load the real test parquet for feature extraction tests."""
    df = read_df(TEST_PARQUET, KEYPOINTS_SCHEMA)
    return df.filter(
        pl.col("individual").is_in(INDIVS_2),
        pl.col("bodypart").is_in(BPTS_8),
    )


@pytest.fixture(scope="module")
def features_df_2x8(keypoints_df):
    """Compute features for 2 individuals × 8 bodyparts."""
    from behavysis.funcs.extract_features import compute_features

    return compute_features(
        keypoints_df,
        individuals=INDIVS_2,
        bodyparts=BPTS_8,
        fps=30.0,
        px_per_mm=4.0,
    )


@pytest.fixture(scope="module")
def features_df_1x4(keypoints_df):
    """Compute features for 1 individual × 4 bodyparts."""
    indivs = ["mouse1marked"]
    bps = ["Nose", "BodyCentre", "TailBase1", "TailTip4"]
    df = keypoints_df.filter(
        pl.col("individual").is_in(indivs),
        pl.col("bodypart").is_in(bps),
    )
    from behavysis.funcs.extract_features import compute_features

    return compute_features(
        df, individuals=indivs, bodyparts=bps, fps=30.0, px_per_mm=4.0
    )


class TestGenericFeatures2x8:
    """Tests for generic features on 2 individuals × 8 bodyparts."""

    def test_output_shape(self, keypoints_df, features_df_2x8):
        n_frames = keypoints_df.select("frame").n_unique()
        assert features_df_2x8.height == n_frames

    def test_has_frame_column(self, features_df_2x8):
        assert features_df_2x8.columns[0] == "frame"
        assert features_df_2x8.schema["frame"] == pl.Int64

    def test_frame_monotonic(self, features_df_2x8):
        frames = features_df_2x8.select("frame").to_series()
        assert frames.is_sorted()

    def test_no_nulls(self, features_df_2x8):
        nulls = features_df_2x8.null_count()
        for col in features_df_2x8.columns:
            assert nulls.select(col).item() == 0, f"Null in column: {col}"

    def test_no_infinite(self, features_df_2x8):
        for col in features_df_2x8.columns:
            if col == "frame":
                continue
            series = features_df_2x8.select(col).to_series()
            assert not series.is_infinite().any(), f"Inf in column: {col}"

    def test_feature_count(self, features_df_2x8):
        """Should produce many features for 2-animal 8-bp config."""
        n_cols = len(features_df_2x8.columns)
        assert n_cols > 100, f"Expected 100+ columns, got {n_cols - 1}"

    def test_within_distance_features(self, features_df_2x8):
        col = f"{INDIVS_2[0]}_Nose_to_TailBase1_dist"
        assert col in features_df_2x8.columns, f"Missing: {col}"

    def test_cross_distance_features(self, features_df_2x8):
        col = f"{INDIVS_2[0]}_Nose_to_{INDIVS_2[1]}_Nose_dist"
        assert col in features_df_2x8.columns, f"Missing: {col}"

    def test_movement_features(self, features_df_2x8):
        col = f"{INDIVS_2[0]}_Nose_movement"
        assert col in features_df_2x8.columns, f"Missing: {col}"

    def test_hull_features(self, features_df_2x8):
        assert f"{INDIVS_2[0]}_hull_perimeter" in features_df_2x8.columns

    def test_cdist_features(self, features_df_2x8):
        assert f"{INDIVS_2[0]}_cdist_max" in features_df_2x8.columns
        assert f"{INDIVS_2[0]}_cdist_mean" in features_df_2x8.columns

    def test_total_movement_features(self, features_df_2x8):
        assert f"total_movement_{INDIVS_2[0]}" in features_df_2x8.columns
        assert "total_movement_all" in features_df_2x8.columns

    def test_probability_features(self, features_df_2x8):
        assert "sum_probabilities" in features_df_2x8.columns
        assert "low_prob_detections_0.1" in features_df_2x8.columns

    def test_rolling_features(self, features_df_2x8):
        rolling_cols = [
            c for c in features_df_2x8.columns if "_mean_" in c or "_median_" in c
        ]
        assert len(rolling_cols) > 10, (
            f"Expected 10+ rolling cols, got {len(rolling_cols)}"
        )

    def test_deviation_features(self, features_df_2x8):
        dev_cols = [c for c in features_df_2x8.columns if "_deviation" in c]
        assert len(dev_cols) > 3, f"Expected 3+ deviation cols, got {len(dev_cols)}"

    def test_percentile_rank_features(self, features_df_2x8):
        pr_cols = [c for c in features_df_2x8.columns if "percentile_rank" in c]
        assert len(pr_cols) > 1, f"Expected 1+ percentile rank cols, got {len(pr_cols)}"

    def test_features_are_numeric(self, features_df_2x8):
        for col in features_df_2x8.columns:
            if col == "frame":
                continue
            assert features_df_2x8.schema[col] == pl.Float64, (
                f"Column {col} has type {features_df_2x8.schema[col]}, expected Float64"
            )


class TestGenericFeaturesValues:
    """Tests validating that feature values are in reasonable ranges."""

    def test_distances_non_negative(self, features_df_2x8):
        dist_cols = [
            c
            for c in features_df_2x8.columns
            if "_to_" in c and "_dist" in c and "_deviation" not in c
        ]
        eps = 1e-12
        for col in dist_cols:
            vals = features_df_2x8.select(col).to_series()
            assert vals.min() >= -eps, f"Negative values in {col}: min={vals.min()}"

    def test_movement_non_negative(self, features_df_2x8):
        move_cols = [
            c
            for c in features_df_2x8.columns
            if c.endswith("_movement") and "_deviation" not in c
        ]
        eps = 1e-12
        for col in move_cols:
            vals = features_df_2x8.select(col).to_series()
            assert vals.min() >= -eps, f"Negative values in {col}: min={vals.min()}"

    def test_low_prob_detections_in_range(self, features_df_2x8):
        lp_cols = [
            c for c in features_df_2x8.columns if c.startswith("low_prob_detections")
        ]
        n_bp = 16  # 2 individuals × 8 bodyparts
        for col in lp_cols:
            vals = features_df_2x8.select(col).to_series()
            assert (vals >= 0).all(), f"Negative count in {col}"
            assert (vals <= n_bp).all(), f"Count > {n_bp} in {col}"

    def test_percentile_ranks_in_range(self, features_df_2x8):
        pr_cols = [c for c in features_df_2x8.columns if "percentile_rank" in c]
        for col in pr_cols:
            vals = features_df_2x8.select(col).to_series()
            assert (vals >= 0).all(), f"Negative percentile in {col}"
            assert (vals <= 1).all(), f"Percentile > 1 in {col}"


class TestSingleAnimalFeatures:
    """Tests for generic features on 1 individual × 4 bodyparts."""

    def test_no_cross_individual_features(self, features_df_1x4):
        """Single animal should have no cross-individual distance features."""
        cross_cols = [c for c in features_df_1x4.columns if "mouse2" in c.lower()]
        assert len(cross_cols) == 0

    def test_output_shape(self, features_df_1x4):
        assert features_df_1x4.height == 1801

    def test_no_nulls(self, features_df_1x4):
        nulls = features_df_1x4.null_count()
        for col in features_df_1x4.columns:
            assert nulls.select(col).item() == 0, f"Null in column: {col}"

    def test_no_infinite(self, features_df_1x4):
        for col in features_df_1x4.columns:
            if col == "frame":
                continue
            series = features_df_1x4.select(col).to_series()
            assert not series.is_infinite().any(), f"Inf in column: {col}"

    def test_has_basic_features(self, features_df_1x4):
        assert "mouse1marked_hull_perimeter" in features_df_1x4.columns
        assert "total_movement_mouse1marked" in features_df_1x4.columns
        assert "sum_probabilities" in features_df_1x4.columns
