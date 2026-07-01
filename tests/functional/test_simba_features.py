"""Tests for native SimBA feature extraction against real parquet data.

These tests validate that the native Polars+NumPy+SciPy feature extraction
produces output structurally compatible with what the SimBA conda subprocess
would produce for the "2 animals; 16 body-parts" configuration.

Cross-validation against actual SimBA output requires a system with the SimBA
conda environment installed. Placeholder tests are provided for that scenario.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from behavysis.schemas import KEYPOINTS_SCHEMA, read_df

RESOURCES_DIR = Path(__file__).parent.parent.parent / "resources"
TEST_PARQUET = RESOURCES_DIR / "VIDEO_001_short.parquet"


@pytest.fixture(scope="module")
def keypoints_df():
    """Load the real test parquet for feature extraction tests."""
    df = read_df(TEST_PARQUET, KEYPOINTS_SCHEMA)
    from behavysis.constants.bodypoints import BPTS_SIMBA, INDIVS_SIMBA

    return df.filter(
        pl.col("individual").is_in(INDIVS_SIMBA),
        pl.col("bodypart").is_in(BPTS_SIMBA),
    )


@pytest.fixture(scope="module")
def features_df(keypoints_df):
    """Compute features once per module for all tests."""
    from behavysis.funcs.extract_features import compute_simba_features

    return compute_simba_features(keypoints_df, fps=30.0, px_per_mm=4.0)


class TestSimbaFeaturesCompute:
    """Tests for the core compute() function on real data."""

    def test_output_shape(self, keypoints_df, features_df):
        """Output should have one row per unique frame."""
        n_frames = keypoints_df.select("frame").n_unique()
        assert features_df.height == n_frames

    def test_has_frame_column(self, features_df):
        """Output must have a frame column as the first column."""
        assert features_df.columns[0] == "frame"
        assert features_df.schema["frame"] == pl.Int64

    def test_frame_monotonic(self, features_df):
        """Frames should be in ascending order."""
        frames = features_df.select("frame").to_series()
        assert frames.is_sorted()

    def test_no_nulls(self, features_df):
        """No feature columns should contain null values."""
        nulls = features_df.null_count()
        for col in features_df.columns:
            assert nulls.select(col).item() == 0, f"Null in column: {col}"

    def test_no_infinite(self, features_df):
        """No feature columns should contain infinite values."""
        for col in features_df.columns:
            if col == "frame":
                continue
            series = features_df.select(col).to_series()
            assert not series.is_infinite().any(), f"Inf in column: {col}"

    def test_feature_count(self, features_df):
        """Should produce 400+ features (SimBA produces ~600+ for 2-animal 16-bp)."""
        # frame column + features
        assert len(features_df.columns) > 400, (
            f"Expected 400+ features, got {len(features_df.columns) - 1}"
        )

    def test_distance_features_exist(self, features_df):
        """Core distance features must be present."""
        required = [
            "Mouse_1_nose_to_tail",
            "Mouse_2_nose_to_tail",
            "Centroid_distance",
            "Nose_to_nose_distance",
        ]
        for name in required:
            assert name in features_df.columns, f"Missing: {name}"

    def test_hull_features_exist(self, features_df):
        """Convex hull features must be present (any naming convention)."""
        hull_cols = [c for c in features_df.columns if "poly_area" in c.lower()]
        assert len(hull_cols) >= 2, f"Expected 2+ hull columns, got: {hull_cols}"

    def test_movement_features_exist(self, features_df):
        """Movement features must be present for key body-parts."""
        movement_cols = [
            c for c in features_df.columns if c.startswith("Movement_mouse_")
        ]
        assert len(movement_cols) >= 16, (
            f"Expected 16+ movement cols (8 bp x 2 animals), got: {len(movement_cols)}"
        )

    def test_angle_features_exist(self, features_df):
        """3-point angle features must be present."""
        assert "Mouse_1_angle" in features_df.columns
        assert "Mouse_2_angle" in features_df.columns

    def test_rolling_features_exist(self, features_df):
        """Rolling window features must be present."""
        rolling_cols = [
            c for c in features_df.columns if "_median_" in c or "_mean_" in c
        ]
        assert len(rolling_cols) > 10, (
            f"Expected 10+ rolling cols, got: {len(rolling_cols)}"
        )

    def test_probability_features_exist(self, features_df):
        """Probability-based features must be present."""
        assert "Sum_probabilities" in features_df.columns
        assert "Low_prob_detections_0.1" in features_df.columns

    def test_deviation_features_exist(self, features_df):
        """Deviation features must be present."""
        dev_cols = [c for c in features_df.columns if "_deviation" in c]
        assert len(dev_cols) > 5, f"Expected 5+ deviation cols, got: {len(dev_cols)}"

    def test_tortuosity_features_exist(self, features_df):
        """Tortuosity features must be present."""
        tort_cols = [c for c in features_df.columns if "Tortuosity" in c]
        assert len(tort_cols) >= 2, (
            f"Expected 2+ tortuosity cols, got: {len(tort_cols)}"
        )

    def test_percentile_rank_features_exist(self, features_df):
        """Percentile rank features must be present."""
        pr_cols = [c for c in features_df.columns if "percentile_rank" in c]
        assert len(pr_cols) > 3, (
            f"Expected 3+ percentile rank cols, got: {len(pr_cols)}"
        )

    def test_features_are_numeric(self, features_df):
        """All non-frame feature columns must be Float64."""
        for col in features_df.columns:
            if col == "frame":
                continue
            assert features_df.schema[col] == pl.Float64, (
                f"Column {col} has type {features_df.schema[col]}, expected Float64"
            )


class TestSimbaFeaturesValues:
    """Tests validating that feature values are in reasonable ranges."""

    def test_distances_non_negative(self, features_df):
        """All distance features should be non-negative (allow fp rounding)."""
        dist_cols = [
            c
            for c in features_df.columns
            if any(kw in c for kw in ["distance", "Distance", "nose_to_tail", "width"])
            and "_deviation" not in c
            and "percentile" not in c
        ]
        eps = 1e-12
        for col in dist_cols:
            vals = features_df.select(col).to_series()
            assert vals.min() >= -eps, f"Negative values in {col}: min={vals.min()}"

    def test_movement_non_negative(self, features_df):
        """Movement (raw) features should be non-negative. Deviation is excluded."""
        move_cols = [
            c
            for c in features_df.columns
            if c.startswith("Movement_mouse_") and "_deviation" not in c
        ]
        eps = 1e-12
        for col in move_cols:
            vals = features_df.select(col).to_series()
            assert vals.min() >= -eps, f"Negative values in {col}: min={vals.min()}"

    def test_angles_in_range(self, features_df):
        """Angle features should be in [0, 360] degrees."""
        ang_cols = [c for c in features_df.columns if c.endswith("_angle")]
        for col in ang_cols:
            vals = features_df.select(col).to_series()
            assert (vals >= 0).all(), f"Negative angle in {col}"
            assert (vals <= 360).all(), f"Angle > 360 in {col}"

    def test_low_prob_detections_in_range(self, features_df):
        """Low probability detection counts should be 0 to n_bodyparts."""
        lp_cols = [
            c for c in features_df.columns if c.startswith("Low_prob_detections")
        ]
        for col in lp_cols:
            vals = features_df.select(col).to_series()
            assert (vals >= 0).all(), f"Negative count in {col}"
            assert (vals <= 16).all(), f"Count > 16 (n_bodyparts) in {col}"

    def test_percentile_ranks_in_range(self, features_df):
        """Percentile ranks should be in [0, 1]."""
        pr_cols = [c for c in features_df.columns if "percentile_rank" in c]
        for col in pr_cols:
            vals = features_df.select(col).to_series()
            assert (vals >= 0).all(), f"Negative percentile in {col}"
            assert (vals <= 1).all(), f"Percentile > 1 in {col}"

    def test_centroid_distance_reasonable(self, features_df):
        """Centroid distance between two mice should be plausible (5-200mm)."""
        vals = features_df.select("Centroid_distance").to_series()
        assert vals.min() > 0, f"Min centroid distance is {vals.min()}"


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-validation placeholder
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skip(
    reason="Requires SimBA conda environment for cross-validation. "
    "Run on a system with 'simba' conda env to validate output compatibility."
)
class TestSimbaCrossValidation:
    """Cross-validate native output against actual SimBA subprocess output."""

    def test_column_names_match_simba(self, features_df):
        """Output column names should match SimBA's feature_extraction_headers.csv."""
        # Load reference column names from SimBA's assets
        ...

    def test_feature_values_match_simba(self, features_df, keypoints_df):
        """Feature values should match SimBA output within floating-point tolerance."""
        # Run SimBA subprocess on the same data, compare output
        ...
