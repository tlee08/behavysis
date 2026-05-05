"""Unit tests for DFMixin class."""

from enum import Enum
from pathlib import Path

import pandas as pd
import pytest

from behavysis.utils.df_mixin import DFMixin, _enum_values


class TestEnumValues:
    """Tests for the _enum_values helper function."""

    def test_none_input(self) -> None:
        """None input should return None."""
        assert _enum_values(None) is None

    def test_enum_input(self) -> None:
        """Enum input should return tuple of values."""
        class Color(Enum):
            RED = "red"
            GREEN = "green"
            BLUE = "blue"

        result = _enum_values(Color)
        assert result == ("red", "green", "blue")
        assert isinstance(result, tuple)

    def test_int_enum_input(self) -> None:
        """IntEnum should work correctly."""
        class Numbers(Enum):
            ONE = 1
            TWO = 2
            THREE = 3

        assert _enum_values(Numbers) == (1, 2, 3)


class TestDF(DFMixin):
    """Test DataFrame class with single-level index and columns."""

    class IN(Enum):
        FRAME = "frame"

    class CN(Enum):
        VALUE = "value"

    NULLABLE = True


class TestDFMultiIndex(DFMixin):
    """Test DataFrame class with multi-level index and columns."""

    class IN(Enum):
        FRAME = "frame"
        SUBFRAME = "subframe"

    class CN(Enum):
        COORD = "coord"
        BODYPART = "bodypart"

    NULLABLE = True


class TestDFNonNullable(DFMixin):
    """Test DataFrame class that doesn't allow NaN values."""

    class IN(Enum):
        FRAME = "frame"

    class CN(Enum):
        VALUE = "value"

    NULLABLE = False


class TestDFNoSchema(DFMixin):
    """Test DataFrame class without schema (IN and CN are None)."""

    IN = None
    CN = None


class TestDFMixinInit:
    """Tests for DFMixin.init_df method."""

    def test_init_df_creates_empty_dataframe(self) -> None:
        """init_df should create an empty DataFrame with correct schema."""
        index = pd.Index([0, 1, 2, 3, 4], name="frame")
        df = TestDF.init_df(index)

        assert df.empty
        # init_df creates a MultiIndex even for single-level
        assert df.index.names == ["frame"]
        assert df.columns.names == ["value"]

    def test_init_df_multiindex(self) -> None:
        """init_df should work with multi-level indices."""
        index = pd.MultiIndex.from_tuples(
            [(0, 0), (0, 1), (1, 0)],
            names=["frame", "subframe"],
        )
        df = TestDFMultiIndex.init_df(index)

        assert df.empty
        assert df.index.names == ["frame", "subframe"]
        assert df.columns.names == ["coord", "bodypart"]


class TestDFMixinValidation:
    """Tests for DFMixin schema validation."""

    def test_validate_correct_index_levels(self) -> None:
        """DataFrame with correct index levels should pass validation."""
        df = pd.DataFrame(
            {"value": [1, 2, 3]},
            index=pd.Index([0, 1, 2], name="frame"),
        )
        # Should not raise
        TestDF._validate(df)

    def test_validate_wrong_index_levels(self) -> None:
        """DataFrame with wrong index levels should raise AssertionError."""
        df = pd.DataFrame(
            {"value": [1, 2, 3]},
            index=pd.MultiIndex.from_tuples([(0, 0), (1, 0), (2, 0)]),
        )
        with pytest.raises(AssertionError, match="Index has"):
            TestDF._clean_and_validate(df)

    def test_validate_wrong_column_levels(self) -> None:
        """DataFrame with wrong column levels should raise AssertionError."""
        df = pd.DataFrame(
            [[1, 2, 3]],
            columns=pd.MultiIndex.from_tuples([("a", "x"), ("a", "y"), ("b", "z")]),
            index=pd.Index([0], name="frame"),
        )
        with pytest.raises(AssertionError, match="Columns have"):
            TestDF._clean_and_validate(df)

    def test_validate_non_nullable_with_nan(self) -> None:
        """Non-nullable DataFrame with NaN should raise AssertionError."""
        df = pd.DataFrame(
            {"value": [1.0, float("nan"), 3.0]},
            index=pd.Index([0, 1, 2], name="frame"),
        )
        with pytest.raises(AssertionError, match="NaN values"):
            TestDFNonNullable._validate(df)

    def test_validate_non_nullable_without_nan(self) -> None:
        """Non-nullable DataFrame without NaN should pass validation."""
        df = pd.DataFrame(
            {"value": [1.0, 2.0, 3.0]},
            index=pd.Index([0, 1, 2], name="frame"),
        )
        # Should not raise
        TestDFNonNullable._validate(df)

    def test_validate_nullable_with_nan(self) -> None:
        """Nullable DataFrame with NaN should pass validation."""
        df = pd.DataFrame(
            {"value": [1.0, float("nan"), 3.0]},
            index=pd.Index([0, 1, 2], name="frame"),
        )
        # Should not raise
        TestDF._validate(df)

    def test_validate_not_dataframe(self) -> None:
        """Non-DataFrame should raise AssertionError."""
        with pytest.raises(AssertionError, match="Must be a pandas DataFrame"):
            TestDF._validate([1, 2, 3])  # type: ignore[arg-type]


class TestDFMixinCleanAndValidate:
    """Tests for DFMixin._clean_and_validate method."""

    def test_sets_index_names(self) -> None:
        """Should set index names from IN enum."""
        df = pd.DataFrame(
            {"value": [1, 2, 3]},
            index=pd.Index([0, 1, 2]),
        )
        result = TestDF._clean_and_validate(df)
        assert result.index.name == "frame"

    def test_sets_column_names(self) -> None:
        """Should set column names from CN enum."""
        df = pd.DataFrame(
            [[1], [2], [3]],
            columns=pd.Index(["value"]),
            index=pd.Index([0, 1, 2]),
        )
        result = TestDF._clean_and_validate(df)
        assert result.columns.names == ["value"]

    def test_sorts_index(self) -> None:
        """Should sort the index."""
        df = pd.DataFrame(
            {"value": [1, 2, 3]},
            index=pd.Index([2, 0, 1]),
        )
        result = TestDF._clean_and_validate(df)
        assert list(result.index) == [0, 1, 2]

    def test_sorts_columns(self) -> None:
        """Should sort the columns (single column - no effect)."""
        df = pd.DataFrame(
            {"value": [1, 2, 3]},
            index=pd.Index([0, 1, 2]),
        )
        result = TestDF._clean_and_validate(df)
        assert list(result.columns) == ["value"]


class TestDFMixinReadWrite:
    """Tests for DFMixin read/write operations."""

    def test_write_and_read_parquet(self, temp_dir: Path) -> None:
        """Should write and read parquet files correctly."""
        df = pd.DataFrame(
            {"value": [1.0, 2.0, 3.0]},
            index=pd.Index([0, 1, 2], name="frame"),
        )
        fp = temp_dir / "test.parquet"

        TestDF.write(df, fp)
        result = TestDF.read(fp)

        pd.testing.assert_frame_equal(df, result)

    def test_write_and_read_csv(self, temp_dir: Path) -> None:
        """Should write and read CSV files correctly."""
        df = pd.DataFrame(
            {"value": [1.0, 2.0, 3.0]},
            index=pd.Index([0, 1, 2], name="frame"),
        )
        fp = temp_dir / "test.csv"

        TestDF.write(df, fp, fmt="csv")
        result = TestDF.read(fp, fmt="csv")

        pd.testing.assert_frame_equal(df, result)

    def test_write_creates_parent_directories(self, temp_dir: Path) -> None:
        """Write should create parent directories if they don't exist."""
        df = pd.DataFrame(
            {"value": [1.0]},
            index=pd.Index([0], name="frame"),
        )
        fp = temp_dir / "subdir" / "another" / "test.parquet"

        assert not fp.parent.exists()
        TestDF.write(df, fp)
        assert fp.exists()

    def test_read_unsupported_format(self, temp_dir: Path) -> None:
        """Reading unsupported format should raise ValueError."""
        fp = temp_dir / "test.xyz"
        fp.touch()

        with pytest.raises(ValueError, match="Unsupported format"):
            TestDF.read(fp, fmt="xyz")

    def test_write_unsupported_format(self, temp_dir: Path) -> None:
        """Writing unsupported format should raise ValueError."""
        df = pd.DataFrame(
            {"value": [1.0]},
            index=pd.Index([0], name="frame"),
        )
        fp = temp_dir / "test.xyz"

        with pytest.raises(ValueError, match="Unsupported format"):
            TestDF.write(df, fp, fmt="xyz")

    def test_convenience_read_csv(self, temp_dir: Path) -> None:
        """read_csv convenience method should work."""
        df = pd.DataFrame(
            {"value": [1.0, 2.0]},
            index=pd.Index([0, 1], name="frame"),
        )
        fp = temp_dir / "test.csv"

        TestDF.write_csv(df, fp)
        result = TestDF.read_csv(fp)

        pd.testing.assert_frame_equal(df, result)


class TestDFMixinNoSchema:
    """Tests for DFMixin without schema (IN/CN are None)."""

    def test_read_no_schema(self, temp_dir: Path) -> None:
        """Reading without schema should work without validation."""
        df = pd.DataFrame(
            {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]},
        )
        fp = temp_dir / "test.parquet"

        df.to_parquet(fp)
        result = TestDFNoSchema.read(fp)

        pd.testing.assert_frame_equal(df, result)

    def test_write_no_schema(self, temp_dir: Path) -> None:
        """Writing without schema should work without validation."""
        df = pd.DataFrame(
            {"a": [1, 2, 3], "b": [4, 5, 6]},
        )
        fp = temp_dir / "test.parquet"

        TestDFNoSchema.write(df, fp)
        result = pd.read_parquet(fp)

        pd.testing.assert_frame_equal(df, result)


class TestDFMixinMultiLevel:
    """Tests for DFMixin with multi-level indices."""

    def test_read_write_multiindex(self, temp_dir: Path) -> None:
        """Should handle multi-level indices correctly."""
        index = pd.MultiIndex.from_tuples(
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            names=["frame", "subframe"],
        )
        columns = pd.MultiIndex.from_product(
            [["x", "y"], ["nose", "tail"]],
            names=["coord", "bodypart"],
        )
        df = pd.DataFrame(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            index=index,
            columns=columns,
        )
        fp = temp_dir / "test.parquet"

        TestDFMultiIndex.write(df, fp)
        result = TestDFMultiIndex.read(fp)

        pd.testing.assert_frame_equal(df, result)

    def test_csv_multiindex_roundtrip(self, temp_dir: Path) -> None:
        """CSV roundtrip with multi-level indices should preserve structure."""
        index = pd.MultiIndex.from_tuples(
            [(0, 0), (0, 1), (1, 0)],
            names=["frame", "subframe"],
        )
        columns = pd.MultiIndex.from_product(
            [["x", "y"], ["nose"]],
            names=["coord", "bodypart"],
        )
        df = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=index,
            columns=columns,
        )
        fp = temp_dir / "test.csv"

        TestDFMultiIndex.write(df, fp, fmt="csv")
        result = TestDFMultiIndex.read(fp, fmt="csv")

        pd.testing.assert_frame_equal(df, result)
