"""DataFrame mixin providing unified read/write with schema validation."""

from pathlib import Path

import pandas as pd

from behavysis.constants import DF_IO_FORMAT


class DFMixin:
    """Mixin for DFs with schema validation via IN (index) and CN (columns) enums.

    Validates schema on read and write. Sorts index and columns on both.
    Default IO format is Parquet.
    """

    is_nullable = True
    index_names: tuple | None = None
    column_names: tuple | None = None
    io_format: str = DF_IO_FORMAT

    @classmethod
    def read(cls, fp: Path, fmt: str | None = None) -> pd.DataFrame:
        """Read dataframe from file, validate schema, and sort."""
        fmt = fmt or cls.io_format
        index_levels = len(cls.index_names or (None,))
        column_levels = len(cls.column_names or (None,))

        if fmt == "csv":
            df = pd.read_csv(
                fp,
                index_col=list(range(index_levels)) if index_levels > 0 else None,
                header=list(range(column_levels)) if column_levels > 0 else None,
            )
        elif fmt == "h5":
            df = pd.DataFrame(pd.read_hdf(fp, mode="r"))
        elif fmt == "feather":
            df = pd.read_feather(fp)
        elif fmt == "parquet":
            df = pd.read_parquet(fp)
        else:
            msg = f"Unsupported format: {fmt}. Use: csv, h5, feather, parquet."
            raise ValueError(msg)

        return cls.clean_and_validate(df)

    @classmethod
    def write(cls, df: pd.DataFrame, fp: Path, fmt: str | None = None) -> None:
        """Validate schema, sort, and write dataframe to file."""
        df = cls.clean_and_validate(df)
        fp.parent.mkdir(parents=True, exist_ok=True)

        fmt = fmt or cls.io_format
        if fmt == "csv":
            df.to_csv(fp)
        elif fmt == "h5":
            df.to_hdf(fp, key="data", mode="w")
        elif fmt == "feather":
            df.to_feather(fp)
        elif fmt == "parquet":
            df.to_parquet(fp)
        else:
            msg = f"Unsupported format: {fmt}. Use: csv, h5, feather, parquet."
            raise ValueError(msg)

    @classmethod
    def init_df(cls, index: pd.Series | pd.Index) -> pd.DataFrame:
        """Create empty dataframe with schema-defined index structure."""
        return pd.DataFrame(
            index=pd.MultiIndex.from_frame(index.to_frame(), names=cls.index_names),
            columns=pd.MultiIndex.from_tuples((), names=cls.column_names),
        )

    @classmethod
    def clean_and_validate(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Set index/column names, sort, and validate schema."""
        if cls.index_names:
            # Check index
            if df.index.nlevels != len(cls.index_names):
                msg = (
                    f"Index has {df.index.nlevels} levels, "
                    "expected {len(cls.index_names)}.\n"
                    f"  Expected index: {cls.index_names}\n"
                    "  Tip: "
                    "Check that your data file has the correct column structure."
                )
                raise AssertionError(msg)
            df.index = df.index.set_names(cls.index_names)

        if cls.column_names:
            if df.columns.nlevels != len(cls.column_names):
                msg = (
                    f"Columns have {df.columns.nlevels} levels, "
                    f"expected {len(cls.column_names)}.\n"
                    f"  Expected columns: {cls.column_names}\n"
                    "  Tip: "
                    "Check that your data file has the correct header structure."
                )
                raise AssertionError(msg)
            df.columns = df.columns.set_names(cls.column_names)

        df = df.sort_index()
        df = df.sort_index(axis=1)

        cls._validate(df)
        return df

    @classmethod
    def _validate(cls, df: pd.DataFrame) -> None:
        """Override to add custom validation. Called on read and write."""
        assert isinstance(df, pd.DataFrame), "Must be a pandas DataFrame"

        if not cls.is_nullable and df.isna().to_numpy().any():
            msg = "DataFrame contains NaN values but is_nullable=False."
            raise AssertionError(msg)
