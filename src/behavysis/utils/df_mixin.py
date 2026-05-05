"""DataFrame mixin providing unified read/write with schema validation.

Usage:
    class KeypointsDf(DFMixin):
        IN = FramesIN  # Enum for index names
        CN = KeypointsCN  # Enum for column names
        NULLABLE = False  # Set to True to allow NaN values

    df = KeypointsDf.read(path)
    KeypointsDf.write(df, path)
"""

from enum import EnumType
from pathlib import Path

import pandas as pd

from behavysis.constants import DF_IO_FORMAT


def _enum_values(e: type | None) -> tuple | None:
    """Extract tuple of values from an enum, or None if not an enum."""
    if e is None:
        return None
    return tuple(i.value for i in e)


class DFMixin:
    """Mixin for DataFrames with schema validation via IN (index) and CN (columns) enums.

    Validates schema on read and write. Sorts index and columns on both.
    Default IO format is Parquet.
    """

    NULLABLE = True
    IN: EnumType | None = None
    CN: EnumType | None = None
    IO: str = DF_IO_FORMAT

    @classmethod
    def read(cls, fp: Path, fmt: str | None = None) -> pd.DataFrame:
        """Read dataframe from file, validate schema, and sort."""
        fmt = fmt or cls.IO
        index_levels = len(_enum_values(cls.IN) or (None,))
        column_levels = len(_enum_values(cls.CN) or (None,))

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

        return cls._clean_and_validate(df)

    @classmethod
    def write(cls, df: pd.DataFrame, fp: Path, fmt: str | None = None) -> None:
        """Validate schema, sort, and write dataframe to file."""
        df = cls._clean_and_validate(df)
        fp.parent.mkdir(parents=True, exist_ok=True)

        fmt = fmt or cls.IO
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
        in_names = _enum_values(cls.IN)
        cn_names = _enum_values(cls.CN)
        return pd.DataFrame(
            index=pd.MultiIndex.from_frame(index.to_frame(), names=in_names),
            columns=pd.MultiIndex.from_tuples((), names=cn_names),
        )

    @classmethod
    def _clean_and_validate(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Set index/column names, sort, and validate schema."""
        in_names = _enum_values(cls.IN)
        cn_names = _enum_values(cls.CN)

        if in_names:
            if df.index.nlevels != len(in_names):
                msg = (
                    f"Index has {df.index.nlevels} levels, expected {len(in_names)}.\n"
                    f"  Expected index: {in_names}\n"
                    f"  Tip: Check that your data file has the correct column structure."
                )
                raise AssertionError(msg)
            df.index = df.index.set_names(in_names)

        if cn_names:
            if df.columns.nlevels != len(cn_names):
                msg = (
                    f"Columns have {df.columns.nlevels} levels, expected {len(cn_names)}.\n"
                    f"  Expected columns: {cn_names}\n"
                    f"  Tip: Check that your data file has the correct header structure."
                )
                raise AssertionError(msg)
            df.columns = df.columns.set_names(cn_names)

        df = df.sort_index()
        df = df.sort_index(axis=1)

        cls._validate(df)
        return df

    @classmethod
    def _validate(cls, df: pd.DataFrame) -> None:
        """Override to add custom validation. Called on read and write."""
        assert isinstance(df, pd.DataFrame), "Must be a pandas DataFrame"

        if not cls.NULLABLE:
            if df.isna().to_numpy().any():
                msg = "DataFrame contains NaN values but NULLABLE=False."
                raise AssertionError(msg)

    # Convenience methods for specific formats
    @classmethod
    def read_csv(cls, fp: Path) -> pd.DataFrame:
        return cls.read(fp, fmt="csv")

    @classmethod
    def write_csv(cls, df: pd.DataFrame, fp: Path) -> None:
        cls.write(df, fp, fmt="csv")
