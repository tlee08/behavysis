"""DataFrame mixin class providing unified read/write operations with schema validation."""

from enum import EnumType
from pathlib import Path

import pandas as pd

from behavysis.constants import DF_IO_FORMAT
from behavysis.utils.misc_utils import enum2tuple


class DFMixin:
    """Mixin providing read/write operations for DataFrames with schema validation.

    Subclasses define schema via IN (index names) and CN (column names) enums.
    All I/O operations validate schema and ensure consistent formatting.
    """

    NULLABLE = True
    IN = None
    CN = None
    IO = DF_IO_FORMAT

    ###############################################################################################
    # DF Read Functions
    ###############################################################################################

    @classmethod
    def read(cls, fp: Path, fmt: str | None = None) -> pd.DataFrame:
        """Read dataframe from file.

        Parameters
        ----------
        fp : Path
            File path to read from.
        fmt : str | None
            File format. If None, uses cls.IO. Supports: csv, h5, feather, parquet.

        Returns:
        -------
        pd.DataFrame
            Loaded dataframe with validated schema.

        Raises:
        ------
        AssertionError
            If format is not supported.
        """
        fmt = fmt or cls.IO
        if fmt == "csv":
            df = pd.read_csv(
                fp,
                index_col=list(range(len(enum2tuple(cls.IN) if cls.IN else (None,)))),
                header=list(range(len(enum2tuple(cls.CN) if cls.CN else (None,)))),
            )
        elif fmt == "h5":
            df = pd.DataFrame(pd.read_hdf(fp, mode="r"))
        elif fmt == "feather":
            df = pd.read_feather(fp)
        elif fmt == "parquet":
            df = pd.read_parquet(fp)
        else:
            msg = f"File type, {fmt}, not supported. Supported formats: csv, h5, feather, parquet."
            raise AssertionError(msg)
        return cls.basic_clean(df)

    @classmethod
    def read_csv(cls, fp: Path) -> pd.DataFrame:
        """Read dataframe from CSV file."""
        return cls.read(fp, fmt="csv")

    @classmethod
    def read_h5(cls, fp: Path) -> pd.DataFrame:
        """Read dataframe from HDF5 file."""
        return cls.read(fp, fmt="h5")

    @classmethod
    def read_feather(cls, fp: Path) -> pd.DataFrame:
        """Read dataframe from Feather file."""
        return cls.read(fp, fmt="feather")

    @classmethod
    def read_parquet(cls, fp: Path) -> pd.DataFrame:
        """Read dataframe from Parquet file."""
        return cls.read(fp, fmt="parquet")

    ###############################################################################################
    # DF Write Functions
    ###############################################################################################

    @classmethod
    def write(cls, df: pd.DataFrame, fp: Path, fmt: str | None = None) -> None:
        """Write dataframe to file.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to write.
        fp : Path
            File path to write to.
        fmt : str | None
            File format. If None, uses cls.IO. Supports: csv, h5, feather, parquet.

        Raises:
        ------
        AssertionError
            If format is not supported.
        """
        df = cls.basic_clean(df)
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
            msg = f"File type, {fmt}, not supported. Supported formats: csv, h5, feather, parquet."
            raise AssertionError(msg)

    @classmethod
    def write_csv(cls, df: pd.DataFrame, fp: Path) -> None:
        """Write dataframe to CSV file."""
        cls.write(df, fp, fmt="csv")

    @classmethod
    def write_h5(cls, df: pd.DataFrame, fp: Path) -> None:
        """Write dataframe to HDF5 file."""
        cls.write(df, fp, fmt="h5")

    @classmethod
    def write_feather(cls, df: pd.DataFrame, fp: Path) -> None:
        """Write dataframe to Feather file."""
        cls.write(df, fp, fmt="feather")

    @classmethod
    def write_parquet(cls, df: pd.DataFrame, fp: Path) -> None:
        """Write dataframe to Parquet file."""
        cls.write(df, fp, fmt="parquet")

    ###############################################################################################
    # DF init functions
    ###############################################################################################

    @classmethod
    def init_df(cls, index: pd.Series | pd.Index) -> pd.DataFrame:
        """Initialize empty dataframe with schema-defined index and column structure.

        Parameters
        ----------
        index : pd.Series | pd.Index
            Index values for the new dataframe.

        Returns:
        -------
        pd.DataFrame
            Empty dataframe with proper MultiIndex structure.
        """
        IN = enum2tuple(cls.IN) if cls.IN else None
        CN = enum2tuple(cls.CN) if cls.CN else None
        return pd.DataFrame(
            index=pd.MultiIndex.from_frame(index.to_frame(), names=IN),
            columns=pd.MultiIndex.from_tuples((), names=CN),
        )

    ###############################################################################################
    # DF Validation Functions
    ###############################################################################################

    @classmethod
    def basic_clean(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate dataframe structure.

        Sets index/column names from schema and sorts both axes.
        Validates structure with check_df.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to clean.

        Returns:
        -------
        pd.DataFrame
            Cleaned dataframe with validated structure.
        """
        if cls.IN:
            assert df.index.nlevels == len(enum2tuple(cls.IN)), (
                "Different number of column levels than expected.\n"
                f"Expected columns are {enum2tuple(cls.IN)} but got {df.index.nlevels} levels.\n"
                f"{df}"
            )
            df.index = df.index.set_names(enum2tuple(cls.IN))
        if cls.CN:
            assert df.columns.nlevels == len(enum2tuple(cls.CN)), (
                "Different number of column levels than expected.\n"
                f"Expected columns are {enum2tuple(cls.CN)} but got {df.columns.nlevels} levels.\n"
                f"{df}"
            )
            df.columns = df.columns.set_names(enum2tuple(cls.CN))
        df = df.sort_index()
        df = df.sort_index(axis=1)
        cls.check_df(df)
        return df

    @classmethod
    def check_df(cls, df: pd.DataFrame) -> None:
        """Validate dataframe structure matches schema.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to validate.

        Raises:
        ------
        AssertionError
            If dataframe doesn't match expected schema.
        """
        # Checking that df is a DataFrame
        assert isinstance(df, pd.DataFrame), "The dataframe must be a pandas DataFrame."
        # Checking there are no null values
        if not cls.NULLABLE:
            assert not df.isna().to_numpy().any(), (
                "The dataframe contains null values but it should not."
            )
        # Checking that the index levels are correct
        if cls.IN:
            cls._check_levels(df.index, cls.IN, "index")
        # Checking that the column levels are correct
        if cls.CN:
            cls._check_levels(df.columns, cls.CN, "columns")

    @classmethod
    def _check_levels(
        cls,
        obj: pd.Index | pd.MultiIndex,
        levels: EnumType | tuple[str] | str,
        name: str,
    ) -> None:
        """Validate that index/column levels match expected names.

        Parameters
        ----------
        obj : pd.Index | pd.MultiIndex
            Index or columns to validate.
        levels : EnumType | tuple[str] | str
            Expected level names.
        name : str
            Name for error messages ("index" or "columns").
        """
        # Converting `levels` to a tuple
        if isinstance(levels, EnumType):  # If Enum
            levels = enum2tuple(levels)
        elif isinstance(levels, str):  # If str
            levels = (levels,)
        assert obj.names == levels, (
            f"The {name} levels are incorrect. Expected {levels} but got {obj.names}."
        )
