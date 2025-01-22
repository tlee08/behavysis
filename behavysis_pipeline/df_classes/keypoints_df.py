"""
Utility functions.
"""

from enum import Enum

import numpy as np
import pandas as pd

from behavysis_pipeline.df_classes.df_mixin import DFMixin, FramesIN


class CoordsCols(Enum):
    X = "x"
    Y = "y"
    LIKELIHOOD = "likelihood"


class IndivCols(Enum):
    SINGLE = "single"
    PROCESSED = "processed"  # TODO: remove this


class KeypointsCN(Enum):
    SCORER = "scorer"
    INDIVIDUALS = "individuals"
    BODYPARTS = "bodyparts"
    COORDS = "coords"


class KeypointsDf(DFMixin):
    """
    Mixin for behaviour DF
    (generated from maDLC keypoint detection)
    functions.
    """

    NULLABLE = False
    IN = FramesIN
    CN = KeypointsCN

    @classmethod
    def check_bpts_exist(cls, df: pd.DataFrame, bodyparts: list) -> None:
        """
        _summary_

        Parameters
        ----------
        df : pd.DataFrame
            _description_
        bodyparts : list
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        # Checking that the bodyparts are all valid:
        bodyparts_exist = np.isin(bodyparts, df.columns.unique("bodyparts"))
        if not bodyparts_exist.all():
            msg = "Some bodyparts in the config file are missing from the csv file. They are:"
            for bp in np.array(bodyparts)[~bodyparts_exist]:
                msg += f"\n    - {bp}"
            raise ValueError(msg)

    @classmethod
    def get_indivs_bpts(cls, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        """
        Returns a tuple of the individuals (only animals, not "single" or "processed"), and tuple of
        the multi-animal bodyparts.

        Parameters
        ----------
        df : pd.DataFrame
            Keypoints pd.DataFrame.

        Returns
        -------
        tuple[list[str], list[str]]
            `(indivs_ls, bpts_ls)` tuples. It is recommended to unpack these vals.
        """
        # Getting column MultiIndex
        columns = df.columns
        # Filtering out any single and processing columns
        # Not incl. the `single` or `process`columns
        columns_filt = np.isin(
            df.columns.get_level_values(cls.CN.INDIVIDUALS.value),
            [IndivCols.PROCESSED.value, IndivCols.SINGLE.value],
            invert=True,
        )
        columns = df.columns[columns_filt]
        # Getting individuals list
        indivs = columns.unique("individuals").to_list()
        # Getting bodyparts list
        bpts = columns.unique("bodyparts").to_list()
        return indivs, bpts

    @classmethod
    def clean_headings(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops the "scorer" level in the column
        header of the dataframe. This makes subsequent processing easier.

        Parameters
        ----------
        df : pd.DataFrame
            Keypoints pd.DataFrame.

        Returns
        -------
        pd.DataFrame
            Keypoints pd.DataFrame.

        Notes
        -----
        Does not return a dataframe with the same KeypointsDf schema
        (it is missing the SCORER column level).
        """
        df = df.copy()
        # Keeping only the "individuals", "bodyparts", and "coords" levels
        # (i.e. dropping "scorer" level)
        columns = df.columns.to_frame(index=False)
        columns = columns[[cls.CN.INDIVIDUALS.value, cls.CN.BODYPARTS.value, cls.CN.COORDS.value]]
        df.columns = pd.MultiIndex.from_frame(columns)
        return df
