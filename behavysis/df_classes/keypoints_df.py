"""
Utility functions.
"""

from enum import Enum

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from behavysis.utils.df_mixin import DFMixin


class FramesIN(Enum):
    FRAME = "frame"


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
        # Checking that the bodyparts are all valid (i.e. there are no missing bodyparts)
        bpts_not_exist = np.isin(bodyparts, df.columns.unique("bodyparts"), invert=True)
        if bpts_not_exist.any():
            bpts_ls_msg = "".join([f"\n    - {bpt}" for bpt in np.array(bodyparts)[bpts_not_exist]])
            raise ValueError(
                f"Some bodyparts in the config file are missing from the dataframe. They are:{bpts_ls_msg}"
            )

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
        # Filtering out any single and processing columns
        # Not incl. the `single` or `process`columns
        columns_filter = np.isin(
            df.columns.get_level_values(cls.CN.INDIVIDUALS.value),
            [IndivCols.PROCESSED.value, IndivCols.SINGLE.value],
            invert=True,
        )
        columns = df.columns[columns_filter]
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

    @classmethod
    def resolution_scale_df(cls, df: pd.DataFrame, width_x_scale: float, height_y_scale: float) -> pd.DataFrame:
        scaled_df = cls.basic_clean(df)
        idx = pd.IndexSlice
        # Scaling width coords
        scaled_df[CoordsCols.X.value] = scaled_df.loc[:, idx[:, :, :, CoordsCols.X.value]] * width_x_scale  # type: ignore
        # Scaling height coords
        scaled_df[CoordsCols.Y.value] = scaled_df.loc[:, idx[:, :, :, CoordsCols.Y.value]] * height_y_scale  # type: ignore
        return cls.basic_clean(scaled_df)


class KeyptsAnnotationsCN(Enum):
    ATTRIBUTES = "attributes"


class KeypointsAnnotationsDf(DFMixin):
    IN = FramesIN
    CN = KeyptsAnnotationsCN

    @classmethod
    def keypoint2annotationsdf(cls, keypoints_df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts a single keypoint to an attributes dataframe.

        Parameters
        ----------
        keypoint : pd.Series
            A single keypoint.

        Returns
        -------
        pd.DataFrame
            An attributes dataframe.
        """
        df = KeypointsDf.basic_clean(keypoints_df)
        # Filtering out IndivColumns.PROCESS.value columns
        columns = np.isin(
            keypoints_df.columns.get_level_values(KeypointsDf.CN.INDIVIDUALS.value),
            [IndivCols.PROCESSED.value],
            invert=True,
        )
        df = df.loc[:, columns]
        # Rounding and converting to correct dtypes - "x" and "y" values are ints
        xy_columns = df.columns[
            df.columns.get_level_values(KeypointsDf.CN.COORDS.value).isin([CoordsCols.X.value, CoordsCols.Y.value])
        ]
        df[xy_columns] = df[xy_columns].round(0).astype(int)
        # Changing the columns MultiIndex to a single-level index. For speedup
        df.columns = [f"{indiv}_{bpt}_{coord}" for scorer, indiv, bpt, coord in df.columns]
        return cls.basic_clean(df)

    @classmethod
    def get_indivs_bpts(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a tuple of the unique (indiv, bpt) pairs.

        Parameters
        ----------
        df : pd.DataFrame
            Keypoints pd.DataFrame.

        Returns
        -------
        tuple[list[str], list[str]]
            `(indivs_ls, bpts_ls)` tuples. It is recommended to unpack these vals.
        """
        df = cls.basic_clean(df)
        if df.columns.shape[0] == 0:
            return pd.DataFrame(columns=[KeypointsDf.CN.INDIVIDUALS.value, KeypointsDf.CN.BODYPARTS.value])
        indivs_bpts_df = df.columns.to_frame(index=False)[cls.CN.ATTRIBUTES.value].str.split("_", expand=True)
        indivs_bpts_df = indivs_bpts_df.iloc[:, :2]
        indivs_bpts_df.columns = [KeypointsDf.CN.INDIVIDUALS.value, KeypointsDf.CN.BODYPARTS.value]
        indivs_bpts_df = indivs_bpts_df.drop_duplicates().reset_index(drop=True)
        return indivs_bpts_df

    @classmethod
    def make_colours(cls, category_vals: pd.Series, cmap: str) -> np.ndarray:
        """
        Makes a list of colours for each bodypart instance.

        Parameters
        ----------
        measures_ls : list
            _description_
        cmap : str
            _description_

        Returns
        -------
        list
            _description_

        Example
        -------
        ```
        [1,2,4,2,3,1,5]
        --> [Red, Blue, Green, Blue, Yellow, Red, Purple]
        ```
        """
        # If vals is an empty list, return colours_ls as an empty list
        if len(category_vals) == 0:
            return np.array([])
        # Encoding colours as 0, 1, 2, ... for each unique value
        category_idx, _ = pd.factorize(category_vals)
        # Normalising between 0 and 1 (if only 1 unique value, it will be 0 div so setting values to 0)
        category_idx = np.nan_to_num(category_idx / category_idx.max())
        # Getting corresponding colour for each item in `vals` list and from cmap
        colours_arr = plt.cm.get_cmap(cmap)(category_idx)
        # Reassigning the order of the colours to be RGBA (not BGRA)
        colours_arr = colours_arr[:, [2, 1, 0, 3]]
        # Converting to (0, 255) range
        colours_arr = colours_arr * 255
        return colours_arr
