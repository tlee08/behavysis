"""Keypoints DataFrame for pose estimation data."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from behavysis.constants import (
    BODYPARTS,
    COORDS,
    FRAME,
    INDIVIDUALS,
    PROCESSED,
    SCORER,
    SINGLE,
    X,
    Y,
)

from .df_mixin import DFMixin


class KeypointsDf(DFMixin):
    """KeypointsDf."""

    is_nullable = False
    index_names = (FRAME,)
    column_names = (SCORER, INDIVIDUALS, BODYPARTS, COORDS)

    @classmethod
    def _validate(cls, df: pd.DataFrame) -> None:
        super()._validate(df)

    @classmethod
    def check_bpts_exist(cls, df: pd.DataFrame, bodyparts: list) -> None:
        """Check if bodyparts exists."""
        missing = [b for b in bodyparts if b not in df.columns.unique("bodyparts")]
        if missing:
            max_list = 5
            available = df.columns.unique("bodyparts").to_list()[:max_list]
            suffix = "..." if len(df.columns.unique("bodyparts")) > max_list else ""
            msg = (
                f"Bodyparts not found in keypoints data: {missing}\n"
                f"  Available: {', '.join(available)}{suffix}\n"
                f"  Check your config file's bodyparts list."
            )
            raise ValueError(msg)

    @classmethod
    def get_indivs_bpts(cls, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        """Get individuals and bodyparts (excluding 'single' and 'processed')."""
        filter_mask = ~df.columns.get_level_values(INDIVIDUALS).isin(
            [PROCESSED, SINGLE]
        )
        columns = df.columns[filter_mask]
        indivs = columns.unique("individuals").to_list()
        bpts = columns.unique("bodyparts").to_list()
        return indivs, bpts

    @classmethod
    def clean_headings(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Drop the 'scorer' column level."""
        df = df.copy()
        columns = df.columns.to_frame(index=False)
        columns = columns[
            [
                INDIVIDUALS,
                BODYPARTS,
                COORDS,
            ]
        ]
        df.columns = pd.MultiIndex.from_frame(columns)
        return df

    @classmethod
    def resolution_scale(
        cls, df: pd.DataFrame, width_x_scale: float, height_y_scale: float
    ) -> pd.DataFrame:
        """Scale x and y coordinates."""
        df = cls.clean_and_validate(df)
        idx = pd.IndexSlice
        df.loc[:, idx[:, :, :, X]] *= width_x_scale
        df.loc[:, idx[:, :, :, Y]] *= height_y_scale
        return cls.clean_and_validate(df)


class KeypointsAnnotationsDf(DFMixin):
    """KeypointsAnnotationsDf."""

    index_names = (FRAME,)
    column_names = ("attributes",)

    @classmethod
    def keypoint2annotationsdf(cls, keypoints_df: pd.DataFrame) -> pd.DataFrame:
        """Convert keypoints to flat column format: 'indiv_bpt_coord'."""
        df = KeypointsDf.clean_and_validate(keypoints_df)
        filter_mask = ~df.columns.get_level_values(INDIVIDUALS).isin([PROCESSED])
        df = df.loc[:, filter_mask]
        xy_cols = df.columns[df.columns.get_level_values(COORDS).isin([X, Y])]
        df[xy_cols] = df[xy_cols].round(0).astype(int)
        df.columns = [f"{indiv}_{bpt}_{coord}" for _, indiv, bpt, coord in df.columns]
        return cls.clean_and_validate(df)

    @classmethod
    def get_indivs_bpts(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Get unique (indiv, bpt) pairs from flat columns."""
        df = cls.clean_and_validate(df)
        if df.columns.shape[0] == 0:
            return pd.DataFrame(columns=pd.Index([INDIVIDUALS, BODYPARTS]))
        parts = df.columns.to_frame(index=False)["attributes"].str.split(
            "_", expand=True
        )
        parts = parts.iloc[:, :2]
        parts.columns = [INDIVIDUALS, BODYPARTS]
        return parts.drop_duplicates().reset_index(drop=True)

    @classmethod
    def make_colours(cls, category_vals: pd.Series, cmap: str) -> np.ndarray:
        """Map categorical values to RGBA colors."""
        if len(category_vals) == 0:
            return np.array([])
        _, unique = pd.factorize(category_vals)
        idx = np.nan_to_num(np.arange(len(category_vals)) / max(len(unique) - 1, 1))
        colours = plt.cm.get_cmap(cmap)(idx)[:, [2, 1, 0, 3]] * 255
        return colours
