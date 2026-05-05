"""Keypoints DataFrame for pose estimation data."""

from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from behavysis.utils.df_mixin import DFMixin


class FramesIN(Enum):
    FRAME = "frame"


class CoordsCols(Enum):
    X = "x"
    Y = "y"
    LIKELIHOOD = "likelihood"


class IndivCols(Enum):
    SINGLE = "single"
    PROCESSED = "processed"


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
    def _validate(cls, df: pd.DataFrame) -> None:
        super()._validate(df)

    @classmethod
    def check_bpts_exist(cls, df: pd.DataFrame, bodyparts: list) -> None:
        missing = [b for b in bodyparts if b not in df.columns.unique("bodyparts")]
        if missing:
            available = df.columns.unique("bodyparts").to_list()[:5]
            suffix = "..." if len(df.columns.unique("bodyparts")) > 5 else ""
            msg = (
                f"Bodyparts not found in keypoints data: {missing}\n"
                f"  Available: {', '.join(available)}{suffix}\n"
                f"  Check your config file's bodyparts list."
            )
            raise ValueError(msg)

    @classmethod
    def get_indivs_bpts(cls, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        """Get individuals and bodyparts (excluding 'single' and 'processed')."""
        filter_mask = ~df.columns.get_level_values(cls.CN.INDIVIDUALS.value).isin(
            [IndivCols.PROCESSED.value, IndivCols.SINGLE.value]
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
            [cls.CN.INDIVIDUALS.value, cls.CN.BODYPARTS.value, cls.CN.COORDS.value]
        ]
        df.columns = pd.MultiIndex.from_frame(columns)
        return df

    @classmethod
    def resolution_scale(
        cls, df: pd.DataFrame, width_x_scale: float, height_y_scale: float
    ) -> pd.DataFrame:
        """Scale x and y coordinates."""
        df = cls._clean_and_validate(df)
        idx = pd.IndexSlice
        df.loc[:, idx[:, :, :, CoordsCols.X.value]] *= width_x_scale
        df.loc[:, idx[:, :, :, CoordsCols.Y.value]] *= height_y_scale
        return cls._clean_and_validate(df)


class KeyptsAnnotationsCN(Enum):
    ATTRIBUTES = "attributes"


class KeypointsAnnotationsDf(DFMixin):
    IN = FramesIN
    CN = KeyptsAnnotationsCN

    @classmethod
    def keypoint2annotationsdf(cls, keypoints_df: pd.DataFrame) -> pd.DataFrame:
        """Convert keypoints to flat column format: 'indiv_bpt_coord'."""
        df = KeypointsDf._clean_and_validate(keypoints_df)
        filter_mask = ~df.columns.get_level_values(KeypointsDf.CN.INDIVIDUALS.value).isin(
            [IndivCols.PROCESSED.value]
        )
        df = df.loc[:, filter_mask]
        xy_cols = df.columns[
            df.columns.get_level_values(KeypointsDf.CN.COORDS.value).isin(
                [CoordsCols.X.value, CoordsCols.Y.value]
            )
        ]
        df[xy_cols] = df[xy_cols].round(0).astype(int)
        df.columns = [f"{indiv}_{bpt}_{coord}" for _, indiv, bpt, coord in df.columns]
        return cls._clean_and_validate(df)

    @classmethod
    def get_indivs_bpts(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Get unique (indiv, bpt) pairs from flat columns."""
        df = cls._clean_and_validate(df)
        if df.columns.shape[0] == 0:
            return pd.DataFrame(
                columns=[
                    KeypointsDf.CN.INDIVIDUALS.value,
                    KeypointsDf.CN.BODYPARTS.value,
                ]
            )
        parts = df.columns.to_frame(index=False)[cls.CN.ATTRIBUTES.value].str.split(
            "_", expand=True
        )
        parts = parts.iloc[:, :2]
        parts.columns = [KeypointsDf.CN.INDIVIDUALS.value, KeypointsDf.CN.BODYPARTS.value]
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
