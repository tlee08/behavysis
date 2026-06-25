"""_summary_."""

import contextlib

import cv2
import numpy as np
import pandas as pd

from behavysis.constants import LIKELIHOOD, X, Y
from behavysis.df_classes import KeypointsAnnotationsDf, KeypointsDf
from behavysis.models import ExperimentConfig


class KeypointsModel:
    """_summary_."""

    raw_dlc_df: pd.DataFrame
    keypoints_df: pd.DataFrame
    indivs_bpts_df: pd.DataFrame
    colours: np.ndarray
    pcutoff: float
    radius: int
    colour_level: str
    cmap: str

    def __init__(self) -> None:
        self.load_empty()

    def load_from_df(
        self, keypoints_df: pd.DataFrame, config: ExperimentConfig
    ) -> None:
        """Load in the raw DLC dataframe.

        Use the given dlc_fp and config.
        """
        # Config
        config_filt = config.user.evaluate_vid
        self.colour_level = config.get_ref(config_filt.colour_level)
        self.pcutoff = config.get_ref(config_filt.pcutoff)
        self.radius = config.get_ref(config_filt.radius)
        self.cmap = config.get_ref(config_filt.cmap)
        # Keypoints dataframe
        self.keypoints_df = KeypointsAnnotationsDf.keypoint2annotationsdf(keypoints_df)
        self.indivs_bpts_df = KeypointsAnnotationsDf.get_indivs_bpts(self.keypoints_df)
        self.colours = KeypointsAnnotationsDf.make_colours(
            self.indivs_bpts_df[self.colour_level], self.cmap
        )

    def load(self, fp: str, config: ExperimentConfig) -> None:
        df = KeypointsDf.init_df(pd.Series())
        with contextlib.suppress(FileNotFoundError):
            df = KeypointsDf.read(fp)
        self.load_from_df(df, config)

    def load_empty(self) -> None:
        """Load an empty dataset into the instance.

        An empty dataset is used as placeholder.
        """
        self.load_from_df(KeypointsDf.init_df(pd.Series()), ExperimentConfig())

    def annot_keypoints(self, frame: np.ndarray, frame_num: int) -> np.ndarray:
        """Adding the keypoints (given in frame number) to the frame and returning the annotated frame.

        Parameters
        ----------
        frame : np.ndarray
            cv2 frame array.
        frame_num : int
            index (i.e. frame number) in DLC dataframe.

        Returns:
        -------
        np.ndarray
            cv2 frame array.
        """
        # Getting frame_num row and asserting the idx exists
        try:
            row = self.keypoints_df.loc[frame_num]
        except KeyError:
            return frame
        # For each indiv-bpt, if likelihood is above pcutoff, draw the keypoint
        for i, indiv, bpt in self.indivs_bpts_df.itertuples(name=None):
            if row[f"{indiv}_{bpt}_{LIKELIHOOD}"] >= self.pcutoff:
                cv2.circle(
                    img=frame,
                    center=(
                        int(row[f"{indiv}_{bpt}_{X}"]),
                        int(row[f"{indiv}_{bpt}_{Y}"]),
                    ),
                    radius=self.radius,
                    color=self.colours[i],
                    thickness=-1,
                )
        return frame
