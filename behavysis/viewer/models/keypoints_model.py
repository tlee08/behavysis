"""
_summary_
"""

import cv2
import numpy as np
import pandas as pd

from behavysis.df_classes.keypoints_df import CoordsCols, KeypointsAnnotationsDf, KeypointsDf
from behavysis.pydantic_models.configs import ExperimentConfigs


class KeypointsModel:
    """
    _summary_
    """

    raw_dlc_df: pd.DataFrame
    keypoints_df: pd.DataFrame
    indivs_bpts_df: pd.DataFrame
    colours: np.ndarray
    pcutoff: float
    radius: int
    colour_level: str
    cmap: str

    def __init__(self):
        self.load_from_df(KeypointsDf.init_df(pd.Series()), ExperimentConfigs())

    def load_from_df(self, keypoints_df: pd.DataFrame, configs: ExperimentConfigs):
        """
        load in the raw DLC dataframe and set the configurations, from
        the given dlc_fp and configs.
        """
        # Configs
        configs_filt = configs.user.evaluate_vid
        self.colour_level = configs.get_ref(configs_filt.colour_level)
        self.pcutoff = configs.get_ref(configs_filt.pcutoff)
        self.radius = configs.get_ref(configs_filt.radius)
        self.cmap = configs.get_ref(configs_filt.cmap)
        # Keypoints dataframe
        self.keypoints_df = KeypointsAnnotationsDf.keypoint2annotationsdf(keypoints_df)
        self.indivs_bpts_df = KeypointsAnnotationsDf.get_indivs_bpts(self.keypoints_df)
        self.colours = KeypointsAnnotationsDf.make_colours(self.indivs_bpts_df[self.colour_level], self.cmap)

    def load(self, fp: str, configs: ExperimentConfigs):
        self.load_from_df(KeypointsDf.read(fp), configs)

    def annot_keypoints(self, frame: np.ndarray, frame_num: int) -> np.ndarray:
        """
        Adding the keypoints (given in frame number) to the frame and returning the annotated frame.

        Parameters
        ----------
        frame : np.ndarray
            cv2 frame array.
        frame_num : int
            index (i.e. frame number) in DLC dataframe.

        Returns
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
            if row[f"{indiv}_{bpt}_{CoordsCols.LIKELIHOOD.value}"] >= self.pcutoff:
                cv2.circle(
                    img=frame,
                    center=(
                        int(row[f"{indiv}_{bpt}_{CoordsCols.X.value}"]),
                        int(row[f"{indiv}_{bpt}_{CoordsCols.Y.value}"]),
                    ),
                    radius=self.radius,
                    color=self.colours[i],
                    thickness=-1,
                )
        return frame
