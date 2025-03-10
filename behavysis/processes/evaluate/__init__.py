"""
Functions have the following format:

Parameters
----------
vid_fp : str
    the GPU's number so computation is done on this GPU.
dlc_fp : str
    _description_
behavs_fp : str
    _description_
dst_dir : str
    _description_
configs_fp : str
    _description_
overwrite : bool
    Whether to overwrite the output file (if it exists).

Returns
-------
str
    Description of the function's outcome.

Notes
-----
Given the `dst_dir`, we save the files to `dst_dir/<func_name>/<exp_name>.<ext>`
"""

import os

import pandas as pd
import seaborn as sns

from behavysis.df_classes.behav_df import BehavScoredDf
from behavysis.df_classes.keypoints_df import CoordsCols, KeypointsDf
from behavysis.pydantic_models.experiment_configs import ExperimentConfigs
from behavysis.utils.diagnostics_utils import file_exists_msg
from behavysis.utils.io_utils import get_name


class Evaluate:
    """__summary__"""

    ###############################################################################################
    #               MAKE KEYPOINTS PLOTS
    ###############################################################################################

    @staticmethod
    def keypoints_plot(
        vid_fp: str,
        dlc_fp: str,
        behavs_fp: str,
        dst_dir: str,
        configs_fp: str,
        overwrite: bool,
    ) -> str:
        """
        Make keypoints evaluation plot of likelihood of each bodypart through time.
        """
        name = get_name(dlc_fp)
        dst_dir = os.path.join(dst_dir, Evaluate.keypoints_plot.__name__)
        dst_fp = os.path.join(dst_dir, f"{name}.png")
        os.makedirs(dst_dir, exist_ok=True)

        # Getting necessary config parameters
        configs = ExperimentConfigs.read_json(configs_fp)
        configs_filt = configs.user.evaluate.keypoints_plot
        bpts = configs.get_ref(configs_filt.bodyparts)
        fps = configs.auto.formatted_vid.fps

        # Read the file
        df = KeypointsDf.clean_headings(KeypointsDf.read(dlc_fp))
        # Checking the bodyparts specified in the configs exist in the dataframe
        KeypointsDf.check_bpts_exist(df, bpts)
        # Making data-long ways
        idx = pd.IndexSlice
        df = (
            df.loc[:, idx[:, bpts]]
            .stack([KeypointsDf.CN.INDIVIDUALS.value, KeypointsDf.CN.BODYPARTS.value])
            .reset_index()
        )
        # Adding the timestamp column
        df["timestamp"] = df[KeypointsDf.IN.FRAME.value] / fps
        # Making plot
        g = sns.FacetGrid(
            df,
            row=KeypointsDf.CN.INDIVIDUALS.value,
            height=5,
            aspect=10,
        )
        g.map_dataframe(
            sns.lineplot,
            x="timestamp",
            y=CoordsCols.LIKELIHOOD.value,
            hue=KeypointsDf.CN.BODYPARTS.value,
            alpha=0.4,
        )
        g.add_legend()
        # Saving plot
        g.savefig(dst_fp)
        g.figure.clf()
        return ""

    ###############################################################################################
    # MAKE BEHAVIOUR PLOTS
    ###############################################################################################

    @staticmethod
    def behav_plot(
        vid_fp: str,
        dlc_fp: str,
        behavs_fp: str,
        dst_dir: str,
        configs_fp: str,
        overwrite: bool,
    ) -> str:
        """
        Make behaviour evaluation plot of the predicted and actual behaviours through time.
        """
        name = get_name(behavs_fp)
        dst_dir = os.path.join(dst_dir, Evaluate.behav_plot.__name__)
        dst_fp = os.path.join(dst_dir, f"{name}.png")
        os.makedirs(dst_dir, exist_ok=True)
        # If overwrite is False, checking if we should skip processing
        if not overwrite and os.path.exists(dst_fp):
            return file_exists_msg()

        # Getting necessary config parameters
        configs = ExperimentConfigs.read_json(configs_fp)
        # configs_filt = configs.user.evaluate.behav_plot
        fps = float(configs.auto.formatted_vid.fps)

        # Read the file
        df = BehavScoredDf.read(behavs_fp)
        # Making data-long ways
        df = (
            df.stack([BehavScoredDf.CN.BEHAVS.value, BehavScoredDf.CN.OUTCOMES.value])
            .reset_index()
            .rename(columns={0: "value"})
        )
        # Adding the timestamp column
        df["timestamp"] = df[BehavScoredDf.IN.FRAME.value] / fps
        # Making plot
        g = sns.FacetGrid(
            df,
            row=BehavScoredDf.CN.BEHAVS.value,
            height=5,
            aspect=10,
        )
        g.map_dataframe(
            sns.lineplot,
            x="timestamp",
            y="value",
            hue=BehavScoredDf.CN.OUTCOMES.value,
            alpha=0.4,
        )
        g.add_legend()
        # Saving plot
        g.savefig(dst_fp)
        g.figure.clf()
        return ""
