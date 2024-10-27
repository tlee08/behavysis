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
out_dir : str
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
Given the `out_dir`, we save the files to `out_dir/<func_name>/<exp_name>.<ext>`
"""

import os

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from behavysis_core.df_classes.analyse_df import AnalyseDf
from behavysis_core.df_classes.behav_df import BehavDf
from behavysis_core.df_classes.keypoints_df import Coords, IndivColumns, KeypointsDf
from behavysis_core.mixins.diagnostics_mixin import DiagnosticsMixin
from behavysis_core.mixins.io_mixin import IOMixin
from behavysis_core.pydantic_models.experiment_configs import ExperimentConfigs
from tqdm import trange

from behavysis_pipeline.processes.evaluate.evaluate_vid_funcs import (
    VidFuncRunner,
)


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
        out_dir: str,
        configs_fp: str,
        overwrite: bool,
    ) -> str:
        """
        Make keypoints evaluation plot of likelihood of each bodypart through time.
        """
        outcome = ""
        name = IOMixin.get_name(dlc_fp)
        out_dir = os.path.join(out_dir, Evaluate.keypoints_plot.__name__)
        out_fp = os.path.join(out_dir, f"{name}.png")
        os.makedirs(out_dir, exist_ok=True)
        # If overwrite is False, checking if we should skip processing
        if not overwrite and os.path.exists(out_fp):
            return DiagnosticsMixin.warning_msg()

        # Getting necessary config parameters
        configs = ExperimentConfigs.read_json(configs_fp)
        configs_filt = configs.user.evaluate.keypoints_plot
        bpts = configs.get_ref(configs_filt.bodyparts)
        fps = configs.auto.formatted_vid.fps

        # Read the file
        df = KeypointsDf.clean_headings(KeypointsDf.read_feather(dlc_fp))
        # Checking the bodyparts specified in the configs exist in the dataframe
        KeypointsDf.check_bpts_exist(df, bpts)
        # Making data-long ways
        idx = pd.IndexSlice
        df = (
            df.loc[:, idx[:, bpts]]  # type: ignore
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
            y=Coords.LIKELIHOOD.value,
            hue=KeypointsDf.CN.BODYPARTS.value,
            alpha=0.4,
        )
        g.add_legend()
        # Saving plot
        g.savefig(out_fp)
        g.figure.clf()
        # Returning outcome string
        return outcome

    ###############################################################################################
    # MAKE BEHAVIOUR PLOTS
    ###############################################################################################

    @staticmethod
    def behav_plot(
        vid_fp: str,
        dlc_fp: str,
        behavs_fp: str,
        out_dir: str,
        configs_fp: str,
        overwrite: bool,
    ) -> str:
        """
        Make behaviour evaluation plot of the predicted and actual behaviours through time.
        """
        outcome = ""
        name = IOMixin.get_name(behavs_fp)
        out_dir = os.path.join(out_dir, Evaluate.behav_plot.__name__)
        out_fp = os.path.join(out_dir, f"{name}.png")
        os.makedirs(out_dir, exist_ok=True)
        # If overwrite is False, checking if we should skip processing
        if not overwrite and os.path.exists(out_fp):
            return DiagnosticsMixin.warning_msg()

        # Getting necessary config parameters
        configs = ExperimentConfigs.read_json(configs_fp)
        # configs_filt = configs.user.evaluate.behav_plot
        fps = float(configs.auto.formatted_vid.fps)

        # Read the file
        df = BehavDf.read_feather(behavs_fp)
        # Making data-long ways
        df = (
            df.stack([BehavDf.CN.BEHAVIOURS.value, BehavDf.CN.OUTCOMES.value])
            .reset_index()
            .rename(columns={0: "value"})
        )
        # Adding the timestamp column
        df["timestamp"] = df[BehavDf.IN.FRAME.value] / fps
        # Making plot
        g = sns.FacetGrid(
            df,
            row=BehavDf.CN.BEHAVIOURS.value,
            height=5,
            aspect=10,
        )
        g.map_dataframe(
            sns.lineplot,
            x="timestamp",
            y="value",
            hue=BehavDf.CN.OUTCOMES.value,
            alpha=0.4,
        )
        g.add_legend()
        # Saving plot
        g.savefig(out_fp)
        g.figure.clf()
        # Returning outcome string
        return outcome

    ###############################################################################################
    #               MAKE KEYPOINTS VIDEO
    ###############################################################################################

    @staticmethod
    def eval_vid(
        vid_fp: str,
        dlc_fp: str,
        # behavs_fp: str,
        analysis_fp: str,
        out_dir: str,
        configs_fp: str,
        overwrite: bool,
    ) -> str:
        """
        Run the DLC model on the formatted video to generate a DLC annotated video and DLC file for
        all experiments. The DLC model's config.yaml filepath must be specified in the `config_path`
        parameter in the `user` section of the config file.

        # TODO: implement analysis in eval vid.
        """
        outcome = ""
        name = IOMixin.get_name(vid_fp)
        out_dir = os.path.join(out_dir, Evaluate.eval_vid.__name__)
        out_fp = os.path.join(out_dir, f"{name}.mp4")
        os.makedirs(out_dir, exist_ok=True)
        # If overwrite is False, checking if we should skip processing
        if not overwrite and os.path.exists(out_fp):
            return DiagnosticsMixin.warning_msg()

        # Getting necessary config parameters
        configs = ExperimentConfigs.read_json(configs_fp)
        configs_filt = configs.user.evaluate.eval_vid
        funcs_names = configs.get_ref(configs_filt.funcs)
        pcutoff = configs.get_ref(configs_filt.pcutoff)
        colour_level = configs.get_ref(configs_filt.colour_level)
        radius = configs.get_ref(configs_filt.radius)
        cmap = configs.get_ref(configs_filt.cmap)

        # Modifying dlc_df and making list of how to select dlc_df components to optimise processing
        # Specifically:
        # - Filtering out "process" columns
        # - Rounding and converting to correct dtypes - "x" and "y" values are ints
        # - Changing the columns MultiIndex to a single-level index. For speedup
        # - Making the corresponding colours list for each bodypart instance (colours depend on indiv/bpt)
        # Getting dlc df
        dlc_df = KeypointsDf.clean_headings(KeypointsDf.read_feather(dlc_fp))
        # Filtering out IndivColumns.PROCESS.value columns
        if IndivColumns.PROCESS.value in dlc_df.columns.unique("individuals"):
            dlc_df.drop(columns=IndivColumns.PROCESS.value, level="individuals")
        # Getting (indivs, bpts) MultiIndex
        # TODO: make explicitly selecting (indivs, bpts) levels
        indivs_bpts_ls = dlc_df.columns.droplevel("coords").unique()
        # Rounding and converting to correct dtypes - "x" and "y" values are ints
        dlc_df = dlc_df.fillna(0)
        columns = dlc_df.columns[
            dlc_df.columns.get_level_values("coords").isin(["x", "y"])
        ]
        dlc_df[columns] = dlc_df[columns].round(0).astype(int)
        # Changing the columns MultiIndex to a single-level index. For speedup
        dlc_df.columns = [
            f"{indiv}_{bpt}_{coord}" for indiv, bpt, coord in dlc_df.columns
        ]
        # Making the corresponding colours list for each bodypart instance
        # (colours depend on indiv/bpt)
        colours_i, _ = pd.factorize(indivs_bpts_ls.get_level_values(colour_level))
        colours = (plt.cm.get_cmap(cmap)(colours_i / colours_i.max()) * 255)[
            :, [2, 1, 0, 3]
        ]

        # TODO: eventually ONLY need combined analysis df
        # Modifying analysis_df to optimise processing
        # Specifically:
        # - Making sure all relevant behaviour outcome columns exist by imputing
        # - Changing the columns MultiIndex to a single-level index. For speedup
        # Getting behavs df
        try:
            analysis_df = AnalyseDf.read_feather(analysis_fp)
        except FileNotFoundError:
            outcome += (
                "WARNING: behavs file not found or could not be loaded."
                + "Disregarding behaviour."
                + "If you have run the behaviour classifier, please check this file.\n"
            )
            analysis_df = AnalyseDf.init_df(dlc_df.index)

        # OPENING INPUT VIDEO
        # Open the input video
        in_cap = cv2.VideoCapture(vid_fp)
        # Storing output vid dimensions
        # as they can change depending on funcs_names
        in_width = int(in_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        in_height = int(in_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = in_cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(in_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # MAKING ANNOTATED VIDEO
        # Making VidFuncOrganiser object to annotate each frame with
        vid_func_runner = VidFuncRunner(
            func_names=funcs_names,
            w_i=in_width,
            h_i=in_height,
            # kwargs for EvalVidFuncBase
            dlc_df=dlc_df,
            analysis_df=analysis_df,
            indivs_bpts_ls=indivs_bpts_ls,
            colours=colours,
            pcutoff=pcutoff,
            radius=radius,
        )
        # Define the codec and create VideoWriter object
        out_cap = cv2.VideoWriter(
            out_fp,
            cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
            fps,
            (vid_func_runner.w_o, vid_func_runner.h_o),
        )
        # Annotating each frame using the created functions
        # TODO: NOTE: The funcs themselves will modify the frame size.
        # Not self-contained or modular but this is a workaround for now.
        # Annotating frames
        for i in trange(total_frames):
            # Reading next vid frame
            ret, frame = in_cap.read()
            if ret is False:
                break
            # Annotating frame
            arr_out = vid_func_runner(frame, i)
            # Writing annotated frame to the VideoWriter
            out_cap.write(arr_out)
        # Release video objects
        in_cap.release()
        out_cap.release()
        # Returning outcome string
        return outcome
