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
import pandas as pd
import seaborn as sns
from tqdm import trange

from behavysis_pipeline.df_classes.behav_df import BehavDf
from behavysis_pipeline.df_classes.keypoints_df import Coords, KeypointsDf
from behavysis_pipeline.processes.evaluate_vid import (
    VidFuncRunner,
)
from behavysis_pipeline.pydantic_models.experiment_configs import ExperimentConfigs
from behavysis_pipeline.utils.diagnostics_utils import file_exists_msg
from behavysis_pipeline.utils.io_utils import get_name


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
        name = get_name(dlc_fp)
        out_dir = os.path.join(out_dir, Evaluate.keypoints_plot.__name__)
        out_fp = os.path.join(out_dir, f"{name}.png")
        os.makedirs(out_dir, exist_ok=True)

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
        name = get_name(behavs_fp)
        out_dir = os.path.join(out_dir, Evaluate.behav_plot.__name__)
        out_fp = os.path.join(out_dir, f"{name}.png")
        os.makedirs(out_dir, exist_ok=True)
        # If overwrite is False, checking if we should skip processing
        if not overwrite and os.path.exists(out_fp):
            return file_exists_msg()

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
        name = get_name(vid_fp)
        out_dir = os.path.join(out_dir, Evaluate.eval_vid.__name__)
        out_fp = os.path.join(out_dir, f"{name}.mp4")
        os.makedirs(out_dir, exist_ok=True)
        # If overwrite is False, checking if we should skip processing
        if not overwrite and os.path.exists(out_fp):
            return file_exists_msg()

        # Getting necessary config parameters
        configs = ExperimentConfigs.read_json(configs_fp)
        configs_filt = configs.user.evaluate.eval_vid
        funcs_names = configs.get_ref(configs_filt.funcs)
        pcutoff = configs.get_ref(configs_filt.pcutoff)
        colour_level = configs.get_ref(configs_filt.colour_level)
        radius = configs.get_ref(configs_filt.radius)
        cmap = configs.get_ref(configs_filt.cmap)
        padding = configs.get_ref(configs_filt.padding)

        # Getting dlc df
        dlc_df = KeypointsDf.clean_headings(KeypointsDf.read_feather(dlc_fp))

        try:
            analysis_df = AnalyseCombineDf.read_feather(analysis_fp)
        except FileNotFoundError:
            outcome += (
                "WARNING: behavs file not found or could not be loaded."
                "Disregarding behaviour."
                "If you have run the behaviour classifier, please check this file.\n"
            )
            analysis_df = AnalyseCombineDf.init_df(dlc_df.index)

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
            width_input=in_width,
            height_input=in_height,
            # kwargs for EvalVidFuncBase
            dlc_df=dlc_df,
            analysis_df=analysis_df,
            colour_level=colour_level,
            pcutoff=pcutoff,
            radius=radius,
            cmap=cmap,
            padding=padding,
        )
        # Define the codec and create VideoWriter object
        out_cap = cv2.VideoWriter(
            out_fp,
            cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
            fps,
            (vid_func_runner.width_out, vid_func_runner.height_out),
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
