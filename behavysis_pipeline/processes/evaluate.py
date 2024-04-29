"""
Functions have the following format:

Parameters
----------
vid_fp : str
    the GPU's number so computation is done on this GPU.
dlc_fp : str
    _description_
behav_fp : str
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
from typing import Callable, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from behavysis_core.data_models.experiment_configs import ExperimentConfigs
from behavysis_core.mixins.df_io_mixin import DFIOMixin
from behavysis_core.mixins.diagnostics_mixin import DiagnosticsMixin
from behavysis_core.mixins.io_mixin import IOMixin
from behavysis_core.mixins.keypoints_mixin import KeypointsMixin
from behavysis_core.utils.constants import (
    BEHAV_ACTUAL_COL,
    BEHAV_COLUMN_NAMES,
    BEHAV_PRED_COL,
    BEHAV_PROB_COL,
    PLOT_DPI,
    PLOT_STYLE,
    PROCESS_COL,
)
from tqdm import trange

#####################################################################
#               MAKE KEYPOINTS PLOTS
#####################################################################


class Evaluate:
    @staticmethod
    def keypoints_plot(
        vid_fp: str,
        dlc_fp: str,
        behav_fp: str,
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
        # Read the file
        dlc_df = KeypointsMixin.clean_headings(DFIOMixin.read_feather(dlc_fp))
        # Getting indivs and bpts list
        _, bpts = KeypointsMixin.get_headings(dlc_df)
        cols_to_keep = dlc_df.columns.get_level_values("bodyparts").isin(bpts)
        # Making data-long ways
        dlc_df = (
            dlc_df.loc[:, cols_to_keep]
            .stack(["individuals", "bodyparts"], dropna=False)
            .reset_index()
            .rename(columns={"level_0": "frame"})
        )
        g = sns.FacetGrid(
            dlc_df,
            col="individuals",
            col_wrap=2,
            height=4,
        )
        g.map_dataframe(
            sns.lineplot,
            x="frame",
            y="likelihood",
            hue="bodyparts",
            alpha=0.4,
        )
        g.add_legend()
        # Saving plot
        g.savefig(out_fp)
        g.figure.clf()
        # Returning outcome string
        return outcome

    #####################################################################
    #               MAKE KEYPOINTS VIDEO
    #####################################################################

    @staticmethod
    def eval_vid(
        vid_fp: str,
        dlc_fp: str,
        behav_fp: str,
        out_dir: str,
        configs_fp: str,
        overwrite: bool,
    ) -> str:
        """
        Run the DLC model on the formatted video to generate a DLC annotated video and DLC file for
        all experiments. The DLC model's config.yaml filepath must be specified in the `config_path`
        parameter in the `user` section of the config file.
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
        funcs_names = configs_filt.funcs
        pcutoff = configs_filt.pcutoff
        colour_level = configs_filt.colour_level
        radius = configs_filt.radius
        cmap = configs_filt.cmap

        # Modifying dlc_df and making list of how to select dlc_df components to optimise processing
        dlc_df = KeypointsMixin.clean_headings(DFIOMixin.read_feather(dlc_fp))
        # Filtering out PROCESS_COL columns
        if PROCESS_COL in dlc_df.columns.unique("individuals"):
            dlc_df.drop(columns=PROCESS_COL, level="individuals")
        # Getting (indivs, bpts) MultiIndex
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
        # Making the corresponding colours list for each bodypart instance (colours depend on indiv/bpt)
        colours_i, _ = pd.factorize(indivs_bpts_ls.get_level_values(colour_level))
        colours = (plt.get_cmap(cmap)(colours_i / colours_i.max()) * 255)[
            :, [2, 1, 0, 3]
        ]

        # Getting behavs df
        try:
            behav_df = DFIOMixin.read_feather(behav_fp)
        except Exception:
            outcome += (
                "WARNING: behavs file not found or could not be loaded."
                + "Disregarding behaviour."
                + "If you have run the behaviour classifier, please check this file.\n"
            )
            behav_df = pd.DataFrame(
                index=dlc_df.index,
                columns=pd.MultiIndex.from_tuples((), names=BEHAV_COLUMN_NAMES),
            )
        # Getting list of behaviours
        behavs_ls = behav_df.columns.unique("behaviours")
        # Making sure all relevant behaviour outcome columns exist
        for behav in behavs_ls:
            for i in [BEHAV_PROB_COL, BEHAV_PRED_COL, BEHAV_ACTUAL_COL]:
                if (behav, i) not in behav_df:
                    behav_df[(behav, i)] = 0
        # Changing the columns MultiIndex to a single-level index. For speedup
        behav_df.columns = [f"{behav}_{outcome}" for behav, outcome in behav_df.columns]

        # MAKING ANNOTATED VIDEO
        # Settings the funcs for how to annotate the video
        funcs = list()
        for f_name in funcs_names:
            if f_name == "johansson":
                outcome += f"Added {f_name} to video. \n"
                new_func: Callable[[np.ndarray, int], np.ndarray] = (
                    lambda frame, i: annot_johansson(frame)
                )
            elif f_name == "keypoints":
                outcome += f"Added {f_name} to video. \n"
                new_func: Callable[[np.ndarray, int], np.ndarray] = (
                    lambda frame, i: annot_keypoints(
                        frame, dlc_df.loc[i], indivs_bpts_ls, colours, pcutoff, radius
                    )
                )
            elif f_name == "behavs":
                outcome += f"Added {f_name} to video. \n"
                new_func: Callable[[np.ndarray, int], np.ndarray] = (
                    lambda frame, i: annot_behav(frame, behav_df.loc[i], behavs_ls)
                )
            else:
                continue
            funcs.append(new_func)
        # Open the input video
        in_cap = cv2.VideoCapture(vid_fp)
        fps = in_cap.get(cv2.CAP_PROP_FPS)
        width = int(in_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(in_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(in_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Define the codec and create VideoWriter object
        out_cap = cv2.VideoWriter(
            out_fp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        # Annotating each frame using the created functions
        outcome += annotate(in_cap, out_cap, funcs, total_frames)
        # Release video objects
        in_cap.release()
        out_cap.release()
        # Returning outcome string
        return outcome


def annotate(
    in_cap: cv2.VideoCapture,
    out_cap: cv2.VideoWriter,
    funcs: Sequence[Callable],
    n: int,
) -> str:
    """
    Given a frame, and the annotation functions to perform on it, returns the annotated frame.

    Expects that each func in the array is given in the following form:
    ```f(frame: np.ndarray, i: int) -> np.ndarray```

    Parameters
    ----------
    in_cap : str
        cv2 frame array.
    out_cap : str
        _description_
    funcs : Sequence[Callable]
        _description_
    n : int
        _description_

    Returns
    -------
    str
        Outcome string.
    """
    outcome = ""
    # Annotating frames
    for i in trange(n):
        # Reading next frame
        ret, frame = in_cap.read()
        if ret is False:
            break
        # Annotating frame
        for f in funcs:
            try:
                frame = f(frame, i)
            except KeyError:
                pass
        # Writing annotated frame to the VideoWriter
        out_cap.write(frame)
    # Returning outcome string
    return outcome


def annot_johansson(frame: np.ndarray) -> np.ndarray:
    """
    Making black frame, in the style of Johansson.
    This means we see only the keypoints (i.e., what SimBA will see)

    Parameters
    ----------
    frame : np.ndarray
        cv2 frame array.

    Returns
    -------
    np.ndarray
        cv2 frame array.
    """
    return np.zeros(frame.shape, dtype=np.uint8)


def annot_keypoints(
    frame: np.ndarray,
    row: pd.Series | pd.DataFrame,
    indivs_bpts_ls: Sequence[tuple[str, str]],
    colours: Sequence[tuple[float, float, float, float]],
    pcutoff: float,
    radius: int,
) -> np.ndarray:
    """
    Adding the keypoints (given in `row`) to the frame.

    Parameters
    ----------
    frame : np.ndarray
        cv2 frame array.
    row : pd.Series
        row in DLC dataframe.
    indivs_bpts_ls : Sequence[tuple[str, str]]
        list of `(indiv, bpt)` tuples to include.
    colours : Sequence[tuple[float, float, float, float]]
        list of colour tuples, which correspond to each `indivs_bpts_ls` element.
    pcutoff : float
        _description_
    radius : int
        _description_

    Returns
    -------
    np.ndarray
        cv2 frame array.
    """
    # Making the bpts keypoints annot
    for i, (indiv, bpt) in enumerate(indivs_bpts_ls):
        if row[f"{indiv}_{bpt}_likelihood"] >= pcutoff:
            cv2.circle(
                frame,
                (int(row[f"{indiv}_{bpt}_x"]), int(row[f"{indiv}_{bpt}_y"])),
                radius=radius,
                color=colours[i],
                thickness=-1,
            )
    return frame


def annot_behav(
    frame: np.ndarray,
    row: pd.Series,
    behavs_ls: Sequence[str] | pd.Index,
) -> np.ndarray:
    """
    Annotates a text table in the top-left corner, with the format:
    ```
            actual pred
    Behav_1   X     X
    Behav_2         X
    ...
    ```

    Parameters
    ----------
    frame : np.ndarray
        cv2 frame array.
    row : pd.Series
        row in scored_behavs dataframe.
    behavs_ls : tuple[str]
        list of behaviours to include.

    Returns
    -------
    np.ndarray
        cv2 frame array.
    """
    # colour = (3, 219, 252)  # Yellow
    colour = (0, 0, 0)  # Black
    # Making outcome headings
    for j, outcome in enumerate((BEHAV_PRED_COL, BEHAV_ACTUAL_COL)):
        x = 120 + j * 40
        y = 50
        cv2.putText(frame, outcome, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
    # Making behav rows
    for i, behav in enumerate(behavs_ls):
        x = 20
        y = 100 + i * 30
        # Annotating with label
        cv2.putText(frame, behav, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
        for j, outcome in enumerate((BEHAV_PRED_COL, BEHAV_ACTUAL_COL)):
            x = 120 + j * 40
            if row[f"{behav}_{outcome}"] == 1:
                cv2.putText(
                    frame, "X", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2
                )
    return frame
