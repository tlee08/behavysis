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

from enum import Enum

import cv2
import numpy as np
import pandas as pd
from behavysis_core.constants import (
    BehavColumns,
)


class EvaluateVidFuncBase:
    """
    Calling the function returns the frame image (i.e. np.ndarray)
    with the function applied.
    """

    name = "evaluate_vid_func"

    x: int
    y: int
    w: int
    h: int

    def __init__(self, **kwargs):
        """
        Prepare function
        """
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0

    def __call__(self, frame: np.ndarray, idx: int) -> np.ndarray:
        """
        Run function
        """
        # TODO: make an abstract func?
        return np.array(0)

    # @staticmethod
    # def update_width(out_width):
    #     """
    #     Updates the output videos width
    #     """
    #     return out_width

    # @staticmethod
    # def update_height(out_height):
    #     """
    #     Updates the output videos height
    #     """
    #     return out_height


class Johansson(EvaluateVidFuncBase):
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

    name = "johansson"

    def __init__(self, **kwargs):
        pass

    def __call__(self, frame: np.ndarray, idx: int) -> np.ndarray:
        return np.zeros(frame.shape, dtype=np.uint8)


class Keypoints(EvaluateVidFuncBase):
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

    name = "keypoints"

    def __init__(self, dlc_df, indivs_bpts_ls, colours, pcutoff, radius, **kwargs):
        self.dlc_df: pd.DataFrame = dlc_df
        self.indivs_bpts_ls = indivs_bpts_ls
        self.colours = colours
        self.pcutoff = pcutoff
        self.radius = radius

    def __call__(self, frame: np.ndarray, idx: int) -> np.ndarray:
        # Getting row
        row = self.dlc_df.loc[idx]
        # Making the bpts keypoints annot
        for i, (indiv, bpt) in enumerate(self.indivs_bpts_ls):
            if row[f"{indiv}_{bpt}_likelihood"] >= self.pcutoff:
                cv2.circle(
                    frame,
                    (int(row[f"{indiv}_{bpt}_x"]), int(row[f"{indiv}_{bpt}_y"])),
                    radius=self.radius,
                    color=self.colours[i],
                    thickness=-1,
                )
        return frame


class Behavs(EvaluateVidFuncBase):
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

    name = "behavs"

    def __init__(self, behavs_df, behavs_ls, **kwargs):
        self.behavs_df: pd.DataFrame = behavs_df
        self.behavs_ls = behavs_ls

    def __call__(self, frame: np.ndarray, idx: int) -> np.ndarray:
        # Initialising the behav frame panel
        # TODO: a more modular way for frame shape?
        behav_frame = np.full(
            shape=frame.shape,
            fill_value=(255, 255, 255),
            dtype=np.uint8,
        )
        # Getting row
        row = self.behavs_df.loc[idx]
        # colour = (3, 219, 252)  # Yellow
        colour = (0, 0, 0)  # Black
        # Making outcome headings
        for j, outcome in enumerate((BehavColumns.PRED, BehavColumns.ACTUAL)):
            outcome = outcome.value
            x = 120 + j * 40
            y = 50
            cv2.putText(
                behav_frame, outcome, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2
            )
        # Making behav rows
        for i, behav in enumerate(self.behavs_ls):
            x = 20
            y = 100 + i * 30
            # Annotating with label
            cv2.putText(
                behav_frame, behav, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2
            )
            for j, outcome in enumerate((BehavColumns.PRED, BehavColumns.ACTUAL)):
                outcome = outcome.value
                x = 120 + j * 40
                if row[f"{behav}_{outcome}"] == 1:
                    cv2.putText(
                        behav_frame,
                        "X",
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        colour,
                        2,
                    )
        # Combining vid and behav frame panels
        frame = np.concatenate((frame, behav_frame), axis=1)
        return frame

    @staticmethod
    def update_width(out_width):
        # TODO: find a better way to update width (compounding x2 isn't sustainable for multiple funcs)
        return out_width * 2


class Analysis(EvaluateVidFuncBase):
    """
    Annotates a text table in the top-left corner, with the format:
    ```
            actual pred
    Behav_1   X     X
    Behav_2         X
    ...
    """

    name = "analysis"

    def __init__(self, behavs_df, behavs_ls, **kwargs):
        self.behavs_df: pd.DataFrame = behavs_df
        self.behavs_ls = behavs_ls

    def __call__(self, frame: np.ndarray, idx: int) -> np.ndarray:
        # Initialising the behav frame panel
        # TODO: use with pyqtgraph??
        return frame


class EvaluateVidFuncs(Enum):
    """Enum of the evaluate video functions."""

    JOHANSSON = Johansson
    KEYPOINTS = Keypoints
    BEHAVS = Behavs
    ANALYSIS = Analysis


# TODO have a VidFuncOrganiser class, which has list of vid func objects and can
# a) call them in order
# b) organise where they are in the frame (x, y)
# c) had vid metadata (height, width)
# Then implement in evaluate
class VidFuncOrganiser:
    """__summary__"""

    funcs: list[EvaluateVidFuncBase]
    w_i: int
    h_i: int
    w_o: int
    h_o: int

    def __init__(self, funcs):
        self.funcs = funcs
        self.w_i = 0
        self.h_i = 0
        self.w_o = 0
        self.h_o = 0

    def update_w_h(self, w_i: int, h_i: int):
        # Depends on funcs and input frame size
        # Updating input frame size
        self.w_i = w_i
        self.h_i = h_i
        # Updating output frame size
        # TODO

    def __call__(self, vid_frame: np.ndarray, idx: int):
        # Initialise output arr (image) with given dimensions
        arr = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        # For each function, get the outputted "video tile" and superimpose on arr
        for func in self.funcs:
            # Running func
            arr_i = func(vid_frame, idx)
            # Superimposing
            arr[
                func.y : func.y + func.h,
                func.x : func.x + func.w,
            ] = arr_i
        # Returning output arr
        return arr
