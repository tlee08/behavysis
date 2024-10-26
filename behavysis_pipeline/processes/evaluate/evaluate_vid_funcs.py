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

import cv2
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pg.exporters import ImageExporter
import matplotlib.pyplot as plt

from behavysis_pipeline.processes.analyse.analyse_combine import AnalyseCombineCN


class EvalVidFuncBase:
    """
    Calling the function returns the frame image (i.e. np.ndarray)
    with the function applied.
    """

    name = "evaluate_vid_func"

    w: int
    h: int

    def __init__(self, **kwargs):
        """
        Prepare function
        """
        self.w = 0
        self.h = 0

    def __call__(self, frame: np.ndarray, idx: int) -> np.ndarray:
        """
        Run function
        """
        # TODO: make an abstract func?
        return np.array(0)


class Johansson(EvalVidFuncBase):
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

    def __init__(self, w_i: int, h_i: int, **kwargs):
        self.w_i = w_i
        self.h_i = h_i

    def __call__(self, frame: np.ndarray, idx: int) -> np.ndarray:
        return np.full(
            shape=(self.h_i, self.w_i, 3),
            fill_value=(0, 0, 0),
            dtype=np.uint8,
        )


# TODO wrangle dlc_df HERE (not in eval_vid) for encapsulation
class Keypoints(EvalVidFuncBase):
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

    def __init__(
        self,
        w_i: int,
        h_i: int,
        dlc_df,
        indivs_bpts_ls,
        colours,
        pcutoff,
        radius,
        **kwargs,
    ):
        self.w_i = w_i
        self.h_i = h_i
        self.dlc_df: pd.DataFrame = dlc_df
        self.indivs_bpts_ls = indivs_bpts_ls
        self.colours = colours
        self.pcutoff = pcutoff
        self.radius = radius

    def __call__(self, frame: np.ndarray, idx: int) -> np.ndarray:
        # Getting row
        # Also checking that the idx exists
        # TODO: is this the fastest way?
        try:
            row = self.dlc_df.loc[idx]
        except KeyError:
            return frame
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


# class Behavs(EvalVidFuncBase):
#     """
#     Annotates a text table in the top-left corner, with the format:
#     ```
#             actual pred
#     Behav_1   X     X
#     Behav_2         X
#     ...
#     ```

#     Parameters
#     ----------
#     frame : np.ndarray
#         cv2 frame array.
#     row : pd.Series
#         row in scored_behavs dataframe.
#     behavs_ls : tuple[str]
#         list of behaviours to include.

#     Returns
#     -------
#     np.ndarray
#         cv2 frame array.
#     """

#     name = "behavs"

#     def __init__(self, w_i: int, h_i: int, behavs_df, behavs_ls, **kwargs):
#         self.w_i = w_i
#         self.h_i = h_i
#         self.behavs_df: pd.DataFrame = behavs_df
#         self.behavs_ls = behavs_ls

#     def __call__(self, frame: np.ndarray, idx: int) -> np.ndarray:
#         # Initialising the behavs frame panel
#         behav_tile = np.full(
#             shape=(self.h_i, self.w_i, 3),
#             fill_value=(255, 255, 255),
#             dtype=np.uint8,
#         )
#         # Getting row
#         # Also checking that the idx exists
#         # TODO: is this the fastest way?
#         try:
#             row = self.behavs_df.loc[idx]
#         except KeyError:
#             return behav_tile
#         # colour = (3, 219, 252)  # Yellow
#         colour = (0, 0, 0)  # Black
#         # Making outcome headings
#         for j, outcome in enumerate((BehavColumns.PRED, BehavColumns.ACTUAL)):
#             outcome = outcome.value
#             x = 120 + j * 40
#             y = 50
#             cv2.putText(
#                 behav_tile, outcome, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2
#             )
#         # Making behav rows
#         for i, behav in enumerate(self.behavs_ls):
#             x = 20
#             y = 100 + i * 30
#             # Annotating with label
#             cv2.putText(
#                 behav_tile, behav, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2
#             )
#             for j, outcome in enumerate((BehavColumns.PRED, BehavColumns.ACTUAL)):
#                 outcome = outcome.value
#                 x = 120 + j * 40
#                 if row[f"{behav}_{outcome}"] == 1:
#                     cv2.putText(
#                         behav_tile,
#                         "X",
#                         (x, y),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.5,
#                         colour,
#                         2,
#                     )
#         return behav_tile


class Analysis(EvalVidFuncBase):
    """
    Annotates a text table in the top-left corner, with the format:
    ```
            actual pred
    Behav_1   X     X
    Behav_2         X
    ...
    """

    name = "analysis"

    qimage_format = pg.QtGui.QImage.Format.Format_RGB888

    def __init__(
        self, w_i: int, h_i: int, analysis_df: pd.DataFrame, cmap: str, **kwargs
    ):
        self.w_i = w_i
        self.h_i = h_i
        self.analysis_df: pd.DataFrame = analysis_df
        self.cmap = cmap
        self.plot_arr = None
        self.x_line_arr = None

    def init_graph(self):
        # Making multi-plot widget
        plots_layout = pg.GraphicsLayoutWidget()
        # Getting list of different groups (`analysis`, `individuals` levels)
        df_columns = self.analysis_df.columns
        # For making separate plots in layout
        analysis_ls = df_columns.unique(AnalyseCombineCN.ANALYSIS.value)
        indivs_ls = df_columns.unique(AnalyseCombineCN.INDIVIDUALS.value)
        # Making each plot (from "analysis")
        self.plot_arr = np.zeros(shape=(len(analysis_ls), len(indivs_ls)), dtype=object)
        self.x_line_arr = np.copy(self.plot_arr)
        for i, analysis_i in enumerate(analysis_ls):
            for j, indivs_j in enumerate(indivs_ls):
                # Making plot
                self.plot_arr[i, j] = plots_layout.addPlot(
                    row=i,
                    col=j,
                    title=f"{analysis_i} - {indivs_j}",
                    labels={"left": "value", "bottom": "second"},
                )
                # Plot middle (current time) line
                self.x_line_arr[i, j] = pg.InfiniteLine(pos=0, angle=90)
                self.x_line_arr[i, j].setZValue(10)
                self.plot_arr[i, j].addItem(x_line)
                # Setting data
                # TODO implement for bouts as well
                measures_ls = df_columns.unique(AnalyseCombineCN.MEASURES.value)
                # Making the corresponding colours list for each bodypart instance
                # (colours depend on indiv/bpt)
                colours_idx, _ = pd.factorize(measures_ls)
                colours_ls = (
                    plt.cm.get_cmap(self.cmap)(colours_idx / colours_idx.max()) * 255
                )[:, [2, 1, 0, 3]]
                # make legend
                legend = self.plot_arr[i, j].addLegend()
                for k, measures_k in enumerate(measures_ls):
                    colours_k = colours_ls[k]
                    # make measure's line
                    line_item = pg.PlotDataItem(
                        x=self.analysis_df.index,
                        y=self.analysis_df[(analysis_i, indivs_j, measures_k)],
                        pen=pg.mkPen(color=colours_k),
                        brush=pg.mkBrush(color=colours_k),
                    )
                    line_item.setFillLevel(0.5)
                    self.plot_arr[i, j].addItem(line_item)
                    # make measure's legend
                    legend.addItem(item=line_item, title=measures_k)

    def __call__(self, frame: np.ndarray, idx: int) -> np.ndarray:
        # For each plot (rows (analysis), columns (indivs))
        # NOTE: may not allow np arrays in np arrays
        plot_imgs_arr = np.zeros(shape=self.plot_arr.shape, dtype=object)
        for i in range(self.plot_arr.shape[0]):
            for j in range(self.plot_arr.shape[1]):
                self.update_plot(idx, i, j)
                pself.plot2cv(i, j)

    def update_plot(self, idx: int, i: int, j: int):
        """
        For a single plot
        (as the plots_layout has rows (analysis) and columns (indivs)).
        """
        # TODO: implement custom seconds
        padding = 100
        self.x_line_arr[i, j].setPos(idx)
        self.plot_arr[i, j].setXRange(i - padding, i + padding)

    @classmethod
    def qt2cv(cls, img_qt: pg.QtGui.QImage) -> np.ndarray:
        """Convert from a QImage to an opencv image."""
        # QImage to RGB888 format
        img_qt = img_qt.convertToFormat(cls.qimage_format)
        # Get shape of image
        w, h = img_qt.width(), img_qt.height()
        # Get bytes pointer to image data
        ptr = img_qt.bits()
        # Bytes to cv2 image
        img_cv = np.array(ptr, dtype=np.uint8).reshape(h, w, 3)
        # Return cv2 image
        return img_cv

    @classmethod
    def plot2cv_(cls, plot):
        # Making pyqtgraph image exporter to bytes
        exporter = ImageExporter(plot.plotItem)
        # exporter.parameters()["width"] = self.width()
        # Exporting to QImage (bytes)
        img_qt = exporter.export(toBytes=True)
        # QImage to cv2 image (using mixin)
        img_cv = cls.qt2cv(img_qt)
        # cv2 BGR to RGB
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        # Resize to widget size
        # w, h = self.width(), self.height()
        # img_cv = cv2.resize(img_cv, (w, h), interpolation=cv2.INTER_AREA)
        # Return cv2 image
        return img_cv

    @classmethod
    def plot2cv(self, i: int, j: int):
        return self.plot2cv_(self.plot_arr[i, j])


# TODO have a VidFuncOrganiser class, which has list of vid func objects and can
# a) call them in order
# b) organise where they are in the frame (x, y)
# c) had vid metadata (height, width)
# Then implement in evaluate
# NOTE: maybe have funcs_vid, funcs_behav, and funcs_analysis lists separately
# for the tiles.
class VidFuncRunner:
    """
    Given a list of the EvalVidFuncBase funcs to run in the constructor,
    it can be called as a function to convert a video frame and df index
    (corresponding to keypoints_df, behav_df, and analysis_fbf_df) to an
    "evaluation frame", which is annotated with keypoints and tiled with
    analysis/behav graphs.

    Tiling is always:
    +----------------------+
    | vid       | analysis |
    | keypoints | graphs   |
    +-----------|          |
    | blank     |          |
    |           |          |
    +-----------+----------+
    """

    johansson: Johansson | None
    keypoints: Keypoints | None
    analysis: Analysis | None

    w_i: int
    h_i: int
    w_o: int
    h_o: int

    def __init__(self, func_names: list[str], w_i: int, h_i: int, **kwargs):
        """
        NOTE: kwargs are the constructor parameters for
        EvalVidFuncBase classes.
        """
        # Storing frame input dimensions
        self.w_i = w_i
        self.h_i = h_i

        # Initialising funcs from func_names_ls
        self.funcs = []
        # NOTE: ORDER MATTERS so going through in predefined order
        # Concatenating Vid, Behav, and Analysis funcs together in order
        func_check_ls = [Johansson, Keypoints, Analysis]
        for func in func_check_ls:
            setattr(self, func.name, None)
        # Creating EvalVidFuncBase instances and adding to funcs list
        for func in func_check_ls:
            if func.name in func_names:
                setattr(
                    self,
                    func.name,
                    func(w_i=w_i, h_i=h_i, **kwargs),
                )

        # TODO: update w_o and h_o accoridng to analysis_df
        # Storing frame output dimensions
        # width
        # vid panel
        self.w_o = self.w_i
        # # analysis panel
        # if self.analysis:
        #     self.w_o = self.w_o * 2
        # # height
        # # vid panel
        self.h_o = self.h_i
        # # behav panel
        # if self.behavs:
        #     self.h_o = self.h_o * 2

    def __call__(self, vid_frame: np.ndarray, idx: int):
        # Initialise output arr (image) with given dimensions
        arr_out = np.zeros(shape=(self.h_o, self.w_o, 3), dtype=np.uint8)
        # For overwriting vid_frame
        arr_video = np.copy(vid_frame)

        # video tile
        if self.johansson:
            arr_video = self.johansson(arr_video, idx)
        if self.keypoints:
            arr_video = self.keypoints(arr_video, idx)
        arr_out[: self.h_i, : self.w_i] = arr_video
        # analysis tile
        if self.analysis:
            arr_analysis = self.analysis(arr_video, idx)
            arr_out[: self.h_o, self.w_i : self.w_o] = arr_analysis
        # Returning output arr
        return arr_out
