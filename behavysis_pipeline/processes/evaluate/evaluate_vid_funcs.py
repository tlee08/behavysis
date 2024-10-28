"""
__summary__
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyqtgraph as pg
from behavysis_core.df_classes.analyse_combine_df import AnalyseCombineDf
from behavysis_core.df_classes.keypoints_df import IndivColumns, KeypointsDf
from pyqtgraph.exporters import ImageExporter
from PySide6 import QtGui


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
        colour_level,
        cmap,
        pcutoff,
        radius,
        **kwargs,
    ):
        self.w_i = w_i
        self.h_i = h_i
        self.dlc_df: pd.DataFrame = dlc_df
        self.colour_level = colour_level
        self.cmap = cmap
        self.pcutoff = pcutoff
        self.radius = radius

        self.init_df()

    def init_df(self):
        """
        Modifying dlc_df and making list of how to select dlc_df components to optimise processing
        Specifically:
        - Filtering out "process" columns
        - Rounding and converting to correct dtypes - "x" and "y" values are ints
        - Changing the columns MultiIndex to a single-level index. For speedup
        - Making the corresponding colours list for each bodypart instance (colours depend on indiv/bpt)
        """
        # Filtering out IndivColumns.PROCESS.value columns
        if IndivColumns.PROCESS.value in self.dlc_df.columns.unique(
            KeypointsDf.CN.INDIVIDUALS.value
        ):
            self.dlc_df.drop(
                columns=IndivColumns.PROCESS.value,
                level=KeypointsDf.CN.INDIVIDUALS.value,
            )
        # Getting (indivs, bpts) MultiIndex
        # TODO: make explicitly selecting (indivs, bpts) levels
        self.indivs_bpts_ls = self.dlc_df.columns.droplevel(
            level=KeypointsDf.CN.COORDS.value
        ).unique()
        # Rounding and converting to correct dtypes - "x" and "y" values are ints
        self.dlc_df = self.dlc_df.fillna(0)
        columns = self.dlc_df.columns[
            self.dlc_df.columns.get_level_values("coords").isin(["x", "y"])
        ]
        self.dlc_df[columns] = self.dlc_df[columns].round(0).astype(int)
        # Changing the columns MultiIndex to a single-level index. For speedup
        self.dlc_df.columns = [
            f"{indiv}_{bpt}_{coord}" for indiv, bpt, coord in self.dlc_df.columns
        ]
        # Making the corresponding colours list for each bodypart instance
        # (colours depend on indiv/bpt)
        measures_ls = self.indivs_bpts_ls.get_level_values(self.colour_level)
        self.colours = make_colours(measures_ls, self.cmap)

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
                    (int(row[f"{indiv}_{bpt}_x"]), int(row[f"{indiv}_{bpt}_y"])),  # type: ignore
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

    qimage_format = QtGui.QImage.Format.Format_RGB888

    def __init__(
        self,
        w_i: int,
        h_i: int,
        analysis_df: pd.DataFrame,
        cmap: str,
        padding: int,
        **kwargs,
    ):
        # TODO make aspect ratio weighted w_i. Maybe have custom configs value
        self.w_i = w_i
        self.h_i = h_i
        self.analysis_df: pd.DataFrame = analysis_df
        self.cmap = cmap
        self.padding = padding
        self.init_graph()

    def init_graph(self):
        """
        Modifying analysis_df to optimise processing
        Specifically:
        - Making sure all relevant behaviour outcome columns exist by imputing
        - Changing the columns MultiIndex to a single-level index. For speedup
        Getting behavs df
        """
        # Making multi-plot widget
        self.plots_layout = pg.GraphicsLayoutWidget()
        # Getting list of different groups (`analysis`, `individuals` levels)
        # For making separate plots in layout
        df_columns = self.analysis_df.columns
        analysis_ls = df_columns.unique(AnalyseCombineDf.CN.ANALYSIS.value)
        indivs_ls = df_columns.unique(AnalyseCombineDf.CN.INDIVIDUALS.value)
        # Calculating each plot's width and height
        self.h_p = int(np.round(self.h_i / len(analysis_ls)))
        self.w_p = int(np.round(self.w_i / len(indivs_ls)))
        # Making each plot (from "analysis")
        self.plot_arr = np.zeros(shape=(len(analysis_ls), len(indivs_ls)), dtype=object)
        self.x_line_arr = np.copy(self.plot_arr)
        for i, analysis_i in enumerate(analysis_ls):
            for j, indivs_j in enumerate(indivs_ls):
                # Getting measures_ls, based on current analysis_i and indivs_j
                measures_ls = self.analysis_df[(analysis_i, indivs_j)].columns.unique(
                    AnalyseCombineDf.CN.MEASURES.value
                )
                # Making plot
                self.plot_arr[i, j] = self.plots_layout.addPlot(  # type: ignore
                    row=i,
                    col=j,
                    title=f"{analysis_i} - {indivs_j}",
                    labels={"left": "value", "bottom": "second"},
                )
                # Setting width and height
                self.plot_arr[i, j].setFixedHeight(self.h_p)
                self.plot_arr[i, j].setFixedWidth(self.w_p)
                # Plot middle (current time) line
                self.x_line_arr[i, j] = pg.InfiniteLine(pos=0, angle=90)
                self.x_line_arr[i, j].setZValue(10)
                self.plot_arr[i, j].addItem(self.x_line_arr[i, j])
                # Setting data
                # TODO implement for bouts as well, NOT just line graph
                # Making the corresponding colours list for each measures instance
                colours_ls = make_colours(measures_ls, self.cmap)
                # make legend
                legend = self.plot_arr[i, j].addLegend()
                for k, measures_k in enumerate(measures_ls):
                    colours_k = colours_ls[k]
                    # make measure's line
                    line_item = pg.PlotDataItem(
                        x=self.analysis_df.index.values,
                        y=self.analysis_df[(analysis_i, indivs_j, measures_k)].values,
                        pen=pg.mkPen(color=colours_k, width=5),
                        # brush=pg.mkBrush(color=colours_k),
                    )
                    # line_item.setFillLevel(0)
                    self.plot_arr[i, j].addItem(line_item)
                    # make measure's legend
                    legend.addItem(item=line_item, name=measures_k)

    def __call__(self, frame: np.ndarray, idx: int) -> np.ndarray:
        # For each plot (rows (analysis), columns (indivs))
        plot_frame = np.full(
            shape=(self.h_i, self.w_i, 3),
            fill_value=(0, 0, 0),
            dtype=np.uint8,
        )
        # plot_frame = self.grl2cv_(self.plots_layout)
        for i in range(self.plot_arr.shape[0]):
            for j in range(self.plot_arr.shape[1]):
                self.update_plot(idx, i, j)
                plot_frame[
                    self.h_p * i : self.h_p * (i + 1),
                    self.w_p * j : self.w_p * (j + 1),
                ] = self.plot2cv_(self.plot_arr[i, j])
        # Returning
        return plot_frame

    def update_plot(self, idx: int, i: int, j: int):
        """
        For a single plot
        (as the plots_layout has rows (analysis) and columns (indivs)).
        """
        self.x_line_arr[i, j].setPos(idx)
        self.plot_arr[i, j].setXRange(idx - self.padding, idx + self.padding)

    @classmethod
    def qt2cv(cls, img_qt: QtGui.QImage) -> np.ndarray:
        """Convert from a QImage to an opencv image."""
        # NOTE: TODO: Implement remove padding in behavysis_viewer too
        # QImage to RGB888 format
        img_qt = img_qt.convertToFormat(cls.qimage_format)
        # Get shape of image
        w = img_qt.width()
        h = img_qt.height()
        bpl = img_qt.bytesPerLine()
        # Get bytes pointer to image data
        ptr = img_qt.bits()
        # Bytes to numpy 1d arr
        img_cv = np.array(ptr, dtype=np.uint8)
        # Reshaping to height-bytesPerLine format
        img_cv = img_cv.reshape(h, bpl)
        # Remove the padding bytes
        img_cv = img_cv[:, : w * 3]
        # Reshaping to cv2 format
        img_cv = img_cv.reshape(h, w, 3)
        # Return cv2 image
        return img_cv

    @classmethod
    def grl2cv_(cls, grl):
        # Making pyqtgraph image exporter to bytes
        # TODO: was original. check this still works
        exporter = ImageExporter(grl.scene())
        # exporter.parameters()["width"] = self.width()
        # Exporting to QImage (bytes)
        img_qt = exporter.export(toBytes=True)
        # QImage to cv2 image (using mixin)
        img_cv = cls.qt2cv(img_qt)  # type: ignore
        # cv2 BGR to RGB
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        # Resize to widget size
        # w, h = self.width(), self.height()
        # img_cv = cv2.resize(img_cv, (w, h), interpolation=cv2.INTER_AREA)
        # Return cv2 image
        return img_cv

    @classmethod
    def plot2cv_(cls, plot):
        # Making pyqtgraph image exporter to bytes
        # TODO: was original. check this still works
        # exporter = ImageExporter(plot.plotItem)
        exporter = ImageExporter(plot)
        # exporter.parameters()["width"] = self.width()
        # Exporting to QImage (bytes)
        img_qt = exporter.export(toBytes=True)
        # QImage to cv2 image (using mixin)
        img_cv = cls.qt2cv(img_qt)  # type: ignore
        # cv2 BGR to RGB
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        # Resize to widget size
        # w, h = self.width(), self.height()
        # img_cv = cv2.resize(img_cv, (w, h), interpolation=cv2.INTER_AREA)
        # Return cv2 image
        return img_cv


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
        # Creating EvalVidFuncBase instances and adding to funcs list
        for func in func_check_ls:
            if func.name in func_names:
                setattr(
                    self,
                    func.name,
                    func(w_i=w_i, h_i=h_i, **kwargs),
                )
            else:
                setattr(self, func.name, None)

        # TODO: update w_o and h_o accoridng to analysis_df
        # Storing frame output dimensions
        # width
        # vid panel
        self.w_o = self.w_i
        # analysis panel
        if self.analysis:
            self.w_o = self.w_o * 2
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
            arr_out[
                : self.analysis.h_i,
                self.w_i : self.w_i + self.analysis.w_i,
            ] = arr_analysis
        # Returning output arr
        return arr_out


def make_colours(vals, cmap):
    colours_idx, _ = pd.factorize(vals)
    colours_ls = (plt.cm.get_cmap(cmap)(colours_idx / colours_idx.max()) * 255)[
        :, [2, 1, 0, 3]
    ]
    return colours_ls
