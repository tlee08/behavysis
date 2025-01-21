"""
__summary__
"""

import os
from abc import ABC, abstractmethod

import cv2
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from PySide6 import QtGui
from tqdm import trange

from behavysis_pipeline.df_classes.analyse_combined_df import AnalyseCombinedDf
from behavysis_pipeline.df_classes.keypoints_df import IndivColumns, KeypointsDf
from behavysis_pipeline.pydantic_models.experiment_configs import ExperimentConfigs
from behavysis_pipeline.utils.diagnostics_utils import file_exists_msg
from behavysis_pipeline.utils.logging_utils import init_logger, logger_func_decorator
from behavysis_pipeline.utils.plotting_utils import make_colours

###################################################################################################
# EVALUATE VID FUNC, WHICH FACES OUT
###################################################################################################


class EvaluateVid:
    """__summary__"""

    logger = init_logger(__name__)

    @classmethod
    @logger_func_decorator(logger)
    def evaluate_vid(
        cls,
        vid_fp: str,
        dlc_fp: str,
        analyse_combined_fp: str,
        out_fp: str,
        configs_fp: str,
        overwrite: bool,
    ) -> str:
        """
        Run the DLC model on the formatted video to generate a DLC annotated video and DLC file for
        all experiments. The DLC model's config.yaml filepath must be specified in the `config_path`
        parameter in the `user` section of the config file.
        """
        if not overwrite and os.path.exists(out_fp):
            return file_exists_msg(out_fp)
        outcome = ""
        # Getting necessary config parameters
        configs = ExperimentConfigs.read_json(configs_fp)
        configs_filt = configs.user.evaluate_vid
        funcs_names = configs.get_ref(configs_filt.funcs)
        pcutoff = configs.get_ref(configs_filt.pcutoff)
        colour_level = configs.get_ref(configs_filt.colour_level)
        radius = configs.get_ref(configs_filt.radius)
        cmap = configs.get_ref(configs_filt.cmap)
        padding = configs.get_ref(configs_filt.padding)
        width_input = configs.auto.formatted_vid.width_px
        height_input = configs.auto.formatted_vid.height_px
        fps = configs.auto.formatted_vid.fps
        total_frames = configs.auto.formatted_vid.total_frames

        # Asserting input video metadata is valid
        assert_msg = "Input %s must be greater than 0. Run `exp.format_vid`."
        assert width_input > 0, assert_msg % "video width"
        assert height_input > 0, assert_msg % "video height"
        assert fps > 0, assert_msg % "video fps"
        assert total_frames > 0, assert_msg % "video total frames"

        # Getting dlc df
        dlc_df = KeypointsDf.clean_headings(KeypointsDf.read(dlc_fp))
        # Getting analysis combined df
        try:
            analysis_df = AnalyseCombinedDf.read(analyse_combined_fp)
        except FileNotFoundError:
            outcome += "WARNING: analysis combined file not found or could not be loaded." "Disregarding analysis."
            analysis_df = AnalyseCombinedDf.init_df(dlc_df.index)

        # MAKING ANNOTATED VIDEO
        # Making VidFuncsRunner object to annotate each frame with.
        vid_funcs_runner = VidFuncsRunner(
            func_names=funcs_names,
            width_input=width_input,
            height_input=height_input,
            # kwargs for EvalVidFuncBase
            dlc_df=dlc_df,
            analysis_df=analysis_df,
            colour_level=colour_level,
            pcutoff=pcutoff,
            radius=radius,
            cmap=cmap,
            padding=padding,
            fps=fps,
        )
        # Opening the input video
        in_cap = cv2.VideoCapture(vid_fp)
        # Making output folder
        os.makedirs(os.path.dirname(out_fp), exist_ok=True)
        # Define the codec and create VideoWriter object
        out_cap = cv2.VideoWriter(
            out_fp,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (vid_funcs_runner.width_out, vid_funcs_runner.height_out),
        )
        # Annotating each frame using the created functions
        for idx in trange(total_frames):
            # Reading next vid frame
            ret, frame = in_cap.read()
            if ret is False:
                break
            # Annotating frame
            arr_out = vid_funcs_runner(frame, idx)
            # Writing annotated frame to the VideoWriter
            out_cap.write(arr_out)
        # Release video objects
        in_cap.release()
        out_cap.release()
        # Returning outcome string
        return outcome


###################################################################################################
# INDIVIDUAL VID FUNCS
###################################################################################################


class EvalVidFuncBase(ABC):
    """
    Calling the function returns the frame image (i.e. np.ndarray)
    with the function applied.
    """

    name: str
    width_output: int
    height_output: int

    @abstractmethod
    def __init__(self, **kwargs):
        """Prepare function"""
        pass

    @abstractmethod
    def __call__(self, frame: np.ndarray, idx: int) -> np.ndarray:
        """Run function"""
        pass


class Johansson(EvalVidFuncBase):
    """
    Making black frame, in the style of Johansson.
    This means we see only the keypoints (i.e., what SimBA will see)
    """

    name = "johansson"

    def __init__(self, width_input: int, height_input: int, **kwargs):
        self.width_output = width_input
        self.heigth_output = height_input

    def __call__(self, frame: np.ndarray, idx: int) -> np.ndarray:
        return np.full(
            shape=(self.heigth_output, self.width_output, 3),
            fill_value=(0, 0, 0),
            dtype=np.uint8,
        )


# TODO wrangle dlc_df HERE (not in eval_vid) for encapsulation
class Keypoints(EvalVidFuncBase):
    """
    Adding the keypoints (given in `row`) to the frame.
    """

    name = "keypoints"

    def __init__(
        self,
        width_input: int,
        height_input: int,
        dlc_df,
        colour_level,
        cmap,
        pcutoff,
        radius,
        **kwargs,
    ):
        self.width_output = width_input
        self.height_output = height_input
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
        if IndivColumns.PROCESS.value in self.dlc_df.columns.unique(KeypointsDf.CN.INDIVIDUALS.value):
            self.dlc_df.drop(
                columns=IndivColumns.PROCESS.value,
                level=KeypointsDf.CN.INDIVIDUALS.value,
            )
        # Getting (indivs, bpts) MultiIndex
        # TODO: make explicitly selecting (indivs, bpts) levels
        self.indivs_bpts_ls = self.dlc_df.columns.droplevel(level=KeypointsDf.CN.COORDS.value).unique()
        # Rounding and converting to correct dtypes - "x" and "y" values are ints
        self.dlc_df = self.dlc_df.fillna(0)
        columns = self.dlc_df.columns[self.dlc_df.columns.get_level_values("coords").isin(["x", "y"])]
        self.dlc_df[columns] = self.dlc_df[columns].round(0).astype(int)
        # Changing the columns MultiIndex to a single-level index. For speedup
        self.dlc_df.columns = [f"{indiv}_{bpt}_{coord}" for indiv, bpt, coord in self.dlc_df.columns]
        # Making the corresponding colours list for each bodypart instance
        # (colours depend on indiv/bpt)
        measures_ls = self.indivs_bpts_ls.get_level_values(self.colour_level)
        self.colours = make_colours(measures_ls, self.cmap)

    def __call__(self, frame: np.ndarray, idx: int) -> np.ndarray:
        # Asserting the frame's dimensions
        assert frame.shape[0] == self.height_output
        assert frame.shape[1] == self.width_output
        # Getting idx row and asserting the idx exists
        try:
            row = self.dlc_df.loc[idx]
        except KeyError:
            return frame
        # Making the bpts keypoints annot
        for i, (indiv, bpt) in enumerate(self.indivs_bpts_ls):
            if row[f"{indiv}_{bpt}_likelihood"] >= self.pcutoff:
                cv2.circle(
                    img=frame,
                    center=(int(row[f"{indiv}_{bpt}_x"]), int(row[f"{indiv}_{bpt}_y"])),
                    radius=self.radius,
                    color=self.colours[i],
                    thickness=-1,
                )
        return frame


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
        width_input: int,
        height_input: int,
        analysis_df: pd.DataFrame,
        cmap: str,
        padding: int,
        fps: float,
        **kwargs,
    ):
        # TODO make aspect-ratio-weighted value for w_i.
        # Maybe have custom configs value `w_h_ratio`
        self.width_output = width_input
        self.height_output = height_input
        self.analysis_df: pd.DataFrame = analysis_df
        self.cmap = cmap
        self.padding = padding
        self.fps = fps
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
        # Getting the uniques analysis group names
        # And calculating each plot's height
        analysis_ls = self.analysis_df.columns.unique(AnalyseCombinedDf.CN.ANALYSIS.value)
        height_plot = int(np.round(self.height_output / len(analysis_ls)))
        # Making list of lists to store each plot (for "analysis")
        self.plot_arr = []
        self.x_line_arr = []
        for i, analysis_i in enumerate(analysis_ls):
            # Getting the uniques individual names in the analysis group
            # And calculating the width of each plot in the current row
            indivs_ls = self.analysis_df[(analysis_i,)].columns.unique(AnalyseCombinedDf.CN.INDIVIDUALS.value)
            width_plot = int(np.round(self.width_output / len(indivs_ls)))
            # Making list to store each plot (for "individuals")
            plot_arr_i = []
            x_line_arr_i = []
            for j, indivs_j in enumerate(indivs_ls):
                # Getting measures_ls, based on current analysis_i and indivs_j
                measures_ls = self.analysis_df[(analysis_i, indivs_j)].columns.unique(
                    AnalyseCombinedDf.CN.MEASURES.value
                )
                # Making plot
                plot_arr_ij = self.plots_layout.addPlot(
                    row=i,
                    col=j,
                    title=f"{analysis_i} - {indivs_j}",
                    labels={"left": "value", "bottom": "second"},
                )
                # Setting width and height
                plot_arr_ij.setFixedHeight(height_plot)
                plot_arr_ij.setFixedWidth(width_plot)
                # Plot "Current Time" vertical line
                x_line_arr_ij = pg.InfiniteLine(pos=0, angle=90)
                x_line_arr_ij.setZValue(10)
                plot_arr_ij.addItem(x_line_arr_ij)
                # Making the corresponding colours list for each measures instance
                colours_ls = make_colours(measures_ls, self.cmap)
                # Making overal plot's legend
                legend = plot_arr_ij.addLegend()
                for k, measures_k in enumerate(measures_ls):
                    # Making measure's line
                    # Using seconds (frames / fps). "update_plot" method also converts to seconds
                    line_item = pg.PlotDataItem(
                        x=self.analysis_df.index.values / self.fps,
                        y=self.analysis_df[(analysis_i, indivs_j, measures_k)].values,
                        pen=pg.mkPen(color=colours_ls[k], width=5),
                        # brush=pg.mkBrush(color=colours_ls[k]),
                    )
                    # line_item.setFillLevel(0)
                    # Adding measure line to plot
                    plot_arr_ij.addItem(line_item)
                    # Adding measure line to legend
                    legend.addItem(item=line_item, name=measures_k)
                # Adding to row lists
                plot_arr_i.append(plot_arr_ij)
                x_line_arr_i.append(x_line_arr_ij)
            # Adding to list-of-lists
            self.plot_arr.append(plot_arr_i)
            self.x_line_arr.append(x_line_arr_i)

    def __call__(self, frame: np.ndarray, idx: int) -> np.ndarray:
        # For each plot (rows (analysis), columns (indivs))
        plot_frame = np.full(
            shape=(self.height_output, self.width_output, 3),
            fill_value=(0, 0, 0),
            dtype=np.uint8,
        )
        # Initialising columns start
        height_plot_start = 0
        for i in range(len(self.plot_arr)):
            # Initialising rows start
            width_plot_start = 0
            for j in range(len(self.plot_arr[i])):
                # Updating plot
                self.update_plot(idx, i, j)
                # Making plot frame (as cv2 image)
                plot_frame_ij = self.plot2cv_(self.plot_arr[i][j])
                # Superimposing plot_frame_ij on plot_frame
                plot_frame[
                    height_plot_start : height_plot_start + plot_frame_ij.shape[0],
                    width_plot_start : width_plot_start + plot_frame_ij.shape[1],
                ] = plot_frame_ij
                # Updating columns start
                width_plot_start += plot_frame_ij.shape[1]
            # Updating rows start
            height_plot_start += plot_frame_ij.shape[0]
        # Returning
        return plot_frame

    def update_plot(self, idx: int, i: int, j: int):
        """
        For a single plot
        (as the plots_layout has rows (analysis) and columns (indivs)).

        NOTE: idx is
        """
        # Converting index from frames to seconds
        secs = idx / self.fps
        self.x_line_arr[i][j].setPos(secs)
        self.plot_arr[i][j].setXRange(secs - self.padding, secs + self.padding)

    @classmethod
    def qt2cv(cls, img_qt: QtGui.QImage) -> np.ndarray:
        """Convert from a QImage to an opencv image."""
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
        # NOTE: adjust the static number for bytes per pixel
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
        img_cv = cls.qt2cv(img_qt)
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
        img_cv = cls.qt2cv(img_qt)
        # cv2 BGR to RGB
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        # Resize to widget size
        # w, h = self.width(), self.height()
        # img_cv = cv2.resize(img_cv, (w, h), interpolation=cv2.INTER_AREA)
        # Return cv2 image
        return img_cv


class VidFuncsRunner:
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

    width_input: int
    height_input: int
    width_out: int
    height_out: int

    def __init__(self, func_names: list[str], width_input: int, height_input: int, **kwargs):
        """
        NOTE: kwargs are the constructor parameters for
        EvalVidFuncBase classes.
        """
        # Storing frame input dimensions
        self.width_input = width_input
        self.height_input = height_input

        # Initialising funcs from func_names_ls
        self.funcs = []
        # NOTE: ORDER MATTERS so going through in predefined order
        # Concatenating Vid, Behav, and Analysis funcs together in order
        func_check_ls: list[EvalVidFuncBase] = [Johansson, Keypoints, Analysis]
        # Creating EvalVidFuncBase instances and adding to funcs list
        for func in func_check_ls:
            if func.name in func_names:
                setattr(
                    self,
                    func.name,
                    func(width_input=width_input, height_input=height_input, **kwargs),
                )
            else:
                setattr(self, func.name, None)

        # TODO: update w_o and h_o accoridng to analysis_df
        # Storing frame output dimensions
        # width
        # vid panel
        self.width_out = self.width_input
        # analysis panel
        if self.analysis:
            self.width_out = self.width_out * 2
        # # height
        # # vid panel
        self.height_out = self.height_input
        # # behav panel
        # if self.behavs:
        #     self.h_o = self.h_o * 2

    def __call__(self, vid_frame: np.ndarray, idx: int):
        # Initialise output arr (image) with given dimensions
        arr_out = np.zeros(shape=(self.height_out, self.width_out, 3), dtype=np.uint8)
        # For overwriting vid_frame
        arr_video = np.copy(vid_frame)
        # video tile
        if self.johansson:
            arr_video = self.johansson(arr_video, idx)
        if self.keypoints:
            arr_video = self.keypoints(arr_video, idx)
        arr_out[: self.height_input, : self.width_input] = arr_video
        # analysis tile
        if self.analysis:
            arr_analysis = self.analysis(arr_video, idx)
            arr_out[
                : self.analysis.height_output,
                self.width_input : self.width_input + self.analysis.width_output,
            ] = arr_analysis
        # Returning output arr
        return arr_out
