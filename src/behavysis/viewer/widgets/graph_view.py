from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pandas as pd
from pyqtgraph import BarGraphItem, InfiniteLine, PlotWidget, mkBrush
from pyqtgraph.exporters import ImageExporter
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from behavysis.constants import VALUE2COLOR
from behavysis.df_classes.behav_df import BehavScoredDf
from behavysis.models.bouts import Bouts
from behavysis.models.experiment_configs import ExperimentConfigs
from behavysis.utils.qt_utils import qt2cv

if TYPE_CHECKING:
    from behavysis.viewer.windows.main import MainWindow


class GraphView(PlotWidget):
    main: MainWindow | None

    bars: dict[int, BarGraphItem]
    time_marker_line: InfiniteLine

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Linking reference to main window
        self.main = None
        # dict of BarGraphItem objects, indicating bout bars
        self.bars = {}
        # InfiniteLine object, indicating current time marker
        self.time_marker_line = InfiniteLine(pos=0, angle=90)
        # Brining time_marker_line to front
        self.time_marker_line.setZValue(10)

    def set_plot_attr(self, **kwargs):
        if "xmin" in kwargs and "xmax" in kwargs:
            self.setXRange(
                kwargs["xmin"],
                kwargs["xmax"],
                # padding=kwargs.get("xpadding", 0),  # type: ignore
            )
        if "ymin" in kwargs and "ymax" in kwargs:
            self.setYRange(
                kwargs["ymin"],
                kwargs["ymax"],
                # padding=kwargs.get("ypadding", 0),  # type: ignore
            )

    def plot_bouts_init(self, bouts: Bouts, configs: ExperimentConfigs):
        # Getting necessary configs
        fps = configs.auto.formatted_vid.fps
        # Getting data
        start_ls = np.array([i.start for i in bouts.bouts]) / fps
        stop_ls = np.array([i.stop for i in bouts.bouts]) / fps
        behavs_ls = np.array([i.behav for i in bouts.bouts])
        actual_ls = np.array([i.actual for i in bouts.bouts])
        # Plotting data
        self.plot_init(start_ls, stop_ls, behavs_ls, actual_ls)

    def plot_init(self, start_ls, stop_ls, behavs_ls, actual_ls, **kwargs):
        # Clearing plot and data
        self.clear()
        self.bars = {}
        # Setting current time marker line
        self.time_marker_line.setPos(0)
        self.addItem(self.time_marker_line)
        # Plotting bouts as bars
        behavs_ls_i, behavs_cat = pd.factorize(behavs_ls)
        for i, (start, stop, behav, actual) in enumerate(zip(start_ls, stop_ls, behavs_ls_i, actual_ls)):
            # Making bar item
            bar_ = BarGraphItem(
                x0=start,
                x1=stop,
                y=behav,
                height=0.5,
                brush=mkBrush(color=VALUE2COLOR[actual]),
            )
            # Adding double click event
            bar_.mouseDoubleClickEvent = lambda event, id_=i: self._on_bar_double_click(event, id_)
            # Storing in bars dict
            self.bars[i] = bar_
            # Adding to plot
            self.addItem(bar_)
        # Setting y-ticks to behavs_cat (categorical)
        self.getAxis("left").setTicks([[(i, v) for i, v in enumerate(behavs_cat)]])
        # Setting plot aesthetics
        self.set_plot_attr(**kwargs)

    def _on_bar_double_click(self, event, id_):
        if self.main is not None:
            # Getting index of bout with given `id_`
            index = self.main.bouts_model.index(id_)
            # Selecting this bout in bouts_view list
            self.main.ui.bouts_view.setCurrentIndex(index)

    def plot_update(self, i, **kwargs):
        # Plotting data
        self.time_marker_line.setPos(i)
        # Plot aesthetics
        self.set_plot_attr(**kwargs)

    def update_bar(self, id_: int, opts: dict):
        # Get old bar
        old_bar = self.bars[id_]
        # Make new bar
        new_bar = BarGraphItem(**{**old_bar.opts, **opts})
        # Ensure the new bar has the same mouseDoubleClickEvent
        new_bar.mouseDoubleClickEvent = old_bar.mouseDoubleClickEvent

        # Remove old bar from plot
        self.removeItem(old_bar)
        # Add new bar to plot
        self.addItem(new_bar)
        # Replace old bar with new bar
        self.bars[id_] = new_bar

    def plot2cv(self):
        # Making pyqtgraph image exporter to bytes
        exporter = ImageExporter(self.plotItem)
        # exporter.parameters()["width"] = self.width()
        # Exporting to QImage (bytes)
        img_qt = exporter.export(toBytes=True)
        assert isinstance(img_qt, QImage)
        # QImage to cv2 image
        img_cv = qt2cv(img_qt)
        # cv2 BGR to RGB
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        # Resize to widget size
        w, h = self.width(), self.height()
        img_cv = cv2.resize(img_cv, (w, h), interpolation=cv2.INTER_AREA)
        # Return cv2 image
        return img_cv


if __name__ == "__main__":
    app = QApplication(sys.argv)

    graph_viewer = GraphView()
    graph_viewer.setWindowTitle("CvView Test")
    graph_viewer.resize(800, 600)
    graph_viewer.move(100, 100)

    # fp = "/Users/timothylee/Desktop/Work/dev/behavysis/viewer/tests/resources/2_Round1.1_20220530_AGG-MOA_test3-M3_a2.parquet"

    behavs_df = BehavScoredDf.read(fp)
    # frames_df to bouts_dict
    bouts_dict = BehavScoredDf.frames2bouts(behavs_df)
    fps = 15
    # Updating graph_viewer
    start_ls = np.array([bout.start for bout in bouts_dict.bouts])
    stop_ls = np.array([bout.stop for bout in bouts_dict.bouts])
    behavs_ls = np.array([bout.behav for bout in bouts_dict.bouts])
    actual_ls = np.array([bout.actual for bout in bouts_dict.bouts])
    start_ls = start_ls / fps
    stop_ls = stop_ls / fps
    graph_viewer.plot_init(start_ls, stop_ls, behavs_ls, actual_ls)

    i = 15
    v = 5
    graph_viewer.plot_update(i, xmin=i - v, xmax=i + v)

    img_cv = graph_viewer.plot2cv()
    print(img_cv.shape)
    print(graph_viewer.height(), graph_viewer.width())
    print()

    graph_viewer.setFixedSize(200, 1000)

    img_cv = graph_viewer.plot2cv()
    print(img_cv.shape)
    print(graph_viewer.height(), graph_viewer.width())
    print()

    # cv2.imwrite("graph_viewer.png", img_cv)

    graph_viewer.show()
    sys.exit(app.exec())
