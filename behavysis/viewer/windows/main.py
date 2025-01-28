from multiprocessing import Process

import cv2
import numpy as np
from behavysis_core.df_classes.bouts_df import BoutsDf
from behavysis_core.df_classes.df_mixin import DFMixin
from behavysis_core.pydantic_models.experiment_configs import ExperimentConfigs
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
)
from tqdm import trange

from behavysis.viewer.models.bout_inspect_list_model import BoutInspectListModel
from behavysis.viewer.models.bouts_list_model import BoutsListModel
from behavysis.viewer.models.exp_file_manager import ExpFileManager
from behavysis.viewer.models.keypoints_model import KeypointsModel
from behavysis.viewer.models.vid_model import VidModel
from behavysis.viewer.ui.main_ui import Ui_MainWindow
from behavysis.viewer.utils.constants import STATUS_MSG_TIMEOUT, VALUE2COLOR
from behavysis.viewer.widgets.graph_view import GraphView
from behavysis.viewer.windows.help import HelpWindow
from behavysis.viewer.windows.settings import SettingsWindow
from behavysis.viewer.windows.window_mixin import WindowMixin


class MainWindow(QMainWindow, WindowMixin):
    """__summary__"""

    ui: Ui_MainWindow
    preferences_window: SettingsWindow
    help_window: HelpWindow

    file_manager: ExpFileManager
    vid_model: VidModel
    keypoints_model: KeypointsModel

    _window_size_frames: int
    _vid_speed: int
    _focus_size_frames: int
    curr_i: int

    bouts_model: BoutsListModel
    bout_inspect_model: BoutInspectListModel

    def __init__(self, *args, **kwargs):
        # Instatiating QMainWindow
        super().__init__(*args, **kwargs)

        # Connecting main window with UI (from Qt-Designer)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Initialising settings window widget
        self.settings_window = SettingsWindow(main=self)
        # Initialising help window widget
        self.help_window = HelpWindow(main=self)

        # Initialising vid graph model
        self._init_models()

        # Grouping radio-buttons together in dict
        self.rbtns = {
            1: self.ui.is_behav_rbtn,
            0: self.ui.not_behav_rbtn,
            -1: self.ui.select_behav_rbtn,
        }

        self.resize_viewer(360, 240)

        # Initialising models and connecting to views
        self._init_vid_graph_widgets()
        self._init_models()
        self._init_model_views()

        # Connecting signals and slots
        self._init_conns_vid()
        self._init_conns_models()
        self._init_conns_views()
        self._init_conns_scoring()
        self._init_conns_io()
        self._init_conns_shortcuts()
        self._init_conns_hotkeys()

        # Initialising timer to play video
        self._init_timer_vid()

    ####################################################################
    # PROPERTIES
    ####################################################################

    @property
    def window_size_frames(self) -> int:
        """__summary__"""
        return self._window_size_frames

    @window_size_frames.setter
    def window_size_frames(self, window_size_frames: int) -> None:
        """__summary__"""
        self._window_size_frames = window_size_frames
        # Updating the frame plot with new viewable window size
        self.update_frame_plot()

    @property
    def vid_speed(self) -> int:
        """__summary__"""
        return self._vid_speed

    @vid_speed.setter
    def vid_speed(self, vid_speed: int) -> None:
        """__summary__"""
        self._vid_speed = vid_speed
        # Updating the timer with the new fps speed
        viewer_fps = self.vid_speed * self.vid_model.fps
        self.timer.start(1000 // viewer_fps)

    @property
    def focus_size_frames(self) -> int:
        """__summary__"""
        return self._focus_size_frames

    @focus_size_frames.setter
    def focus_size_frames(self, focus_size_frames: int) -> None:
        """__summary__"""
        self._focus_size_frames = focus_size_frames

    ####################################################################
    # INITIALIZATION METHODS
    ####################################################################

    def _init_vid_graph_widgets(self):
        """__summary__"""
        # Initialising video viewer and graph viewer
        self.ui.vid_viewer.main = self
        self.ui.graph_viewer.main = self

    def _init_models(self):
        """__summary__"""
        # Init file manager
        self.file_manager = ExpFileManager()
        # Init vid and keypoints models
        self.vid_model = VidModel()
        self.keypoints_model = KeypointsModel()
        # Init bouts and bout_inspect list models
        self.bouts_model = BoutsListModel()
        self.bout_inspect_model = BoutInspectListModel()
        # Init primitive attributes
        self._window_size_frames = 25
        self._vid_speed = 1
        self._focus_size_frames = 5
        self.curr_i = 0

    def _init_model_views(self):
        """__summary__"""
        # MODEL VIEWS
        # Linking BoutsModel
        self.ui.bouts_view.setModel(self.bouts_model)
        # Linking BoutInspectModel
        self.ui.bout_inspect_view.setModel(self.bout_inspect_model)

    def _init_conns_models(self):
        """__summary__"""
        # SIGNALS AND SLOTS: MODELS
        # bouts_model
        m = self.bouts_model
        # nil connections
        # bout_inspect_model
        m = self.bout_inspect_model
        # Making bout_inspect enabled when a bout is selected
        m.layoutChanged.connect(lambda: self.ui.bout_inspect_widget.setEnabled(m.id >= 0))
        # Making bout_inspect select user-defined behavs enabled when bout "actual" is True
        m.actual_signal.connect(lambda: self.ui.bout_inspect_view.setEnabled(m.bout.actual == 1))
        # Remaking bout's graph_viewer bar when bout's "actual" changes
        m.actual_signal.connect(lambda: self.ui.graph_viewer.update_bar(m.id, {"brush": VALUE2COLOR[m.bout.actual]}))

    def _init_conns_views(self):
        """__summary__"""
        # SIGNALS AND SLOTS: VIEWS
        # bouts_view
        v = self.ui.bouts_view.selectionModel()
        v.selectionChanged.connect(self.select_bout)
        # bout_inspect_view
        v = self.ui.bout_inspect_view.selectionModel()
        v.selectionChanged.connect(v.clearSelection)

    def _init_conns_scoring(self):
        """__summary__"""
        # SIGNALS AND SLOTS: SCORING BOUT
        # Choosing is_actual bout
        for k, v in self.rbtns.items():
            v.toggled.connect(lambda r, k=k: self.toggle_actual_rbtns(r, k))
        # Choosing specific behaviours
        # NOTE: already done in bout_inspect_model.setData, when CheckStateRole changes.

    def _init_conns_vid(self):
        """__summary__"""
        # SIGNALS AND SLOTS: VIDEO AND PLOT
        # Handle user moving slider
        self.ui.slider.sliderMoved.connect(self.set_frame)
        self.ui.slider.sliderPressed.connect(lambda: self.set_frame(self.ui.slider.value()))
        # Handle video buttons
        self.ui.start_stop_btn.toggled.connect(lambda i: self.timer.stop() if i else self.timer.start())
        self.ui.vid_back_btn.clicked.connect(lambda: self.set_frame(self.curr_i - self.vid_model.jump_size))
        self.ui.vid_fwd_btn.clicked.connect(lambda: self.set_frame(self.curr_i + self.vid_model.jump_size))
        # Handle bout-related video buttons
        self.ui.bout_replay_btn.clicked.connect(self.select_bout)

    def _init_conns_io(self):
        """__summary__"""
        # SIGNALS AND SLOTS: I/O
        # Handle opening settings and help windows
        self.ui.action_settings.triggered.connect(lambda: self.toggle_window(self.settings_window))
        self.ui.action_help.triggered.connect(lambda: self.toggle_window(self.help_window))
        # Handle opening
        self.ui.action_open.triggered.connect(self.open)
        # Handle saving
        self.ui.action_save.triggered.connect(self.save)
        self.ui.action_save_as_frames.triggered.connect(self.save_frames)
        self.ui.action_save_as_bouts.triggered.connect(self.save_bouts)
        # Handle exporting
        self.ui.action_export_video.triggered.connect(self.export_vid)
        # Handle open/closing windows
        self.ui.action_quit.triggered.connect(self.close)

    def _init_conns_shortcuts(self):
        """__summary__"""
        # SLOTS AND SIGNALS: KEYBOARD SHORTCUTS
        self.ui.action_open.setShortcut(QKeySequence.StandardKey.Open)
        self.ui.action_save.setShortcut(QKeySequence.StandardKey.Save)
        self.ui.action_quit.setShortcut(QKeySequence.StandardKey.Close)
        self.ui.action_settings.setShortcut(Qt.Key.Key_P)
        self.ui.action_help.setShortcut(Qt.Key.Key_H)

    def _init_conns_hotkeys(self):
        """__summary__"""
        # SLOTS AND SIGNALS: HOTKEYS
        # Handle video hot-keys
        QShortcut(Qt.Key.Key_Left, self).activated.connect(self.ui.vid_back_btn.click)
        QShortcut(Qt.Key.Key_Right, self).activated.connect(self.ui.vid_fwd_btn.click)
        QShortcut(Qt.Key.Key_Space, self).activated.connect(self.ui.start_stop_btn.toggle)
        QShortcut(Qt.Key.Key_K, self).activated.connect(self.ui.annot_keypts_cbx.toggle)
        # Handle bout-related hot-keys
        QShortcut(Qt.Key.Key_R, self).activated.connect(self.ui.bout_replay_btn.click)
        QShortcut(Qt.Key.Key_F, self).activated.connect(self.ui.bout_focus_btn.toggle)
        QShortcut(Qt.Key.Key_1, self).activated.connect(self.ui.is_behav_rbtn.toggle)
        QShortcut(Qt.Key.Key_2, self).activated.connect(self.ui.not_behav_rbtn.toggle)
        QShortcut(Qt.Key.Key_3, self).activated.connect(self.ui.select_behav_rbtn.toggle)

    def _init_timer_vid(self):
        """__summary__"""
        # TIMERS AND THREADS
        # Making timer to play video
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    ####################################################################
    # SLOTS
    ####################################################################

    def resize_viewer(self, width, height):
        """__summary__"""
        self.ui.vid_viewer.setFixedSize(width, height)
        self.ui.graph_viewer.setFixedWidth(width)
        self.ui.slider.setFixedWidth(width)

    def closeEvent(self, event):
        """__summary__"""
        # Perform actions when the window is about to close
        reply = QMessageBox.question(
            self,
            "Message",
            "Are you sure you want to close the window?",
            QMessageBox.StandardButton.Yes,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.settings_window.close()
            event.accept()
        else:
            event.ignore()

    def open(self, fp: str | None = None) -> None:
        """__summary__"""
        if not fp:
            fp = QFileDialog.getOpenFileName(self, "", "", "config file (*.json *.yaml)")[0]
        # Getting corresponding experiment filenames
        if not fp:
            self.ui.statusbar.showMessage("No video selected", timeout=STATUS_MSG_TIMEOUT)
            return
        try:
            # Loading filenames in vid file manager
            self.file_manager.load(fp)

            # Reading in configs
            configs = ExperimentConfigs.read_json(self.file_manager.configs_fp)
            # Loading data into vid, bouts, and keypoints models
            self.vid_model.load(self.file_manager.vid_fp)
            self.bouts_model.load(self.file_manager.behavs_df_fp, configs)
            self.keypoints_model.load(self.file_manager.dlc_df_fp, configs)

            # Setting primitive attributes
            # self.window_size_frames = 25
            self.vid_speed = 1
            # self.focus_size_frames = 5
            self.curr_i = 0

            # Preparing slider
            self.ui.slider.setMaximum(self.vid_model.nframes)
            # Updating the graph_viewer with bouts data
            self.ui.graph_viewer.plot_bouts_init(self.bouts_model.bouts, configs)

            # Writing msg to statusbar
            self.ui.statusbar.showMessage(f"Opened video: {fp}", timeout=STATUS_MSG_TIMEOUT)
        except ValueError as e:
            self.ui.statusbar.showMessage(f"Failed to open {fp}: {e}", timeout=STATUS_MSG_TIMEOUT)
            print(e)

    def select_bout(self):
        """__summary__"""
        # Getting selected bout QIndex (if there exists one)
        if len(self.ui.bouts_view.selectedIndexes()) > 0:
            # Getting selected bout QIndex
            index = self.ui.bouts_view.selectedIndexes()[0]
            # Getting bout df row
            bout = self.bouts_model.bouts.bouts[index.row()]
            # Loading in BoutInspectModel row
            self.bout_inspect_model.load(bout, index.row())
            # Setting bout inspect header text
            self.ui.bout_inspect_header.setText(f"{bout.behaviour} - {index.row()}")
            # Linking is_behav rbtns
            self.rbtns[bout.actual].toggle()
            # Jumping to bout start in video
            self.set_frame(bout.start - self.focus_size_frames)
            # Jumping to bout in bouts_view list
            self.ui.bouts_view.scrollTo(index)

    # def graph_viewer_select_bout(self, e, id_):
    #     # Getting index of bout with given `id_`
    #     idx = self.bouts_model.index(id_)
    #     # Selecting this bout in bouts_view list
    #     self.ui.bouts_view.setCurrentIndex(idx)
    #     # Jumping to that bout in bouts_view list
    #     self.ui.bouts_view.scrollTo(idx)

    def update_frame(self):
        """__summary__"""
        # If bout_focus_btn is checked AND video is past the current bout's end,
        # then pause (i.e., don't update the frame)
        if self.ui.bout_focus_btn.isChecked():
            if self.curr_i > self.bout_inspect_model.stop + self.focus_size_frames:
                return
        # Update vid frame
        ret = self.update_frame_vid()
        if ret:
            # Update plot
            self.update_frame_plot()
            # Update slider
            self.ui.slider.setValue(self.curr_i)
            # Update curr_frame value
            self.curr_i += 1

    def update_frame_vid(self):
        """__summary__"""
        # Reading in next frame
        ret, frame = self.vid_model.read()
        if ret:
            # Checking whether to annotate keypoints
            if self.ui.annot_keypts_cbx.isChecked():
                # Updating frame with annotated keypoints
                frame = self.keypoints_model.annot_keypoints(frame, self.curr_i)
            # Displaying frame on cv_viewer
            self.ui.vid_viewer.display_cv2(frame)
        return ret

    def update_frame_plot(self):
        """__summary__"""
        self.ui.graph_viewer.plot_update(
            self.curr_i / self.vid_model.fps,
            xmin=(self.curr_i - self.window_size_frames) / self.vid_model.fps,
            xmax=(self.curr_i + self.window_size_frames) / self.vid_model.fps,
        )

    def set_frame(self, frame_num):
        """__summary__"""
        # Setting curr_frame value
        self.curr_i = np.clip(frame_num, 0, self.vid_model.nframes - 1)
        # Setting video to new frame
        self.vid_model.vid.set(cv2.CAP_PROP_POS_FRAMES, self.curr_i)
        # Updating video and plot view
        self.update_frame()

    def toggle_actual_rbtns(self, r: bool, value: int):
        """__summary__"""
        if r:
            self.bout_inspect_model.actual = value

    def save(self):
        """__summary__"""
        self.save_frames(self.file_manager.behavs_df_fp)

    def save_frames(self, fp=None):
        """__summary__"""
        if not fp:
            fp = QFileDialog.getSaveFileName(
                self,
                "",
                f"{self.file_manager.name}.feather",
                "feather dataframe (*.feather)",
            )[0]
        if fp:
            # bouts to behavs_df
            behavs_df = BoutsDf.bouts2frames(self.bouts_model.bouts)
            # Writing to feather file
            DFMixin.write_feather(behavs_df, fp)
            self.ui.statusbar.showMessage(f"Saved scored behaviour frames to {fp}", timeout=STATUS_MSG_TIMEOUT)

    def save_bouts(self, fp=None):
        """__summary__"""
        if not fp:
            fp = QFileDialog.getSaveFileName(
                self,
                "",
                f"{self.file_manager.name}.json",
                "json file (*.json *.yaml)",
            )[0]
        if fp:
            # Writing to json file
            self.bouts_model.bouts.write_json(fp)
            self.ui.statusbar.showMessage(f"Saved scored behaviour bouts to {fp}", timeout=STATUS_MSG_TIMEOUT)

    def export_vid(self, fp=None):
        """__summary__"""
        if not fp:
            fp = QFileDialog.getSaveFileName(
                self,
                "",
                f"{self.file_manager.name}.mp4",
                "video (*.mp4)",
            )[0]
        if fp:
            # Annotating each frame using the created functions
            p = Process(
                target=self.export_vid_worker,
                args=(
                    self.file_manager,
                    fp,
                    self.ui.vid_viewer.width(),
                    self.ui.vid_viewer.height(),
                    self.window_size_frames,
                ),
            )

            p.start()
            p.join()

            # Displaying message
            self.ui.statusbar.showMessage(f"Exported video to {fp}", timeout=STATUS_MSG_TIMEOUT)

    @staticmethod
    def export_vid_worker(
        file_manager: ExpFileManager,
        out_fp: str,
        w: int,
        h: int,
        window_size: int,
    ):
        """Annotating video with keypoints and scored behaviours loop."""
        # Need to make QApplication for graph_viewer
        app = QApplication()

        # Get configs
        configs = ExperimentConfigs.read_json(file_manager.configs_fp)
        # Make video model
        vid_model = VidModel()
        vid_model.load(file_manager.vid_fp)
        # Make keypoints model
        keypoints_model = KeypointsModel()
        keypoints_model.load(file_manager.dlc_df_fp, configs)
        # Make bouts model
        bouts_model = BoutsListModel()
        bouts_model.load(file_manager.behavs_df_fp, configs)
        # Make graph viewer, plot all data and set widget size
        graph_viewer = GraphView()
        graph_viewer.plot_bouts_init(bouts_model.bouts, configs)
        graph_viewer.setFixedSize(w, h)
        # Must run to enable widget size
        graph_viewer.show()
        graph_viewer.hide()

        # Create VideoWriter object
        out_cap = cv2.VideoWriter(
            out_fp,
            cv2.VideoWriter_fourcc(*"mp4v"),
            vid_model.fps,
            (w, h * 2),
        )
        # Annotating each frame using the created functions
        for i in trange(vid_model.nframes):
            # for _ in np.arange(vid_model.nframes):
            # Reading in next frame
            ret, frame = vid_model.read()
            if ret is False:
                break
            # Add keypoints to frame
            frame = keypoints_model.annot_keypoints(frame, i)
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            # Making graph frame
            graph_viewer.plot_update(
                i / vid_model.fps,
                xmin=(i - window_size) / vid_model.fps,
                xmax=(i + window_size) / vid_model.fps,
            )
            graph_frame = graph_viewer.plot2cv()
            # Writing annotated frame to the VideoWriter
            out_cap.write(np.concatenate((frame, graph_frame), axis=0))
        # Release the VideoWriter (i.e. save)
        out_cap.release()
        # Close the QApplication
        app.quit()
