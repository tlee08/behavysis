from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QShortcut
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget

from behavysis.viewer.ui.settings_ui import Ui_SettingsWindow
from behavysis.viewer.windows.window_mixin import WindowMixin

if TYPE_CHECKING:
    from behavysis.viewer.windows.main import MainWindow


class SettingsWindow(QTabWidget, WindowMixin):
    ui: Ui_SettingsWindow
    main: MainWindow

    def __init__(self, main: MainWindow, *args, **kwargs):
        # Instatiating QMainWindow
        super().__init__(*args, **kwargs)

        # Linking reference to main window
        self.main = main

        # Connecting main window with UI (from Qt-Designer)
        self.ui = Ui_SettingsWindow()
        self.ui.setupUi(self)

        # Setting window flags
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)

        # Connecting signals and slots
        self._init_conns()

    def _init_conns(self):
        # SIGNALS AND SLOTS: PREFERENCES
        # Save signals
        self.ui.confirm_dbtn.accepted.connect(self.save)
        QShortcut(Qt.Key.Key_Return, self).activated.connect(self.save)
        # Cancel signals
        self.ui.confirm_dbtn.rejected.connect(self.cancel)
        QShortcut(Qt.Key.Key_Escape, self).activated.connect(self.cancel)
        QShortcut(Qt.Key.Key_P, self).activated.connect(self.cancel)

    def cancel(self):
        self.toggle_window(self)

    def save(self):
        self.set_vid_size()
        self.set_fps()
        self.set_window_size()
        self.set_focus_size()

    def set_vid_size(self):
        width_str = self.ui.vid_width_le.text()
        if width_str.isnumeric():
            width = int(width_str)
            height = int(width * 2 / 3)
            # Setting new vid_viewer and plot_viewer dimensions
            self.main.resize_viewer(width, height)

    def set_fps(self):
        vid_speed_str = self.ui.vid_speed_le.text()
        try:
            self.main.vid_speed = float(vid_speed_str)
        except ValueError:
            pass

    def set_window_size(self):
        window_size_sec_str = self.ui.window_size_le.text()
        try:
            secs = float(window_size_sec_str)
            self.main.window_size_frames = int(secs * self.main.vid_model.fps)
        except ValueError:
            pass

    def set_focus_size(self):
        focus_size_sec_str = self.ui.focus_size_le.text()
        try:
            secs = float(focus_size_sec_str)
            self.main.focus_size_frames = int(secs * self.main.vid_model.fps)
        except ValueError:
            pass


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = SettingsWindow(QMainWindow())
    window.show()

    sys.exit(app.exec())
