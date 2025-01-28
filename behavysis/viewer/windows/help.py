from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QShortcut
from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout

from behavysis.viewer.windows.window_mixin import WindowMixin

if TYPE_CHECKING:
    from behavysis.viewer.windows.main import MainWindow


class HelpWindow(QDialog, WindowMixin):
    # ui: Ui_TabWidget
    main: MainWindow

    def __init__(self, main: MainWindow, *args, **kwargs):
        # Instatiating QMainWindow
        super().__init__(*args, **kwargs)

        # Linking reference to main window
        self.main = main

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self._add_label("<h1>Help Window</h1>")

        # Video hot-keys
        self._add_label("<h2>Video Hot-Keys</h2>")
        self._add_label("<b>Left:</b> Back")
        self._add_label("<b>Right:</b> Forward")
        self._add_label("<b>Space:</b> Start/Stop")
        self._add_label("<b>K:</b> Toggle Annotated Keypoints")
        # Bout-related hot-keys
        self._add_label("<h2>Bout-Related Hot-Keys</h2>")
        self._add_label("<b>R:</b> Replay Bout")
        self._add_label("<b>F:</b> Toggle Bout Focus")
        self._add_label("<b>1:</b> Toggle Is Behav")
        self._add_label("<b>2:</b> Toggle Not Behav")
        self._add_label("<b>3:</b> Toggle Select Behav")

        # Setting window flags
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)

        # Connecting signals and slots
        self._init_conns()

    def _add_label(self, text: str):
        label = QLabel(text)
        self.layout.addWidget(label)

    def _init_conns(self):
        # SIGNALS AND SLOTS
        # Close signals
        QShortcut(Qt.Key.Key_Escape, self).activated.connect(self.cancel)
        QShortcut(Qt.Key.Key_H, self).activated.connect(self.cancel)

    def cancel(self):
        self.toggle_window(self)
