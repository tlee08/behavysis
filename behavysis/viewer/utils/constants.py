"""
_summary_
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage


VALUE2COLOR = {
    -1: "#BDBDBD",
    0: "#FF5252",
    1: "#69F0AE",
}
COLOR2VALUE = {v: k for k, v in VALUE2COLOR.items()}

VALUE2CHECKSTATE = {
    -1: Qt.CheckState.PartiallyChecked.value,
    0: Qt.CheckState.Unchecked.value,
    1: Qt.CheckState.Checked.value,
}
CHECKSTATE2VALUE = {v: k for k, v in VALUE2CHECKSTATE.items()}

QIMAGE_FORMAT = QImage.Format.Format_RGB888

STATUS_MSG_TIMEOUT = 5000
