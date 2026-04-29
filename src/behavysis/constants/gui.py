"""GUI and image constants."""

from PySide6 import QtCore, QtGui

VALUE2COLOR = {
    -1: "#BDBDBD",
    0: "#FF5252",
    1: "#69F0AE",
}
COLOR2VALUE = {v: k for k, v in VALUE2COLOR.items()}

VALUE2CHECKSTATE = {
    -1: QtCore.Qt.CheckState.PartiallyChecked.value,
    0: QtCore.Qt.CheckState.Unchecked.value,
    1: QtCore.Qt.CheckState.Checked.value,
}
CHECKSTATE2VALUE = {v: k for k, v in VALUE2CHECKSTATE.items()}

QIMAGE_FORMAT = QtGui.QImage.Format.Format_RGB888
STATUS_MSG_TIMEOUT = 5000
