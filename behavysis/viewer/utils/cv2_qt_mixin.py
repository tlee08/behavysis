import numpy as np
from PySide6.QtGui import QImage

from behavysis_viewer.utils.constants import QIMAGE_FORMAT


class Cv2QtMixin:
    @staticmethod
    def cv2qt(img_cv: np.ndarray) -> QImage:
        """Convert from an opencv image to QImage."""
        h, w, ch = img_cv.shape
        bpl = ch * w
        # cv2 to QImage and ensure it is in RGB888 format
        img_qt = QImage(img_cv.data, w, h, bpl, QIMAGE_FORMAT)
        # Return QImage
        return img_qt

    @staticmethod
    def qt2cv(img_qt: QImage) -> np.ndarray:
        """Convert from a QImage to an opencv image."""
        # QImage to RGB888 format
        img_qt = img_qt.convertToFormat(QIMAGE_FORMAT)
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
