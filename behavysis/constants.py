"""
_summary_
"""

import os
import pathlib
from enum import Enum

from PySide6 import QtCore, QtGui

####################################################################################################
# PIPELINE FOLDERS
####################################################################################################

DF_IO_FORMAT = "parquet"


class Folders(Enum):
    """Enum for the pipeline folders."""

    CONFIGS = "0_configs"
    RAW_VID = "1_raw_vid"
    FORMATTED_VID = "2_formatted_vid"
    KEYPOINTS = "3_keypoints"
    PREPROCESSED = "4_preprocessed"
    FEATURES_EXTRACTED = "5_features_extracted"
    PREDICTED_BEHAVS = "6_predicted_behavs"
    SCORED_BEHAVS = "7_scored_behavs"
    # ANALYSIS = "8_analysis"
    ANALYSIS_COMBINED = "9_analysis_combined"
    EVALUATE_VID = "10_evaluate_vid"


class FileExts(Enum):
    CONFIGS = "json"
    RAW_VID = "mp4"
    FORMATTED_VID = "mp4"
    KEYPOINTS = DF_IO_FORMAT
    PREPROCESSED = DF_IO_FORMAT
    FEATURES_EXTRACTED = DF_IO_FORMAT
    PREDICTED_BEHAVS = DF_IO_FORMAT
    SCORED_BEHAVS = DF_IO_FORMAT
    ANALYSIS_COMBINED = DF_IO_FORMAT
    EVALUATE_VID = "mp4"


# TODO: is there a better way to do the subsubdirs?
DIAGNOSTICS_DIR = "0_diagnostics"
ANALYSIS_DIR = "8_analysis"

CACHE_DIR = os.path.join(pathlib.Path.home(), ".behavysis_temp")


####################################################################################################
# DIAGNOSTICS CONSTANTS
####################################################################################################


STR_DIV = "".ljust(50, "-")


####################################################################################################
# PLOT CONSTANTS
####################################################################################################

PLOT_STYLE = "whitegrid"
PLOT_DPI = 75

####################################################################################################
# DEFAULT BODYPOINT CONSTANTS (FOR SIMBA FEATURE EXTRACTION)
####################################################################################################

INDIVS_SIMBA = [
    "mouse1marked",
    "mouse2unmarked",
]

BPTS_SIMBA = [
    "LeftEar",
    "RightEar",
    "Nose",
    "BodyCentre",
    "LeftFlankMid",
    "RightFlankMid",
    "TailBase1",
    "TailTip4",
]

BPTS_CENTRE = ["LeftFlankMid", "BodyCentre", "RightFlankMid", "LeftFlankRear", "RightFlankRear", "TailBase1"]

BPTS_FRONT = ["LeftEar", "RightEar", "Nose", "BodyCentre"]

INDIVS_SINGLE = "single"

BPTS_CORNERS = [
    "TopLeft",
    "TopRight",
    "BottomRight",
    "BottomLeft",
]

####################################################################################################
# GUI AND IMAGE CONSTANTS
####################################################################################################


VALUE2COLOR = {
    -1: "#BDBDBD",
    0: "#FF5252",
    1: "#69F0AE",
}
COLOR2VALUE = {v: k for k, v in VALUE2COLOR.items()}

# TODO: to enum with same values as BehavVals
VALUE2CHECKSTATE = {
    -1: QtCore.Qt.CheckState.PartiallyChecked.value,
    0: QtCore.Qt.CheckState.Unchecked.value,
    1: QtCore.Qt.CheckState.Checked.value,
}
CHECKSTATE2VALUE = {v: k for k, v in VALUE2CHECKSTATE.items()}

QIMAGE_FORMAT = QtGui.QImage.Format.Format_RGB888

STATUS_MSG_TIMEOUT = 5000
