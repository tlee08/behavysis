"""This package is used to interpret lab mice behaviour using computer vision.

The package allows users to perform the entire analytics pipeline
from raw lab footage to
interpretable plotted and tabulated data for different analysises.
This pipeline includes:

- Formatting raw videos to a desired mp4 format (e.g. user defined fps and resolution)
- Performing stance detection on the mp4 file to generate an annotated mp4 file
    that tabulates the x-y coordinates of the subject's body points in each video frame.
    DeepLabCut is used to perform this.
- Preprocessing the coordinates file
- Extracting meaningful data analysis from the preprocessed coordinates file
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from behavysis.behav_classifier.behav_classifier import BehavClassifier
from behavysis.constants import PLOT_DPI, PLOT_STYLE

#####################################################################
#           IMPORTING SUBMODULES
#####################################################################
from behavysis.pipeline.project import Project
from behavysis.processes import (
    EvaluateVid,
    analyse_behavs,
    boris2behav,
    classify_behavs,
    combine_analysis,
    df2csv,
    df2df,
    distance,
    extract_features,
    format_vid,
    freezing,
    in_roi,
    interpolate,
    interpolate_stationary,
    ma_dlc_run_batch,
    ma_dlc_run_single,
    predictedbehavs2scoredbehavs,
    refine_ids,
    speed,
    start_stop_trim,
    stop_frame_from_likelihood,
    update_configs,
)
from behavysis.utils.logging_utils import setup_logging

#####################################################################
#              SETTING UP LOGGING
#####################################################################

setup_logging()

#####################################################################
#           INITIALISE MPL PLOTTING PARAMETERS
#####################################################################

# Makes graphs non-interactive (saves memory)
mpl.use("Agg")  # QtAgg

sns.set_theme(style=PLOT_STYLE)

plt.rcParams["figure.dpi"] = PLOT_DPI
plt.rcParams["savefig.dpi"] = PLOT_DPI


#####################################################################
#          IMPORTING CLASSES
#####################################################################

__all__ = [
    "Analyse",
    "BehavClassifier",
    "CalculateParams",
    "EvaluateVid",
    "Export",
    "Preprocess",
    "Project",
    "analyse_behavs",
    "boris2behav",
    "classify_behavs",
    "combine_analysis",
    "df2csv",
    "df2df",
    "distance",
    "extract_features",
    "format_vid",
    "freezing",
    "in_roi",
    "interpolate",
    "interpolate_stationary",
    "ma_dlc_run_batch",
    "ma_dlc_run_single",
    "predictedbehavs2scoredbehavs",
    "refine_ids",
    "speed",
    "start_stop_trim",
    "stop_frame_from_likelihood",
    "update_configs",
]
