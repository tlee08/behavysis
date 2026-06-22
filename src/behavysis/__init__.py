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

import matplotlib.pyplot as plt
import seaborn as sns

from behavysis.behav_classifier import __all__ as behav_classifier_all
from behavysis.constants import PLOT_DPI, PLOT_STYLE
from behavysis.funcs import __all__ as funcs_all
from behavysis.pipeline import __all__ as pipeline_all
from behavysis.utils.logger_utils import configure_logger

#####################################################################
#              SETTING UP LOGGING
#####################################################################

configure_logger()

#####################################################################
#           INITIALISE MPL PLOTTING PARAMETERS
#####################################################################

sns.set_theme(style=PLOT_STYLE)

plt.rcParams["figure.dpi"] = PLOT_DPI
plt.rcParams["savefig.dpi"] = PLOT_DPI


#####################################################################
#          IMPORTING CLASSES
#####################################################################


__all__ = [  # noqa: PLE0604
    *behav_classifier_all,
    *pipeline_all,
    *funcs_all,
]
