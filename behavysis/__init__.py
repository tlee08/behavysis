"""
This package is used to interprets and interprets lab mice behaviour using computer vision.
The package allows users to perform the entire analytics pipeline from raw lab footage to
interpretable plotted and tabulated data for different analysises. This pipeline includes:

- Formatting raw videos to a desired mp4 format (e.g. user defined fps and resolution)
- Performing stance detection on the mp4 file to generate an annotated mp4 file and file that tabulates the x-y coordinates of the subject's body points in each video frame. DeepLabCut is used to perform this.
- Preprocessing the coordinates file
- Extracting meaningful data analysis from the preprocessed coordinates file
"""

import warnings

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from behavysis.behav_classifier.behav_classifier import BehavClassifier
from behavysis.constants import PLOT_DPI, PLOT_STYLE

#####################################################################
#           IMPORTING SUBMODULES
#####################################################################
from behavysis.pipeline.project import Project
from behavysis.processes.analyse import Analyse
from behavysis.processes.calculate_params import CalculateParams
from behavysis.processes.export import Export
from behavysis.processes.preprocess import Preprocess

#####################################################################
#               FILTERING STDOUT WARNINGS
#####################################################################

warnings.filterwarnings("ignore")

#####################################################################
#           INITIALISE MPL PLOTTING PARAMETERS
#####################################################################


# Makes graphs non-interactive (saves memory)
matplotlib.use("Agg")  # QtAgg

sns.set_theme(style=PLOT_STYLE)

plt.rcParams["figure.dpi"] = PLOT_DPI
plt.rcParams["savefig.dpi"] = PLOT_DPI
