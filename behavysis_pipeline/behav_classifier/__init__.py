"""
_summary_
"""

import logging

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from behavysis_core.constants import PLOT_DPI, PLOT_STYLE

from .behav_classifier import BehavClassifier

#####################################################################
#           INITIALISE MPL PLOTTING PARAMETERS
#####################################################################


# Makes graphs non-interactive (saves memory)
matplotlib.use("Agg")  # QtAgg

sns.set_theme(style=PLOT_STYLE)

plt.rcParams["figure.dpi"] = PLOT_DPI
plt.rcParams["savefig.dpi"] = PLOT_DPI

#####################################################################
#           SETTING UP LOGGING
#####################################################################

# logging.basicConfig(level=logging.INFO)
