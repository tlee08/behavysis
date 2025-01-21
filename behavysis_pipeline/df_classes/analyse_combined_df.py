"""
Functions have the following format:

Parameters
----------
dlc_fp : str
    The DLC dataframe filepath of the experiment to analyse.
ANALYSE_DIR : str
    The analysis directory path.
configs_fp : str
    the experiment's JSON configs file.

Returns
-------
str
    The outcome of the process.
"""

from __future__ import annotations

from enum import Enum

from behavysis_pipeline.df_classes.df_mixin import DFMixin, FramesIN

####################################################################################################
# DF CONSTANTS
####################################################################################################


class AnalyseCombinedCN(Enum):
    ANALYSIS = "analysis"
    INDIVIDUALS = "individuals"
    MEASURES = "measures"


####################################################################################################
# DF CLASS
####################################################################################################


class AnalyseCombinedDf(DFMixin):
    """__summary__"""

    NULLABLE = False
    IN = FramesIN
    CN = AnalyseCombinedCN
