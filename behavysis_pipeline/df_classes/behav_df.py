"""
Utility functions.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

from behavysis_pipeline.df_classes.df_mixin import DFMixin, FramesIN

# TODO: should we combine with BoutsDfMixin?

####################################################################################################
# DF CONSTANTS
####################################################################################################


class BehavColumns(Enum):
    PROB = "prob"
    PRED = "pred"
    ACTUAL = "actual"


class BehavCN(Enum):
    BEHAVIOURS = "behaviours"
    OUTCOMES = "outcomes"


####################################################################################################
# DF CLASS
####################################################################################################


class BehavDf(DFMixin):
    """
    Mixin for behaviour DF
    (generated from maDLC keypoint detection)
    functions.
    """

    NULLABLE = False
    IN = FramesIN
    CN = BehavCN

    @classmethod
    def update_behav(cls, df: pd.DataFrame, behav_src: str, behav_dst: str) -> pd.DataFrame:
        """
        Update the behaviour column with the given outcome and value.
        """
        # Getting columns
        columns = df.columns.to_frame(index=False)
        # Updating the behaviour column
        columns[cls.CN.BEHAVIOURS.value] = columns[cls.CN.OUTCOMES.value].map(
            lambda x: behav_dst if x == behav_src else x
        )
        # Setting the new columns
        df.columns = pd.MultiIndex.from_frame(columns)
        # Returning
        return df

    @classmethod
    def import_boris_tsv(cls, fp: str, behavs_ls: list[str], start_frame: int, stop_frame: int) -> pd.DataFrame:
        """
        Importing Boris TSV file.
        """
        # Making df structure
        df = cls.init_df(np.arange(start_frame, stop_frame))
        # Reading in corresponding BORIS tsv file
        df_boris = pd.read_csv(fp, sep="\t")
        # Initialising new classification columns based on
        # BORIS behavs and given `behavs_ls`
        # TODO: how to reconcile this with the behavs_ls?
        for behav in df_boris["Behavior"].unique():
            df[(behav, BehavColumns.ACTUAL.value)] = 0
            df[(behav, BehavColumns.PRED.value)] = 0
        for behav in behavs_ls:
            df[(behav, BehavColumns.ACTUAL.value)] = 0
            df[(behav, BehavColumns.PRED.value)] = 0
        # Setting the classification values from the BORIS file
        for ind, row in df_boris.iterrows():
            # Getting corresponding frame of this event point
            behav = row["Behavior"]
            frame = row["Image index"]
            status = row["Behavior type"]
            # Updating the classification in the scored df
            df.loc[frame:, (behav, BehavColumns.ACTUAL.value)] = status == "START"
            df.loc[frame:, (behav, BehavColumns.PRED.value)] = status == "START"
        # Setting dtype to int8
        df = df.astype(np.int8)
        return df
