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

import os
from enum import Enum

import pandas as pd
import seaborn as sns

from behavysis_pipeline.df_classes.df_mixin import DFMixin, FramesIN
from behavysis_pipeline.df_classes.keypoints_df import Coords

####################################################################################################
# DF CONSTANTS
####################################################################################################


FBF = "fbf"
SUMMARY = "summary"
BINNED = "binned"
CUSTOM = "custom"


class AnalyseCN(Enum):
    INDIVIDUALS = "individuals"
    MEASURES = "measures"


####################################################################################################
# DF CLASS
####################################################################################################


class AnalyseDf(DFMixin):
    """__summary__"""

    NULLABLE = False
    IN = FramesIN
    CN = AnalyseCN

    @classmethod
    def make_location_scatterplot(
        cls,
        scatter_df: pd.DataFrame,
        corners_df: pd.DataFrame,
        out_fp,
        measure: str,
    ):
        """
        Expects analysis_df index levels to be (frame,),
        and column levels to be (individual, measure).
        """
        scatter_stacked_df = scatter_df.stack(level="individuals").reset_index("individuals")
        g = sns.relplot(
            data=scatter_stacked_df,
            x=Coords.X.value,
            y=Coords.Y.value,
            hue=measure,
            col="individuals",
            kind="scatter",
            col_wrap=2,
            height=8,
            aspect=0.5 * scatter_stacked_df["individuals"].nunique(),
            alpha=0.8,
            linewidth=0,
            marker=".",
            s=10,
            legend=True,
        )
        # Invert the y axis
        g.axes[0].invert_yaxis()
        # Adding region definition (from roi_df) to the plot
        corners_df = pd.concat(
            [corners_df, corners_df.groupby("roi").first().reset_index()],
            ignore_index=True,
        )
        for ax in g.axes:
            sns.lineplot(
                data=corners_df,
                x=Coords.X.value,
                y=Coords.Y.value,
                hue="roi",
                # color=(1, 0, 0),
                linewidth=1,
                marker="+",
                markeredgecolor=(1, 0, 0),
                markeredgewidth=2,
                markersize=5,
                estimator=None,
                sort=False,
                legend=False,
                ax=ax,
            )
            ax.set_aspect("equal")
        # Setting fig titles and labels
        g.set_titles(col_template="{col_name}")
        g.figure.subplots_adjust(top=0.85)
        g.figure.suptitle("Spatial position", fontsize=12)
        # Saving fig
        os.makedirs(os.path.dirname(out_fp), exist_ok=True)
        g.savefig(out_fp)
        g.figure.clf()
