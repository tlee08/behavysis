import os
from enum import Enum

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from behavysis.df_classes.df_mixin import DFMixin, FramesIN
from behavysis.df_classes.keypoints_df import CoordsCols

FBF = "fbf"


class AnalysisCN(Enum):
    INDIVIDUALS = "individuals"
    MEASURES = "measures"


class AnalysisDf(DFMixin):
    """__summary__"""

    NULLABLE = False
    IN = FramesIN
    CN = AnalysisCN

    @classmethod
    def make_location_scatterplot(
        cls,
        scatter_df: pd.DataFrame,
        corners_df: pd.DataFrame,
        dst_fp,
        measure: str,
    ):
        """
        Expects analysis_df index levels to be (frame,),
        and column levels to be (individual, measure).
        """
        scatter_stacked_df = scatter_df.stack(level="individuals").reset_index("individuals")
        g = sns.relplot(
            data=scatter_stacked_df,
            x=CoordsCols.X.value,
            y=CoordsCols.Y.value,
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
                x=CoordsCols.X.value,
                y=CoordsCols.Y.value,
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
        os.makedirs(os.path.dirname(dst_fp), exist_ok=True)
        g.savefig(dst_fp)
        g.figure.clf()

    @classmethod
    def make_location_scatterplot_a(
        cls,
        scatter_df: pd.DataFrame,
        corners_df: pd.DataFrame,
        dst_fp: str,
    ):
        """
        Expects analysis_df index levels to be (frame,),
        and column levels to be (individual, measure).
        """
        # Getting list of individuals and measures
        indivs_ls = scatter_df.columns.unique(cls.CN.INDIVIDUALS.value)
        roi_ls = scatter_df.columns.unique(cls.CN.MEASURES.value)
        roi_ls = roi_ls[np.isin(roi_ls, ["x", "y"], invert=True)]
        # "Looping" ROI bounding corners (to make closed polygons)
        corners_df = pd.concat(
            [corners_df, corners_df.groupby("roi").first().reset_index()],
            ignore_index=True,
        )
        # Rows are rois, columns are individuals
        ax_size = 5
        fig, axes = plt.subplots(
            nrows=roi_ls.shape[0],
            ncols=indivs_ls.shape[0],
            figsize=(ax_size * roi_ls.shape[0], ax_size * indivs_ls.shape[0]),
        )
        axes = axes.reshape(roi_ls.shape[0], indivs_ls.shape[0])
        # For each roi and indiv, plotting the bpts scatter and ROI polygon plots
        for i, roi in enumerate(roi_ls):
            for j, indiv in enumerate(indivs_ls):
                ax = axes[i, j]
                # bpts scatter plot
                sns.scatterplot(
                    data=scatter_df[indiv],
                    x=CoordsCols.X.value,
                    y=CoordsCols.Y.value,
                    hue=roi,
                    alpha=0.3,
                    linewidth=0,
                    marker=".",
                    s=5,
                    legend=None,
                    ax=ax,
                )
                # ROI polygon plot
                sns.lineplot(
                    data=corners_df[corners_df["roi"] == roi],
                    x=CoordsCols.X.value,
                    y=CoordsCols.Y.value,
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
                # Setting titles and labels
                ax.set_title(f"{roi} - {indiv}")
                ax.set_aspect("equal")
        # Saving fig
        os.makedirs(os.path.dirname(dst_fp), exist_ok=True)
        fig.savefig(dst_fp)
        fig.clf()
