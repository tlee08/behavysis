"""Functions have the following format."""

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from behavysis.constants import FBF, INDIVIDUALS, MEASURES, SINGLE, X, Y
from behavysis.df_classes import (
    AnalysisBinnedDf,
    AnalysisDf,
    BehaviourScoredDf,
    KeypointsDf,
)
from behavysis.models import ExperimentConfig


class AnalyseFunc(Protocol):
    """Protocol for analyse functions."""

    def __call__(
        self,
        keypoints_fp: Path,
        formatted_vid_fp: Path,
        dst_dir: Path,
        config_fp: Path,
    ) -> None:
        """Protocol for analyse functions."""


def in_roi(
    keypoints_fp: Path,
    formatted_vid_fp: Path,
    dst_dir: Path,
    config_fp: Path,
) -> None:
    """Determines frames where subject is inside ROI from average bpts.

    Points are `padding_px` padded (away) from center.
    """
    name = keypoints_fp.stem
    dst_subdir = dst_dir / "in_roi"
    # Calculating deltas (changes in body position) between each frame for the subject
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    analysis_config = config.get_analysis_config()
    start_frame = config.auto.start_frame
    config_filt_ls = config.user.analyse.in_roi
    # Loading in dataframe
    keypoints_df = KeypointsDf.clean_headings(KeypointsDf.read(keypoints_fp))
    assert keypoints_df.shape[0] > 0, (
        "No frames in keypoints_df. Please check keypoints file."
    )
    # Getting indivs list
    indivs, _ = KeypointsDf.get_indivs_bpts(keypoints_df)
    # Making analysis_df
    analysis_df_ls = []
    scatter_df_ls = []
    corners_df_ls = []
    roi_names_ls = []
    # For each roi, calculate the in-roi status of the subject
    idx = pd.IndexSlice
    for config_filt in config_filt_ls:
        # Getting necessary config parameters
        roi_name = config.get_ref(config_filt.roi_name)
        is_in = config.get_ref(config_filt.is_in)
        bpts = config.get_ref(config_filt.bodyparts)
        padding_mm = config.get_ref(config_filt.padding_mm)
        roi_corners = config.get_ref(config_filt.roi_corners)
        # Calculating more parameters
        padding_px = padding_mm / analysis_config.px_per_mm
        # Checking bodyparts and roi_corners exist
        KeypointsDf.check_bpts_exist(keypoints_df, bpts)
        KeypointsDf.check_bpts_exist(keypoints_df, roi_corners)
        # Getting average corner coordinates. This assumes ROI corners do not move.
        corners_i_df = pd.DataFrame(
            [keypoints_df[(SINGLE, pt)].mean() for pt in roi_corners]
        ).drop(columns=["likelihood"])
        # Adjusting x-y to have `padding_px` dilation/erosion from the points themselves
        roi_center = corners_i_df.mean()
        for i in corners_i_df.index:
            # Calculating angle from centre to point (going out from centre)
            theta = np.arctan2(
                corners_i_df.loc[i, Y] - roi_center[Y],
                corners_i_df.loc[i, X] - roi_center[X],
            )
            # Getting x, y distances so point is `padding_px` padded (away) from center
            corners_i_df.loc[i, X] = corners_i_df.loc[i, X] + (
                padding_px * np.cos(theta)
            )
            corners_i_df.loc[i, Y] = corners_i_df.loc[i, Y] + (
                padding_px * np.sin(theta)
            )
        # Making the res_df
        analysis_i_df = AnalysisDf.init_df(keypoints_df.index)
        # For each individual, getting the in-roi status
        for indiv in indivs:
            # Getting average body center (x, y) for each individual
            analysis_i_df[(indiv, X)] = (
                keypoints_df.loc[:, idx[indiv, bpts, X]].mean(axis=1).to_numpy()
            )
            analysis_i_df[(indiv, Y)] = (
                keypoints_df.loc[:, idx[indiv, bpts, Y]].mean(axis=1).to_numpy()
            )
            # Determining if the indiv body center is in the ROI
            analysis_i_df[(indiv, roi_name)] = analysis_i_df[indiv].apply(
                lambda pt, corners_i_df=corners_i_df: _pt_in_roi(pt, corners_i_df),
                axis=1,
            )
        # Inverting in_roi status if is_in is False
        if not is_in:
            analysis_i_df.loc[:, idx[:, roi_name]] = ~analysis_i_df.loc[
                :, idx[:, roi_name]
            ]
        analysis_df_ls.append(analysis_i_df.loc[:, idx[:, roi_name]].astype(np.int8))
        scatter_df_ls.append(analysis_i_df)
        corners_df_ls.append(corners_i_df)
        roi_names_ls.append(roi_name)
    # Concatenating all analysis_df_ls and roi_corners_df_ls
    analysis_df = pd.concat(analysis_df_ls, axis=1).T.drop_duplicates().T
    scatter_df = pd.concat(scatter_df_ls, axis=1).T.drop_duplicates().T
    corners_df = (
        pd.concat(corners_df_ls, keys=roi_names_ls, names=["roi"]).T.drop_duplicates().T
    )
    corners_df = corners_df.reset_index(level="roi")
    # Saving analysis_df
    fbf_fp = dst_subdir / FBF / f"{name}.{AnalysisDf.io_format}"
    AnalysisDf.write(analysis_df, fbf_fp)
    # Making scatter plot
    formatted_vid_cap = cv2.VideoCapture(formatted_vid_fp)
    # Getting 100th frame of video (arbitrary)
    for _ in range(start_frame + 100):
        ret, frame = formatted_vid_cap.read()
        if ret is False:
            logger.warning("Video shorter than start_frame")
            break
    # Getting scatter plot
    plot_fp = dst_subdir / "scatter_plot" / f"{name}.png"
    _make_location_scatterplot(scatter_df, corners_df, frame, plot_fp)
    # Summarising and binning analysis_df
    AnalysisBinnedDf.summary_binned_behaviour(
        analysis_df,
        dst_subdir,
        name,
        analysis_config.fps,
        analysis_config.bins_sec,
        analysis_config.custom_bins_sec,
    )


def _pt_in_roi(
    pt: pd.Series,
    corners_df: pd.DataFrame,
) -> bool:
    """Check if point is inside polygon using ray casting algorithm."""
    # Counting crossings over edge in region when point is translated to the right
    crossings = 0
    # To loop back to the first point at the end
    first_corner = pd.DataFrame(corners_df.iloc[0]).T
    corners_df = pd.concat((corners_df, first_corner), axis=0, ignore_index=True)
    # Making x and y aliases
    # For each edge
    for i in range(corners_df.shape[0] - 1):
        # Getting corner points of edge
        c1 = corners_df.iloc[i]
        c2 = corners_df.iloc[i + 1]
        # Getting whether point-y is between corners-y
        y_between = (c1[Y] > pt[Y]) != (c2[Y] > pt[Y])
        # Getting whether point-x is to the left (le) the intersection of corners-x
        x_left_of = pt[X] < (c2[X] - c1[X]) * (pt[Y] - c1[Y]) / (c2[Y] - c1[Y]) + c1[X]
        if y_between and x_left_of:
            crossings += 1
    # Odd number of crossings means point is in region
    return crossings % 2 == 1


def _make_location_scatterplot(
    scatter_df: pd.DataFrame,
    corners_df: pd.DataFrame,
    frame: np.ndarray,
    dst_fp: Path,
) -> None:
    """Make location scatterplot.

    Expects df index_levels=(frame,), column_levels=(individual, measure).
    """
    # Getting list of individuals and measures
    indivs_ls = scatter_df.columns.unique(INDIVIDUALS)
    roi_ls = scatter_df.columns.unique(MEASURES)
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
        figsize=(ax_size * indivs_ls.shape[0], ax_size * roi_ls.shape[0]),
    )
    axes = np.asarray(axes).reshape(roi_ls.shape[0], indivs_ls.shape[0])
    # For each roi and indiv, plotting the bpts scatter and ROI polygon plots
    for i, roi in enumerate(roi_ls):
        for j, indiv in enumerate(indivs_ls):
            ax: Axes = axes[i, j]
            # Adding frame image to plot
            ax.imshow(
                X=frame,
                alpha=0.5,
            )
            # bpts scatter plot
            sns.scatterplot(
                data=pd.DataFrame(scatter_df[indiv]),
                x=X,
                y=Y,
                hue=roi,
                palette={0: "orange", 1: "green"},
                alpha=0.3,
                linewidth=0,
                marker=".",
                s=5,
                legend=False,
                ax=ax,
            )
            # ROI polygon plot
            sns.lineplot(
                data=corners_df[corners_df["roi"] == roi],
                x=X,
                y=Y,
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
            # Setting axes characteristics
            ax.set_title(f"{roi} - {indiv}")
            ax.set_aspect("equal")
    # Saving fig
    dst_fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst_fp)
    fig.clf()


def _compute_movement(
    keypoints_df: pd.DataFrame,
    bpts: list[str],
    indivs: list[str],
    px_per_mm: float,
    smoothing_frames: int,
) -> pd.DataFrame:
    """Compute frame-by-frame movement distance for each individual.

    Returns DataFrame with columns (indiv, 'DistMM') and (indiv, 'DistMMSmoothed').
    """
    analysis_df = AnalysisDf.init_df(keypoints_df.index)
    idx = pd.IndexSlice

    # Smooth to reduce jitter contribution to movement
    jitter_frames = 3
    smoothed_xy_df = keypoints_df.rolling(
        window=jitter_frames, min_periods=1, center=True
    ).agg(np.nanmean)

    for indiv in indivs:
        # Getting changes in x-y values between frames
        delta_x = smoothed_xy_df.loc[:, idx[indiv, bpts, "x"]].mean(axis=1).diff()
        delta_y = smoothed_xy_df.loc[:, idx[indiv, bpts, "y"]].mean(axis=1).diff()
        delta_px = np.sqrt(np.power(delta_x, 2) + np.power(delta_y, 2))

        # Store distance in mm (raw and smoothed)
        analysis_df[(indiv, "DistMM")] = delta_px / px_per_mm
        analysis_df[(indiv, "DistMMSmoothed")] = (
            analysis_df[(indiv, "DistMM")]
            .rolling(window=smoothing_frames, min_periods=1, center=True)
            .agg(np.nanmean)
        )

    return analysis_df.bfill()


def speed(
    keypoints_fp: Path,
    formatted_vid_fp: Path,  # noqa: ARG001
    dst_dir: Path,
    config_fp: Path,
) -> None:
    """Determines the speed of the subject in each frame."""
    name = keypoints_fp.stem
    dst_subdir = dst_dir / "speed"

    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    analysis_config = config.get_analysis_config()
    config_filt = config.user.analyse.speed
    bpts = config.get_ref(config_filt.bodyparts)
    smoothing_sec = config.get_ref(config_filt.smoothing_sec)
    smoothing_frames = int(smoothing_sec * analysis_config.fps)

    keypoints_df = KeypointsDf.clean_headings(KeypointsDf.read(keypoints_fp))
    assert keypoints_df.shape[0] > 0, (
        "No frames in keypoints_df. Please check keypoints file."
    )
    KeypointsDf.check_bpts_exist(keypoints_df, bpts)
    indivs, _ = KeypointsDf.get_indivs_bpts(keypoints_df)

    # Compute movement and convert to speed (distance per second)
    analysis_df = _compute_movement(
        keypoints_df, bpts, indivs, analysis_config.px_per_mm, smoothing_frames
    )
    for indiv in indivs:
        analysis_df[(indiv, "SpeedMMperSec")] = (
            analysis_df[(indiv, "DistMM")] * analysis_config.fps
        )
        analysis_df[(indiv, "SpeedMMperSecSmoothed")] = (
            analysis_df[(indiv, "DistMMSmoothed")] * analysis_config.fps
        )
        # Remove distance columns - we only want speed
        analysis_df = analysis_df.drop(
            columns=[(indiv, "DistMM"), (indiv, "DistMMSmoothed")]
        )

    fbf_fp = dst_subdir / FBF / f"{name}.{AnalysisDf.io_format}"
    AnalysisDf.write(analysis_df, fbf_fp)

    AnalysisBinnedDf.summary_binned_quantitative(
        analysis_df,
        dst_subdir,
        name,
        analysis_config.fps,
        analysis_config.bins_sec,
        analysis_config.custom_bins_sec,
    )


def distance(
    keypoints_fp: Path,
    formatted_vid_fp: Path,  # noqa: ARG001
    dst_dir: Path,
    config_fp: Path,
) -> None:
    """Determines the distance travelled by the subject in each frame."""
    name = keypoints_fp.stem
    dst_subdir = dst_dir / "distance"

    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    analysis_config = config.get_analysis_config()
    config_filt = config.user.analyse.speed  # uses same config as speed
    bpts = config.get_ref(config_filt.bodyparts)
    smoothing_sec = config.get_ref(config_filt.smoothing_sec)
    smoothing_frames = int(smoothing_sec * analysis_config.fps)

    keypoints_df = KeypointsDf.clean_headings(KeypointsDf.read(keypoints_fp))
    assert keypoints_df.shape[0] > 0, (
        "No frames in keypoints_df. Please check keypoints file."
    )
    KeypointsDf.check_bpts_exist(keypoints_df, bpts)
    indivs, _ = KeypointsDf.get_indivs_bpts(keypoints_df)

    analysis_df = _compute_movement(
        keypoints_df, bpts, indivs, analysis_config.px_per_mm, smoothing_frames
    )

    fbf_fp = dst_subdir / FBF / f"{name}.{AnalysisDf.io_format}"
    AnalysisDf.write(analysis_df, fbf_fp)

    AnalysisBinnedDf.summary_binned_quantitative(
        analysis_df,
        dst_subdir,
        name,
        analysis_config.fps,
        analysis_config.bins_sec,
        analysis_config.custom_bins_sec,
    )


def social_distance(
    keypoints_fp: Path,
    formatted_vid_fp: Path,  # noqa: ARG001
    dst_dir: Path,
    config_fp: Path,
) -> None:
    """Determines the speed of the subject in each frame."""
    name = keypoints_fp.stem
    dst_subdir = dst_dir / "social_distance"
    # Calculating deltas (changes in body position) between each frame for the subject
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    analysis_config = config.get_analysis_config()
    config_filt = config.user.analyse.social_distance
    bpts = config.get_ref(config_filt.bodyparts)
    smoothing_sec = config.get_ref(config_filt.smoothing_sec)
    # Calculating more parameters
    smoothing_frames = int(smoothing_sec * analysis_config.fps)

    # Loading in dataframe
    keypoints_df = KeypointsDf.clean_headings(KeypointsDf.read(keypoints_fp))
    assert keypoints_df.shape[0] > 0, (
        "No frames in keypoints_df. Please check keypoints file."
    )
    # Checking body-centre bodypart exists
    KeypointsDf.check_bpts_exist(keypoints_df, bpts)
    # Getting indivs and bpts list
    indivs, _ = KeypointsDf.get_indivs_bpts(keypoints_df)

    # Calculating speed of subject for each frame
    analysis_df = AnalysisDf.init_df(keypoints_df.index)
    idx = pd.IndexSlice
    # Assumes there are only two individuals
    indiv_a = indivs[0]
    indiv_b = indivs[1]
    # Getting distances between each individual
    idx_a = idx[indiv_b, bpts, "x"]
    dist_x = (keypoints_df.loc[:, idx_a] - keypoints_df.loc[:, idx_a]).mean(axis=1)
    idx_b = idx[indiv_a, bpts, "y"]
    dist_y = (keypoints_df.loc[:, idx_b] - keypoints_df.loc[:, idx_b]).mean(axis=1)
    dist = np.array(np.sqrt(np.power(dist_x, 2) + np.power(dist_y, 2)))
    # Adding mm distance to saved analysis_df table
    analysis_df[(f"{indiv_a}_{indiv_b}", "DistMM")] = dist / analysis_config.px_per_mm
    analysis_df[(f"{indiv_a}_{indiv_b}", "DistMMSmoothed")] = (
        analysis_df[(f"{indiv_a}_{indiv_b}", "DistMM")]
        .rolling(window=smoothing_frames, min_periods=1, center=True)
        .agg(np.nanmean)
    )
    # Saving analysis_df
    fbf_fp = dst_subdir / FBF / f"{name}.{AnalysisDf.io_format}"
    AnalysisDf.write(analysis_df, fbf_fp)

    # Summarising and binning analysis_df
    AnalysisBinnedDf.summary_binned_quantitative(
        analysis_df,
        dst_subdir,
        name,
        analysis_config.fps,
        analysis_config.bins_sec,
        analysis_config.custom_bins_sec,
    )


def freezing(
    keypoints_fp: Path,
    formatted_vid_fp: Path,  # noqa: ARG001
    dst_dir: Path,
    config_fp: Path,
) -> None:
    """Determines the frames in which the subject is frozen.

    "Frozen" is defined as not moving outside of a radius of `threshold_mm`, and only
    includes bouts that last longer than `window_sec` spent seconds.

    NOTE: method is "greedy". Looks at a freezing bout from earliest possible frame.
    """
    name = keypoints_fp.stem
    dst_subdir = dst_dir / "freezing"
    # Calculating deltas (changes in body position) between each frame for the subject
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    analysis_config = config.get_analysis_config()
    config_filt = config.user.analyse.freezing
    bpts = config.get_ref(config_filt.bodyparts)
    thresh_mm = config.get_ref(config_filt.thresh_mm)
    smoothing_sec = config.get_ref(config_filt.smoothing_sec)
    window_sec = config.get_ref(config_filt.window_sec)
    # Calculating more parameters
    thresh_px = thresh_mm / analysis_config.px_per_mm
    smoothing_frames = int(smoothing_sec * analysis_config.fps)
    window_frames = int(np.round(analysis_config.fps * window_sec, 0))

    # Loading in dataframe
    keypoints_df = KeypointsDf.clean_headings(KeypointsDf.read(keypoints_fp))
    assert keypoints_df.shape[0] > 0, (
        "No frames in keypoints_df. Please check keypoints file."
    )
    # Checking body-centre bodypart exists
    KeypointsDf.check_bpts_exist(keypoints_df, bpts)
    # Getting indivs and bpts list
    indivs, _ = KeypointsDf.get_indivs_bpts(keypoints_df)

    # Calculating speed of subject for each frame
    analysis_df = AnalysisDf.init_df(keypoints_df.index)
    keypoints_df.index = analysis_df.index
    for indiv in indivs:
        temp_df = pd.DataFrame(index=analysis_df.index)
        # Calculating frame-by-frame delta distances for current bpt
        for bpt in bpts:
            # Getting x and y changes
            delta_x = keypoints_df[(indiv, bpt, "x")].diff()
            delta_y = keypoints_df[(indiv, bpt, "y")].diff()
            # Getting Euclidean distance between frames for bpt
            delta = np.sqrt(np.power(delta_x, 2) + np.power(delta_y, 2))
            # Converting from px to mm
            temp_df[f"{bpt}_dist"] = delta
            # Smoothing
            temp_df[f"{bpt}_dist"] = (
                temp_df[f"{bpt}_dist"]
                .rolling(window=smoothing_frames, min_periods=1, center=True)
                .agg(np.nanmean)
            )
        # If ALL bodypoints do not leave `thresh_px`
        analysis_df[(indiv, "freezing")] = temp_df.apply(
            lambda x: pd.Series(np.all(x < thresh_px)), axis=1
        ).astype(np.int8)

        # Getting start, stop, and duration of each freezing behav bout
        freezingbouts_df = BehaviourScoredDf.vect2bouts_df(
            analysis_df[(indiv, "freezing")] == 1
        )
        # For each freezing bout, if there is less than window_frames, tehn
        # it is not actually freezing
        for _, row in freezingbouts_df.iterrows():
            if row["dur"] < window_frames:
                analysis_df.loc[row["start"] : row["stop"], (indiv, "freezing")] = 0
    # Saving analysis_df
    fbf_fp = dst_subdir / FBF / f"{name}.{AnalysisDf.io_format}"
    AnalysisDf.write(analysis_df, fbf_fp)

    # Summarising and binning analysis_df
    AnalysisBinnedDf.summary_binned_behaviour(
        analysis_df,
        dst_subdir,
        name,
        analysis_config.fps,
        analysis_config.bins_sec,
        analysis_config.custom_bins_sec,
    )
