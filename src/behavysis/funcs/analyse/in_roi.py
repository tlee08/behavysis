"""Analysis functions operating on Polars long-form keypoints DataFrames."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import polars as pl
from pydantic import BaseModel

from behavysis.constants import (
    BODYPART,
    DF_IO_FORMAT,
    FBF,
    FRAME,
    GROUP,
    LIKELIHOOD,
    MEASURE,
    VALUE,
    X,
    Y,
)
from behavysis.schemas import ANALYSIS_SCHEMA, write_df
from behavysis.transforms import (
    bodypart_avg_xy,
    check_bpts_exist,
    check_bpts_have_data,
    get_indivs_bpts,
)

from ._helper import AnalysisResult
from ._summary import summary_binned_behaviour

if TYPE_CHECKING:
    from behavysis.models import ExperimentConfig, ExperimentMetadata


class InRoiConfig(BaseModel):
    """InRoiConfig.

    ``roi_corners`` must be listed in polygon order (consecutive vertices, no
    self-intersections), as the point-in-polygon test assumes ordered corners.
    """

    roi_corners: list[str]
    bodyparts: list[str]
    roi_name: str
    padding_mm: float
    is_in: bool = True
    example_frame_index: int = 150


SPACING = 30
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 1
FONT_COLOR = (0, 0, 0)
GREEN = (0, 255, 0)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
POINT_RADIUS = 2
POINT_ALPHA = 0.4


def in_roi(
    config: ExperimentConfig,
    metadata: ExperimentMetadata,
    *,
    keypoints_df: pl.DataFrame,
    formatted_vid_fp: Path,
) -> list[AnalysisResult]:
    """Determines frames where subject is inside ROI from average bpts."""
    name = metadata.require_name()

    cfg_ls = config.require_analyse().require_list("in_roi", InRoiConfig)

    indivs, _ = get_indivs_bpts(keypoints_df)

    all_analysis_rows = []
    all_corners_rows = []
    roi_names = []
    avg_positions_by_roi_indiv: dict[tuple[str, str], pl.DataFrame] = {}

    for cfg in cfg_ls:
        roi_name = cfg.roi_name
        is_in = cfg.is_in
        bpts = cfg.bodyparts
        padding_mm = cfg.padding_mm
        roi_corners = cfg.roi_corners

        padding_px = padding_mm * metadata.require_px_per_mm()

        check_bpts_exist(keypoints_df, bpts)
        check_bpts_exist(keypoints_df, roi_corners)
        check_bpts_have_data(keypoints_df, bpts)
        check_bpts_have_data(keypoints_df, roi_corners)

        corners_rows = []
        for pt in roi_corners:
            avg = keypoints_df.filter(pl.col(BODYPART) == pt).select(
                pl.col(X).mean().alias(X),
                pl.col(Y).mean().alias(Y),
            )
            corners_rows.append(avg)

        corners_i = pl.concat(corners_rows)
        if LIKELIHOOD in corners_i.columns:
            corners_i = corners_i.drop(LIKELIHOOD)

        roi_center = corners_i.select(pl.col(X).mean(), pl.col(Y).mean())
        cx, cy = roi_center.row(0)

        adjusted = []
        for row in corners_i.iter_rows(named=True):
            theta = np.arctan2(row[Y] - cy, row[X] - cx)
            adjusted.append(
                {
                    X: row[X] + padding_px * np.cos(theta),
                    Y: row[Y] + padding_px * np.sin(theta),
                },
            )
        corners_i = pl.DataFrame(adjusted)

        for indiv in indivs:
            avg = bodypart_avg_xy(keypoints_df, indiv, bpts)

            xs = avg.select(X).to_series().to_numpy()
            ys = avg.select(Y).to_series().to_numpy()

            in_roi_mask = _pts_in_roi(xs, ys, corners_i)
            if not is_in:
                in_roi_mask = ~in_roi_mask

            avg = avg.with_columns(pl.Series(VALUE, in_roi_mask.astype(np.float64)))
            avg_positions_by_roi_indiv[(roi_name, indiv)] = avg

            for f, val in zip(
                avg.select(FRAME).to_series().to_numpy(),
                in_roi_mask,
                strict=True,
            ):
                all_analysis_rows.append(
                    {
                        FRAME: int(f),
                        GROUP: indiv,
                        MEASURE: roi_name,
                        VALUE: float(val),
                    },
                )

        all_corners_rows.extend(
            {
                "roi": roi_name,
                X: row[X],
                Y: row[Y],
            }
            for row in corners_i.iter_rows(named=True)
        )
        roi_names.append(roi_name)

    analysis_df = pl.DataFrame(all_analysis_rows, schema=ANALYSIS_SCHEMA)
    corners_df = pl.DataFrame(all_corners_rows)

    example_frame_index = cfg_ls[0].example_frame_index if cfg_ls else 150
    vid_frame = _get_frame(formatted_vid_fp, metadata, example_frame_index)
    scatter_img = _make_location_scatterplot(
        corners_df,
        avg_positions_by_roi_indiv,
        vid_frame,
        roi_names,
        indivs,
    )

    results = [
        AnalysisResult(
            relative_path=Path(FBF) / f"{name}.{DF_IO_FORMAT}",
            result=analysis_df,
            save_func=lambda fp, obj: write_df(obj, fp, ANALYSIS_SCHEMA),
        ),
        AnalysisResult(
            relative_path=Path("scatter_plot") / f"{name}.png",
            result=scatter_img,
            save_func=lambda fp, obj: cv2.imwrite(str(fp), obj),
        ),
        AnalysisResult(
            relative_path=Path("roi_corners") / f"{name}.{DF_IO_FORMAT}",
            result=corners_df,
            save_func=lambda fp, obj: write_df(
                obj, fp, {"roi": pl.Utf8, X: pl.Float64, Y: pl.Float64}
            ),
        ),
    ]
    results.extend(
        summary_binned_behaviour(
            analysis_df,
            name,
            metadata.require_fps(),
            config.require_analyse().bins_sec_ls,
            config.require_analyse().custom_bins_sec_ls,
        ),
    )
    return results


def _get_frame(
    vid_fp: Path, metadata: ExperimentMetadata, frame_index: int
) -> np.ndarray:
    """Extract specified frame for background plots, or black frame."""
    cap = cv2.VideoCapture(str(vid_fp))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return np.zeros(
            (metadata.require_height_px(), metadata.require_width_px(), 3),
            dtype=np.uint8,
        )
    return frame


def _make_location_scatterplot(
    corners_df: pl.DataFrame,
    avg_positions: dict[tuple[str, str], pl.DataFrame],
    bg_frame: np.ndarray,
    roi_names: list[str],
    indivs: list[str],
) -> np.ndarray:
    """Build a facet-grid scatter plot: rows=ROI, cols=individual.

    Each cell shows the background frame with:
    - ROI polygon (red outline)
    - Individual's bodypart positions colored green (in ROI) or orange (out)
    - Title: "ROI_name - individual"
    """
    n_rois = len(roi_names)
    n_indivs = len(indivs)
    fh, fw = bg_frame.shape[:2]

    canvas_h = n_rois * (fh + SPACING) + SPACING
    canvas_w = n_indivs * (fw + SPACING) + SPACING
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    for ri, roi_name in enumerate(roi_names):
        for ci, indiv in enumerate(indivs):
            y0 = SPACING + ri * (fh + SPACING)
            x0 = SPACING + ci * (fw + SPACING)

            cell = bg_frame.copy()
            labels = _draw_scatter_points(cell, avg_positions, indiv, roi_name)
            labels += _draw_roi_polygon(cell, corners_df, roi_name)

            canvas[y0 : y0 + fh, x0 : x0 + fw] = cell

            title = f"{roi_name} - {indiv}"
            text_x = x0 + 5
            text_y = y0 - 8 if ri == 0 else y0 - 5
            cv2.putText(
                canvas,
                title,
                (text_x, text_y),
                FONT,
                FONT_SCALE,
                FONT_COLOR,
                FONT_THICKNESS,
            )

    return canvas


def _draw_scatter_points(
    img: np.ndarray,
    avg_positions: dict[tuple[str, str], pl.DataFrame],
    indiv: str,
    roi_name: str,
) -> list[str]:
    """Draw scatter points for one individual colored by in/out status."""
    pos = avg_positions.get((roi_name, indiv))
    if pos is None:
        return []

    label_set = set()
    overlay = img.copy()
    for row in pos.iter_rows(named=True):
        is_in = row[VALUE] == 1
        color = GREEN if is_in else ORANGE
        label_set.add("In ROI" if is_in else "Out of ROI")
        cv2.circle(
            overlay,
            (int(row[X]), int(row[Y])),
            POINT_RADIUS,
            color,
            thickness=-1,
        )
    cv2.addWeighted(overlay, POINT_ALPHA, img, 1 - POINT_ALPHA, 0, dst=img)
    return list(label_set)


def _draw_roi_polygon(
    img: np.ndarray,
    corners_df: pl.DataFrame,
    roi_name: str,
) -> list[str]:
    """Draw ROI polygon outline."""
    roi_corners = corners_df.filter(pl.col("roi") == roi_name)
    if roi_corners.height == 0:
        return []
    pts = np.array(
        [[int(row[X]), int(row[Y])] for row in roi_corners.iter_rows(named=True)],
        dtype=np.int32,
    )
    cv2.polylines(img, [pts], isClosed=True, color=RED, thickness=2)
    return [roi_name]


def _pts_in_roi(
    px_arr: np.ndarray,
    py_arr: np.ndarray,
    corners_df: pl.DataFrame,
) -> np.ndarray:
    """Vectorized point-in-polygon using ray casting on numpy arrays.

    Assumes ``corners_df`` rows are in polygon order (consecutive vertices).
    """
    n = corners_df.height
    cx = corners_df.select(X).to_numpy()
    cy = corners_df.select(Y).to_numpy()

    crossings = np.zeros(len(px_arr), dtype=np.int32)
    for i in range(n):
        c1_x, c1_y = cx[i], cy[i]
        c2_x, c2_y = cx[(i + 1) % n], cy[(i + 1) % n]
        y_between = (c1_y > py_arr) != (c2_y > py_arr)
        if y_between.any():
            x_int = (c2_x - c1_x) * (py_arr[y_between] - c1_y) / (c2_y - c1_y) + c1_x
            crossings[y_between] ^= (px_arr[y_between] < x_int).astype(np.int32)
    return (crossings % 2) == 1
