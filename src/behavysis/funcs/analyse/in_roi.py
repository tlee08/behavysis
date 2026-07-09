"""Analysis functions operating on Polars long-form keypoints DataFrames."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import polars as pl
from pydantic import BaseModel, PositiveFloat

from behavysis.constants import DF_IO_FORMAT, FBF
from behavysis.funcs.analyse._helper import _bodypart_avg_xy
from behavysis.models import AnalysisResult
from behavysis.schemas import ANALYSIS_SCHEMA, write_df
from behavysis.transforms.analysis import summary_binned_behaviour
from behavysis.transforms.keypoint import check_bpts_exist, get_indivs_bpts

if TYPE_CHECKING:
    from behavysis.models import ExperimentConfig, ExperimentMetadata


class InRoiConfig(BaseModel):
    """InRoiConfig."""

    roi_corners: list[str]
    bodyparts: list[str]
    roi_name: str
    is_in: bool = True
    padding_mm: PositiveFloat = 0.0


SPACING = 30
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 1
FONT_COLOR = (0, 0, 0)
GREEN = (0, 255, 0)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
POINT_RADIUS = 2


def in_roi(
    keypoints_df: pl.DataFrame,
    vid_frame: np.ndarray,
    config: ExperimentConfig,
    metadata: ExperimentMetadata,
) -> list[AnalysisResult]:
    """Determines frames where subject is inside ROI from average bpts."""
    name = metadata.require_name()

    cfg_ls = config.require_analyse().require_list("in_roi", InRoiConfig)

    indivs, _ = get_indivs_bpts(keypoints_df)

    all_analysis_rows = []
    all_corners_rows = []
    roi_names = []
    avg_positions_by_indiv: dict[str, pl.DataFrame] = {}

    for cfg in cfg_ls:
        roi_name = cfg.roi_name
        is_in = cfg.is_in
        bpts = cfg.bodyparts
        padding_mm = cfg.padding_mm
        roi_corners = cfg.roi_corners

        padding_px = padding_mm / metadata.require_px_per_mm()

        check_bpts_exist(keypoints_df, bpts)
        check_bpts_exist(keypoints_df, roi_corners)

        corners_rows = []
        for pt in roi_corners:
            avg = keypoints_df.filter(pl.col("bodypart") == pt).select(
                pl.col("x").mean().alias("x"),
                pl.col("y").mean().alias("y"),
            )
            corners_rows.append(avg)

        corners_i = pl.concat(corners_rows)
        if "likelihood" in corners_i.columns:
            corners_i = corners_i.drop("likelihood")

        roi_center = corners_i.select(pl.col("x").mean(), pl.col("y").mean())
        cx, cy = roi_center.row(0)

        adjusted = []
        for row in corners_i.iter_rows(named=True):
            theta = np.arctan2(row["y"] - cy, row["x"] - cx)
            adjusted.append(
                {
                    "x": row["x"] + padding_px * np.cos(theta),
                    "y": row["y"] + padding_px * np.sin(theta),
                },
            )
        corners_i = pl.DataFrame(adjusted)

        for indiv in indivs:
            avg = _bodypart_avg_xy(keypoints_df, indiv, bpts)
            avg_positions_by_indiv[indiv] = avg

            frames = avg.select("frame").to_series().to_numpy()
            xs = avg.select("x").to_series().to_numpy()
            ys = avg.select("y").to_series().to_numpy()

            in_roi_mask = _pts_in_roi(xs, ys, corners_i)
            if not is_in:
                in_roi_mask = ~in_roi_mask

            for f, val in zip(frames, in_roi_mask, strict=True):
                all_analysis_rows.append(
                    {
                        "frame": int(f),
                        "individual": indiv,
                        "measure": roi_name,
                        "value": float(val),
                    },
                )

        all_corners_rows.extend(
            {
                "roi": roi_name,
                "x": row["x"],
                "y": row["y"],
            }
            for row in corners_i.iter_rows(named=True)
        )
        roi_names.append(roi_name)

    analysis_df = pl.DataFrame(all_analysis_rows, schema=ANALYSIS_SCHEMA)
    corners_df = pl.DataFrame(all_corners_rows)

    scatter_img = _make_location_scatterplot(
        analysis_df,
        corners_df,
        avg_positions_by_indiv,
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


def _make_location_scatterplot(
    analysis_df: pl.DataFrame,
    corners_df: pl.DataFrame,
    avg_positions: dict[str, pl.DataFrame],
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
            labels = _draw_scatter_points(
                cell, analysis_df, avg_positions, indiv, roi_name
            )
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
    analysis_df: pl.DataFrame,
    avg_positions: dict[str, pl.DataFrame],
    indiv: str,
    roi_name: str,
) -> list[str]:
    """Draw scatter points for one individual colored by in/out status."""
    if indiv not in avg_positions:
        return []
    pos = avg_positions[indiv]

    in_roi_mask = (
        analysis_df.filter(
            pl.col("individual") == indiv,
            pl.col("measure") == roi_name,
        )
        .sort("frame")
        .select("value")
        .to_series()
        .to_numpy()
    )

    frames_pos = pos.select("frame").to_series().to_numpy()
    xs = pos.select("x").to_series().to_numpy()
    ys = pos.select("y").to_series().to_numpy()

    label_set = set()
    for i in range(len(xs)):
        f = frames_pos[i]
        if f >= len(in_roi_mask):
            break
        color = GREEN if in_roi_mask[f] == 1 else ORANGE
        label_set.add("In ROI" if in_roi_mask[f] == 1 else "Out of ROI")
        cv2.circle(img, (int(xs[i]), int(ys[i])), POINT_RADIUS, color, thickness=-1)
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
        [[int(row["x"]), int(row["y"])] for row in roi_corners.iter_rows(named=True)],
        dtype=np.int32,
    )
    cv2.polylines(img, [pts], isClosed=True, color=RED, thickness=2)
    return [roi_name]


def _pt_in_roi(pt_x: float, pt_y: float, corners_df: pl.DataFrame) -> bool:
    """Check if point is inside polygon using ray casting algorithm."""
    crossings = 0
    n = corners_df.height
    for i in range(n):
        c1 = corners_df.row(i, named=True)
        c2 = corners_df.row((i + 1) % n, named=True)
        y_between = (c1["y"] > pt_y) != (c2["y"] > pt_y)
        if y_between:
            x_int = (c2["x"] - c1["x"]) * (pt_y - c1["y"]) / (c2["y"] - c1["y"]) + c1[
                "x"
            ]
            if pt_x < x_int:
                crossings += 1
    return crossings % 2 == 1


def _pts_in_roi(
    px_arr: np.ndarray,
    py_arr: np.ndarray,
    corners_df: pl.DataFrame,
) -> np.ndarray:
    """Vectorized point-in-polygon using ray casting on numpy arrays."""
    n = corners_df.height
    cx = corners_df.select("x").to_numpy()
    cy = corners_df.select("y").to_numpy()

    crossings = np.zeros(len(px_arr), dtype=np.int32)
    for i in range(n):
        c1_x, c1_y = cx[i], cy[i]
        c2_x, c2_y = cx[(i + 1) % n], cy[(i + 1) % n]
        y_between = (c1_y > py_arr) != (c2_y > py_arr)
        if y_between.any():
            x_int = (c2_x - c1_x) * (py_arr[y_between] - c1_y) / (c2_y - c1_y) + c1_x
            crossings[y_between] ^= (px_arr[y_between] < x_int).astype(np.int32)
    return (crossings % 2) == 1
