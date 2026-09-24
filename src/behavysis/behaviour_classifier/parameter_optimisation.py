"""Model Parameter Post-Processing Optimisation."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from sklearn.metrics import recall_score

from behavysis.constants import (
    BOUT_ID,
    EXPERIMENT,
    FRAME,
    PRED,
    PROB,
)
from behavysis.transforms import hysteresis, label_bouts, smooth_pred_bout, smooth_prob

from .config import (
    FRAME_AWARE,
    HYSTERESIS,
    FrameAwarePostprocessing,
    HysteresisPostprocessing,
    ModelRecipe,
)
from .data import ACTUAL, agg_eval_df_by_bouts

if TYPE_CHECKING:
    from collections.abc import Callable


_SMOOTHING_GRID = (0, 1, 2, 3, 5, 8, 12)
# Data-driven bounds (hind-paw-withdrawal labels): real bouts are >= 12 frames and
# distinct bouts are >= 7 frames apart (p5). The merge-aware objective below (hidden
# events first) also stops the sweep from merging distinct withdrawals, so these grids
# can be generous.
_MIN_GAP_GRID = (0, 2, 4, 8, 16)
_MIN_BOUT_GRID = (0, 2, 4, 8)
_LOW_THRESHOLD_GRID = (0.0, 0.001, 0.005, 0.01, 0.02, 0.03)
_N_PCUTOFF = 20
_MIN_PCUTOFF = 1e-6


def _bout_recall(pred_df: pl.DataFrame) -> float:
    """Fraction of real bouts covered by at least one predicted frame."""
    bouts_df = agg_eval_df_by_bouts(pred_df)
    return float(recall_score(bouts_df[ACTUAL], bouts_df[PRED], zero_division=0))


def _count_pred_bouts(pred_df: pl.DataFrame) -> int:
    """Number of contiguous predicted-positive runs (the review burden)."""
    return int(
        label_bouts(pred_df, PRED)
        .filter(pl.col(PRED) == 1)
        .get_column(BOUT_ID)
        .n_unique()
    )


def _count_merge_hidden(pred_df: pl.DataFrame) -> int:
    """Total events hidden by merging.

    A predicted bout spanning ``k`` real bouts counts as one review but hides
    ``k - 1`` events (the reviewer sees only one). Summed over all predicted bouts.
    """
    df = pred_df.sort([EXPERIMENT, FRAME]).with_columns(
        pl.struct([EXPERIMENT, ACTUAL]).rle_id().alias("_rb"),
        pl.struct([EXPERIMENT, PRED]).rle_id().alias("_pb"),
    )
    per = (
        df.filter(pl.col(PRED) == 1)
        .group_by("_pb")
        .agg(pl.col("_rb").filter(pl.col(ACTUAL) == 1).n_unique().alias("n_real"))
    )
    n_real = per.get_column("n_real").cast(pl.Int64)
    return int((n_real - 1).clip(lower_bound=0).sum())


def _review_cost(pred_df: pl.DataFrame) -> tuple[int, int]:
    """Review cost ``(hidden events, predicted bouts)``, compared lexicographically.

    Minimises merges first, then the number of predicted bouts. So merging
    distinct events (which hides them) is forbidden, while merging fragments or
    noise (which hides nothing) is still rewarded.
    """
    return (_count_merge_hidden(pred_df), _count_pred_bouts(pred_df))


def _select_pcutoff(
    make_pred: Callable[[float], pl.DataFrame],
    pcutoffs: np.ndarray,
    target_recall: float,
) -> tuple[float, tuple[int, int]] | None:
    """Highest pcutoff whose bout recall >= target, and its review cost.

    Returns ``None`` if no pcutoff reaches the target recall.
    """
    best_pcutoff = float(pcutoffs[0])
    best_cost: tuple[int, int] | None = None
    for pcutoff in pcutoffs:
        pred_df = make_pred(float(pcutoff))
        if _bout_recall(pred_df) < target_recall:
            continue
        best_pcutoff = float(pcutoff)
        best_cost = _review_cost(pred_df)
    if best_cost is None:
        return None
    return best_pcutoff, best_cost


def _frame_aware_pred(
    smoothed: pl.DataFrame,
    min_gap: int,
    min_bout: int,
    pcutoff: float,
) -> pl.DataFrame:
    """Frame-aware post-processing for a given pcutoff."""
    pred = smoothed.with_columns((pl.col(PROB) > pcutoff).cast(pl.Int64).alias(PRED))
    return smooth_pred_bout(pred, min_gap=min_gap, min_bout=min_bout)


def _hysteresis_pred(
    raw: pl.DataFrame,
    low_threshold: float,
    min_bout: int,
    pcutoff: float,
) -> pl.DataFrame:
    """Hysteresis post-processing for a given pcutoff."""
    pred = hysteresis(raw, pcutoff=pcutoff, low_threshold=low_threshold)
    return smooth_pred_bout(pred, min_gap=0, min_bout=min_bout)


def _optimise_frame_aware(
    raw: pl.DataFrame,
    pcutoffs: np.ndarray,
    target_recall: float,
) -> FrameAwarePostprocessing:
    """Sweep (smoothing, min_gap, min_bout, pcutoff) for the frame-aware step."""
    best_cost: tuple[int, int] | None = None
    best: tuple[int, int, int, float] | None = None
    for smoothing_frames in _SMOOTHING_GRID:
        smoothed = smooth_prob(
            raw, smoothing_frames=smoothing_frames, agg_func="median"
        )
        for min_gap in _MIN_GAP_GRID:
            for min_bout in _MIN_BOUT_GRID:
                result = _select_pcutoff(
                    partial(_frame_aware_pred, smoothed, min_gap, min_bout),
                    pcutoffs,
                    target_recall,
                )
                if result is None:
                    continue
                pcutoff, cost = result
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best = (smoothing_frames, min_gap, min_bout, pcutoff)
    if best is None:
        msg = "No frame-aware parameters reach the target recall."
        raise ValueError(msg)
    smoothing_frames, min_gap, min_bout, pcutoff = best
    return FrameAwarePostprocessing(
        pcutoff=max(float(pcutoff), _MIN_PCUTOFF),
        smoothing_frames=smoothing_frames,
        min_gap_frames=min_gap,
        min_bout_frames=min_bout,
    )


def _optimise_hysteresis(
    raw: pl.DataFrame,
    pcutoffs: np.ndarray,
    target_recall: float,
) -> HysteresisPostprocessing:
    """Sweep (low_threshold, min_bout, pcutoff) for the hysteresis step.

    ``low_threshold`` is the continuation threshold, so it must stay below the
    high threshold: only ``pcutoff > low_threshold`` is considered.
    """
    best_cost: tuple[int, int] | None = None
    best: tuple[float, int, float] | None = None
    for low_threshold in _LOW_THRESHOLD_GRID:
        valid_pcutoffs = pcutoffs[pcutoffs > low_threshold]
        if valid_pcutoffs.size == 0:
            continue
        for min_bout in _MIN_BOUT_GRID:
            result = _select_pcutoff(
                partial(_hysteresis_pred, raw, low_threshold, min_bout),
                valid_pcutoffs,
                target_recall,
            )
            if result is None:
                continue
            pcutoff, cost = result
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best = (low_threshold, min_bout, pcutoff)
    if best is None:
        msg = "No hysteresis parameters reach the target recall."
        raise ValueError(msg)
    low_threshold, min_bout, pcutoff = best
    return HysteresisPostprocessing(
        pcutoff=max(float(pcutoff), _MIN_PCUTOFF),
        low_threshold=low_threshold,
        min_bout_frames=min_bout,
    )


def optimise_postprocessing(
    recipe: ModelRecipe,
    raw: pl.DataFrame,
) -> ModelRecipe:
    """Optimise the post-processing parameters of ``recipe`` on ``raw``.

    Sets the nested post-processing config matching ``recipe.postprocessing_step``
    and returns the recipe.
    """
    pcutoffs = np.unique(
        np.quantile(
            raw.get_column(PROB).to_numpy(),
            np.linspace(0.0, 1.0, _N_PCUTOFF),
        )
    )
    if recipe.postprocessing_step == FRAME_AWARE:
        recipe.frame_aware = _optimise_frame_aware(raw, pcutoffs, recipe.target_recall)
    elif recipe.postprocessing_step == HYSTERESIS:
        recipe.hysteresis = _optimise_hysteresis(raw, pcutoffs, recipe.target_recall)
    else:  # pragma: no cover - guarded by the Literal type
        msg = f"Unknown post-processing step: {recipe.postprocessing_step}"
        raise ValueError(msg)
    return recipe
