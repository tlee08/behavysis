import marimo

__generated_with = "0.23.10"
app = marimo.App()

with app.setup:
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import polars as pl
    from loguru import logger

    from behavysis.constants import (
        ACTUAL,
        BEHAVIOUR,
        FRAME,
        TRUE_NEG,
        TRUE_POS,
    )
    from behavysis.models import (
        ExperimentMetadata,
    )
    from behavysis.schemas import BEHAVIOUR_SCORED_BASE, write_df
    from behavysis.utils import configure_logger, has_output_files

    configure_logger()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Import BORIS CSV to Behavysis Scored Behaviour

    Converts BORIS `.csv` exports into `7_behaviour_scored/*.parquet` files,
    aligned to each experiment's metadata (fps, frame range).
    """)


@app.cell
def _():
    mo.md(r"""## Configure""")


@app.cell
def _():
    boris_dir = Path("/path/to/boris_csvs")
    dst_dir = Path("/path/to/scored_output")
    behaviour_ls = ["behaviour1", "behaviour2"]
    overwrite = False
    point_window_sec = 0.0
    fps = 50

    metadata = ExperimentMetadata()
    mo.accordion(
        {
            "boris_dir": str(boris_dir),
            "dst_dir": str(dst_dir),
            "behaviour_ls": behaviour_ls,
            "overwrite": overwrite,
            "point_window_sec": point_window_sec,
            "fps": fps,
        }
    )
    return behaviour_ls, boris_dir, dst_dir, fps, metadata, overwrite, point_window_sec


@app.function
def import_boris_csv(
    fp: Path,
    behaviour_ls: list[str],
    start_frame: int,
    stop_frame: int,
    fps: int,
    *,
    point_window_sec: float = 0.0,
    pos_value: int = TRUE_POS,
) -> pl.DataFrame:
    """Import BORIS CSV to BEHAVIOUR_SCORED_BASE long-form DataFrame.

    Parameters
    ----------
    fp : Path
        Path to BORIS CSV file.
    behaviour_ls : list[str]
        List of behaviour names to import. Others in the CSV are skipped.
    start_frame : int
        First frame of the experiment.
    stop_frame : int
        Last frame of the experiment (exclusive).
    fps : int
        Frames per second for converting *point_window_sec* to frames.
    point_window_sec : float
        For POINT behaviours, mark ±window seconds around each event as
        ``actual=TRUE_POS``. Default 0.0 (single frame only).
    pos_value: int
        What to use for the "is_behaviour". Defaults to TRUE_POS = 1.

    Returns:
    --------
    pl.DataFrame
        Long-form DataFrame with schema ``BEHAVIOUR_SCORED_BASE``:
        ``{frame, behaviour, actual}``.
    """
    # Read boris df
    # Inferring frames from time so we are FPS agnostic
    df_boris = (
        pl.read_csv(fp)
        .rename({"Behavior": BEHAVIOUR})
        .with_columns(
            (pl.col("Time") * fps).round().cast(pl.Int64).alias(FRAME),
            pl.col("Behavior type").str.strip_chars().str.to_uppercase().alias("type"),
        )
    )
    # Check behaviour exists
    boris_behaviours = df_boris[BEHAVIOUR].unique().to_list()
    missing = [b for b in behaviour_ls if b not in boris_behaviours]
    if missing:
        logger.warning(
            "Behaviours not in BORIS file: {}\nBORIS: {}",
            missing,
            boris_behaviours,
        )
    # Make window
    window = int(point_window_sec * fps)
    frames = np.arange(start_frame, stop_frame, dtype=np.int64)
    # For each given behaviour, construct the fbf df from boris df
    fbf_df_ls: list[pl.DataFrame] = []
    for behaviour in behaviour_ls:
        _df = pl.DataFrame(
            {FRAME: frames, BEHAVIOUR: behaviour, ACTUAL: TRUE_NEG},
            schema=BEHAVIOUR_SCORED_BASE,
        )
        # Filter boris_df by behaviour and sort by frame
        evts_df = df_boris.filter(pl.col(BEHAVIOUR) == behaviour).sort(FRAME)
        for row in evts_df.iter_rows(named=True):
            f = int(row[FRAME])
            typ = row["type"]
            # If START or STOP, then set > curr_frame accordingly
            if typ in ("START", "STOP"):
                val = pos_value if typ == "START" else TRUE_NEG
                _df = _df.with_columns(
                    pl.when(pl.col(FRAME) >= f)
                    .then(val)
                    .otherwise(pl.col(ACTUAL))
                    .alias(ACTUAL),
                )
            # If POINT, then set nearby window to pos_value
            elif typ == "POINT":
                lo = max(f - window, start_frame)
                hi = min(f + window, stop_frame - 1)
                _df = _df.with_columns(
                    pl.when(pl.col(FRAME).is_between(lo, hi))
                    .then(pos_value)
                    .otherwise(pl.col(ACTUAL))
                    .alias(ACTUAL),
                )
        # Add to list
        fbf_df_ls.append(_df)
    # Concatenate fbf behaviours and return
    return pl.concat(fbf_df_ls)


@app.cell
def _(behaviour_ls, boris_dir, dst_dir, fps, metadata, overwrite, point_window_sec):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for csv_fp in sorted(boris_dir.glob("*.csv")):
        name = csv_fp.stem
        dst_fp = dst_dir / f"{name}.parquet"
        if not overwrite and has_output_files(dst_fp):
            continue
        df = import_boris_csv(
            csv_fp,
            behaviour_ls,
            metadata.require_start_frame(),
            metadata.require_stop_frame() + 1,
            point_window_sec=point_window_sec,
            fps=fps,
        )
        write_df(df, dst_fp, BEHAVIOUR_SCORED_BASE)
    sorted(p.name for p in dst_dir.iterdir())


if __name__ == "__main__":
    app.run()
