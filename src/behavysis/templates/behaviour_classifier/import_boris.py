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
        BEHAVIOUR,
        FRAME,
        TRUE_NEG,
        TRUE_POS,
    )
    from behavysis.models import ExperimentMetadata
    from behavysis.schemas import write_df
    from behavysis.utils import configure_logger, has_output_files

    configure_logger()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Import BORIS CSV to Behavysis Scored Behaviour

    Converts BORIS `.csv` exports into `7_behaviour_scored/*.parquet` files,
    aligned to each experiment's metadata (fps, frame range).

    Output is fully-wide format: one row per frame, one column per behaviour.
    """)
    return


@app.cell
def _():
    mo.md(r"""## Configure""")
    return


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
    """Import BORIS CSV to fully-wide scored DataFrame.

    Returns a DataFrame with ``FRAME`` + one column per behaviour,
    each Int64 (TRUE_POS / TRUE_NEG values).
    """
    df_boris = (
        pl.read_csv(fp)
        .rename({"Behavior": BEHAVIOUR})
        .with_columns(
            (pl.col("Time") * fps).round().cast(pl.Int64).alias(FRAME),
            pl.col("Behavior type").str.strip_chars().str.to_uppercase().alias("type"),
        )
    )

    boris_behaviours = df_boris[BEHAVIOUR].unique().to_list()
    missing = [b for b in behaviour_ls if b not in boris_behaviours]
    if missing:
        logger.warning(
            "Behaviours not in BORIS file: {}\nBORIS: {}",
            missing,
            boris_behaviours,
        )

    window = int(point_window_sec * fps)
    frame_count = stop_frame - start_frame
    frames = np.arange(start_frame, stop_frame, dtype=np.int64)

    result = pl.DataFrame({FRAME: frames})

    for behaviour in behaviour_ls:
        vals = np.full(frame_count, TRUE_NEG, dtype=np.int64)
        evts_df = df_boris.filter(pl.col(BEHAVIOUR) == behaviour).sort(FRAME)

        for row in evts_df.iter_rows(named=True):
            f = int(row[FRAME])
            typ = row["type"]
            if typ in ("START", "STOP"):
                val = pos_value if typ == "START" else TRUE_NEG
                vals[f - start_frame :] = val
            elif typ == "POINT":
                lo = max(f - window, start_frame)
                hi = min(f + window, stop_frame - 1)
                vals[lo - start_frame : hi - start_frame + 1] = pos_value

        result = result.with_columns(pl.Series(behaviour, vals, dtype=pl.Int64))

    return result


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
            fps=fps,
            point_window_sec=point_window_sec,
        )
        write_df(df, dst_fp)
    sorted(p.name for p in dst_dir.iterdir())


if __name__ == "__main__":
    app.run()
