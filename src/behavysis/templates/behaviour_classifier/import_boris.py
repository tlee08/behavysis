import marimo

__generated_with = "0.23.10"
app = marimo.App()

with app.setup:
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import polars as pl
    from loguru import logger

    from behavysis.constants import BEHAVIOUR, FRAME, TRUE_NEG, TRUE_POS
    from behavysis.funcs import dur_frames_from_likelihood, px_per_mm
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


@app.cell
def _():
    mo.md(r"""## Configure""")


@app.cell
def _():
    boris_dir = Path("/path/to/boris_csvs")
    dst_dir = Path("/path/to/scored_output")
    metadata_dir = Path("/path/to/metadata")
    behaviour_ls = ["behaviour1", "behaviour2"]
    overwrite = False
    point_window_sec = 0.5

    mo.accordion(
        {
            "boris_dir": str(boris_dir),
            "dst_dir": str(dst_dir),
            "behaviour_ls": behaviour_ls,
            "overwrite": overwrite,
            "point_window_sec": point_window_sec,
        }
    )
    return behaviour_ls, boris_dir, dst_dir, metadata_dir, overwrite, point_window_sec


@app.function
def import_boris_csv(
    fp: Path,
    behaviour_ls: list[str],
    start_frame: int,
    stop_frame: int,
    fps: float,
    *,
    point_window_sec: float = 0.5,
    centre: bool = True,
    pos_value: int = TRUE_POS,
) -> pl.DataFrame:
    """Import BORIS CSV to fully-wide scored DataFrame.

    Returns a DataFrame with ``FRAME`` + one column per behaviour,
    each Int64 (TRUE_POS / TRUE_NEG values).
    """
    # Can either use "Time" column or "Image index"
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

    window = round(point_window_sec * fps)
    window_centre = round(window / 2)
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
                if centre:
                    lo = max(f - window_centre, start_frame)
                    hi = min(f + window_centre, stop_frame - 1)
                else:
                    lo = f
                    hi = min(f + window, stop_frame - 1)
                vals[lo - start_frame : hi - start_frame + 1] = pos_value

        result = result.with_columns(pl.Series(behaviour, vals, dtype=pl.Int64))

    return result


@app.cell
def _(behaviour_ls, boris_dir, dst_dir, metadata_dir, overwrite, point_window_sec):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for csv_fp in sorted(boris_dir.glob("*.csv")):
        name = csv_fp.stem
        dst_fp = dst_dir / f"{name}.parquet"
        metadata = ExperimentMetadata.read_yaml(metadata_dir / f"{name}.yaml")
        if not overwrite and has_output_files(dst_fp):
            continue
        df = import_boris_csv(
            csv_fp,
            behaviour_ls,
            metadata.require_start_frame(),
            metadata.require_stop_frame() + 1,
            metadata.require_fps(),
            point_window_sec=point_window_sec,
        )
        write_df(df, dst_fp)
    sorted(p.name for p in dst_dir.iterdir())


@app.cell
def _():
    mo.md(r"""## Scratch for metadata setup""")


@app.cell
def _(proj):
    proj.calculate_parameters(
        funcs=(
            # start_frame_from_likelihood,
            # stop_frame_from_dur,
            dur_frames_from_likelihood,
            px_per_mm,
        ),
    )


@app.cell
def _(proj, proj_dir):
    for _exp in proj.experiments:
        _metadata = _exp.read_metadata()
        _boris_fp = proj_dir / "boris_scoring_fr" / f"{_exp.name[:-2]}.csv"
        _boris_fp = _boris_fp if "-a" in _exp.name else Path("abcd")
        print(_exp.name)
        if _boris_fp.exists():
            print(_boris_fp)
            _boris_df = pl.read_csv(_boris_fp)
            _start_frame = (
                _boris_df.filter(pl.col("Behavior") == "start")
                .get_column("Image index")
                .item(0)
            )
            _metadata.start_frame = _start_frame
            _metadata.stop_frame = round(_start_frame + 120 * _metadata.require_fps())
        else:
            _metadata.start_frame = 0
            _metadata.stop_frame = _metadata.require_total_frames()
        _exp.write_metadata(_metadata)


if __name__ == "__main__":
    app.run()
