import marimo

__generated_with = "0.23.10"
app = marimo.App()

with app.setup:
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import polars as pl

    from behavysis.schemas import KEYPOINTS_SCHEMA, write_df
    from behavysis.funcs.run_dlc import convert_raw_dlc_to_keypoints
    from behavysis.utils import configure_logger

    configure_logger()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Migrate Keypoints from Old to New Format

    Converts old-format keypoints parquet files to the new long-form schema
    expected by the current pipeline.
    """)
    return


@app.cell
def _():
    mo.md(r"""## Configure""")
    return


@app.cell
def _():
    src_dir = Path("/path/to/old/keypoints")
    dst_dir = Path("/path/to/new/keypoints")

    mo.accordion(
        {
            "src_dir": str(src_dir),
            "dst_dir": str(dst_dir),
        }
    )
    return dst_dir, src_dir


@app.cell
def _(dst_dir, src_dir):
    """Convert keypoints from old to new format."""
    results = {}
    for fp in src_dir.iterdir():
        if not fp.suffix == ".parquet":
            continue
        name = fp.stem
        df = pd.read_parquet(fp)
        converted = convert_raw_dlc_to_keypoints(df)
        write_df(converted, dst_dir / f"{name}.parquet", KEYPOINTS_SCHEMA)
        results[name] = converted
    list(results.keys())


if __name__ == "__main__":
    app.run()
