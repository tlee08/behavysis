from pathlib import Path


def file_exists_msg(fp: Path | str | None = None) -> str:
    """Return a warning message."""
    fp_str = f", {fp}, " if fp else " "
    return (
        f"Output file{fp_str}already exists - not overwriting file.\n"
        "To overwrite, specify `overwrite=True`."
    )
