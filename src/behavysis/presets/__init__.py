"""Preset project templates for common experiment types.

Each preset is a self-contained folder with an annotated ``default_config.yaml``
and a matching ``run_pipeline.py`` marimo notebook.

Usage::

    from behavysis.presets import copy_preset, list_presets

    list_presets()                                   # discover available presets
    copy_preset("open_field_single", "my_project/")  # copy to project dir
"""

import shutil
from importlib.resources import files
from pathlib import Path

_PRESETS_ROOT = Path(str(files("behavysis"))) / "presets"


def list_presets() -> list[str]:
    """Return sorted list of available preset names."""
    return sorted(
        d.name
        for d in _PRESETS_ROOT.iterdir()
        if d.is_dir() and (d / "default_config.yaml").exists()
    )


def copy_preset(name: str, dst: str | Path) -> Path:
    """Copy a preset folder to *dst*, creating a project directory.

    Args:
        name: Preset name (e.g. ``"open_field_single"``).
        dst: Destination directory (created if it doesn't exist).

    Returns:
        The destination path.

    Raises:
        ValueError: If *name* is not a recognised preset.
    """
    if name not in list_presets():
        msg = f"Unknown preset '{name}'. Available presets: {list_presets()}"
        raise ValueError(msg)

    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        _PRESETS_ROOT / name,
        dst,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    return dst


_DESCRIPTIONS: dict[str, str] = {
    "base": (
        "Reference template with all options documented — "
        "good starting point to understand every field."
    ),
    "open_field_single": (
        "Single-mouse open field: speed, in_roi (thigmotaxis) analysis."
    ),
    "social_two_mice": (
        "Two-mouse social interaction: speed, distance, social distance."
    ),
}


def describe_presets() -> dict[str, str]:
    """Return ``{name: description}`` for all available presets."""
    return {name: _DESCRIPTIONS.get(name, "") for name in list_presets()}
