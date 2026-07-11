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

from behavysis.utils import confirm

_PRESETS_ROOT = Path(str(files("behavysis"))) / "presets"


def list_presets() -> list[str]:
    """Return sorted list of available preset names."""
    return sorted(
        d.name
        for d in _PRESETS_ROOT.iterdir()
        if d.is_dir() and (d / "default_config.yaml").exists()
    )


def copy_preset(name: str, dst_dir: Path) -> Path:
    """Copy a preset folder to *dst*, creating a project directory."""
    if name not in list_presets():
        msg = f"Unknown preset '{name}'. Available presets: {list_presets()}"
        raise ValueError(msg)

    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy run_pipeline.py and default_config.yaml
    for _i in ["run_pipeline.py", "default_config.yaml"]:
        _preset_fp = _PRESETS_ROOT / name / _i
        _dst_fp = dst_dir / _i
        if not _dst_fp.exists() or confirm(f"Overwrite {_i}?"):
            shutil.copy2(_preset_fp, _dst_fp)
    # Return dst_dir path
    return dst_dir


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
