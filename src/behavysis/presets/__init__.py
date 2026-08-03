"""Preset project templates for common experiment types.

Each preset is a self-contained folder with an annotated ``default_config.yaml``
and a matching ``run_pipeline.py`` marimo notebook.

Usage::

    from behavysis.presets import copy_preset, list_presets

    list_presets()                                   # discover available presets
    copy_preset("open_field_single", "my_project/")  # copy to project dir
"""

from pathlib import Path

import yaml

_PRESETS_ROOT = Path(__file__).parent

_PRESET_NAMES = sorted(
    d.name
    for d in _PRESETS_ROOT.iterdir()
    if d.is_dir() and (d / "default_config.yaml").exists()
)
_PRESET_METADATAS = {
    preset: yaml.safe_load((_PRESETS_ROOT / preset / "metadata.yaml").read_text())
    for preset in _PRESET_NAMES
}
PRESET_DESCRIPTIONS = {
    metadata["name"]: metadata["description"] for metadata in _PRESET_METADATAS.values()
}
