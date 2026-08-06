"""Make a new behavysis project directory."""

import argparse
import shutil
import sys
from importlib.resources import files
from pathlib import Path

from behavysis.constants import (
    ANALYSIS_COMBINED_DIR,
    CONFIG_DIR,
    DEFAULT_CONFIG_FP,
    RAW_VIDEO_DIR,
    RUN_PIPELINE_FP,
    STAGES,
)
from behavysis.templates.presets import PRESET_DESCRIPTIONS
from behavysis.utils import confirm

_PRESETS_ROOT = Path(str(files("behavysis"))) / "presets"


def main() -> None:
    """Scaffold a behavysis project with a preset config and notebook."""
    parser = argparse.ArgumentParser(
        description="Create a new behavysis project directory.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Preset name (use --list to see available)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available presets and exit",
    )
    parser.add_argument(
        "project_dir",
        nargs="?",
        default=".",
        help="Target directory (default: current directory)",
    )
    args = parser.parse_args()

    if args.list:
        _print_presets()
        sys.exit(0)

    dst_dir = Path(args.project_dir).resolve()
    # Determine preset
    preset_name = args.preset or _choose_preset()
    if not preset_name:
        sys.exit(0)
    # Confirm create project in the directory
    if not confirm(f"Create behavysis project in {dst_dir}?"):
        sys.exit(0)

    # Copy preset
    _copy_preset(preset_name, dst_dir)
    # Make stage folders
    for folder in STAGES:
        (dst_dir / folder).mkdir(parents=True, exist_ok=True)

    _print_next_steps(dst_dir, preset_name)

    sys.exit(0)


def _copy_preset(name: str, dst_dir: Path) -> None:
    """Copy a preset folder to *dst*, creating a project directory."""
    if name not in PRESET_DESCRIPTIONS:
        print(f"Unknown preset '{name}'.")  # noqa: T201
        _print_presets()
        sys.exit(1)

    dst_dir.mkdir(parents=True, exist_ok=True)
    # Copy run_pipeline.py and default_config.yaml
    for _i in [RUN_PIPELINE_FP, DEFAULT_CONFIG_FP]:
        _preset_fp = _PRESETS_ROOT / name / _i
        _dst_fp = dst_dir / _i
        if _dst_fp.exists() and not confirm(f"Overwrite {_i}?"):
            continue
        shutil.copy2(_preset_fp, _dst_fp)


def _print_presets() -> None:
    """Print available presets."""
    print("Available presets:")  # noqa: T201
    for name, description in PRESET_DESCRIPTIONS.items():
        print(f"  {name:<25} {description}")  # noqa: T201


def _choose_preset() -> str | None:
    """Interactively choose a preset."""
    _print_presets()
    print()  # noqa: T201
    return input("Choose a preset (or press Enter to cancel): ").strip()


def _print_next_steps(dst: Path, preset_name: str) -> None:
    """Print next steps after project creation."""
    config_name = DEFAULT_CONFIG_FP
    print(  # noqa: T201
        f"\nProject created ({preset_name}):\n"
        f"  {dst / config_name}     ← edit this for your experiment\n"
        f"  {dst / RUN_PIPELINE_FP}     ← open in Jupyter/marimo/VS Code\n"
        f"  {dst / CONFIG_DIR}/  …  {dst / ANALYSIS_COMBINED_DIR}/  ← stage folders\n"
        f"\nNext:\n"
        f"  1. Copy .mp4 video(s) into {dst / RAW_VIDEO_DIR}/\n"
        f"  2. Edit {config_name}:\n"
        f"     - Set run_dlc.model_fp to your DLC model config.yaml\n"
        f"     - Set px_per_mm.dist_mm to your arena size\n"
        f"  3. Open run_pipeline.py and run cells top-to-bottom\n",
    )


if __name__ == "__main__":
    main()
