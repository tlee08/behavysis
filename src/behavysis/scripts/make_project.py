"""Make a new behavysis project directory."""

import argparse
import sys
from pathlib import Path

from behavysis.constants import (
    ANALYSIS_COMBINED_DIR,
    CONFIG_DIR,
    DEFAULT_CONFIG_FP,
    RAW_VIDEO_DIR,
    STAGES,
)
from behavysis.presets import copy_preset, describe_presets, list_presets
from behavysis.utils.template_utils import confirm


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

    dst = Path(args.project_dir).resolve()

    # Determine preset
    preset_name = args.preset or _choose_preset()
    if preset_name is None:
        sys.exit(0)
    # Validate (catchs --preset xyz before confirmation prompt)
    if preset_name not in list_presets():
        _print_presets()
        print(f"\nUnknown preset '{preset_name}'. Use --list to see available.")  # noqa: T201
        sys.exit(1)
    # Confirm create project in the directory
    if not confirm(f"Create behavysis project in {dst}?"):
        sys.exit(0)

    # Copy preset
    copy_preset(preset_name, dst)
    # Make stage folders
    for folder in STAGES:
        (dst / folder).mkdir(parents=True, exist_ok=True)

    _print_next_steps(dst, preset_name)

    sys.exit(0)


def _print_presets() -> None:
    """Print available presets."""
    descriptions = describe_presets()
    print("Available presets:")  # noqa: T201
    for name in list_presets():
        desc = descriptions.get(name, "")
        print(f"  {name:<25} {desc}")  # noqa: T201


def _choose_preset() -> str | None:
    """Interactively choose a preset."""
    names = list_presets()
    _print_presets()
    print()  # noqa: T201
    while True:
        raw = input("Choose a preset (or press Enter to cancel): ").strip()
        if not raw:
            return None
        if raw in names:
            return raw
        print(f"  Unknown preset '{raw}'. Options: {', '.join(names)}")  # noqa: T201


def _print_next_steps(dst: Path, preset_name: str) -> None:
    """Print next steps after project creation."""
    config_name = DEFAULT_CONFIG_FP
    print(  # noqa: T201
        f"\nProject created ({preset_name}):\n"
        f"  {dst / config_name}     ← edit this for your experiment\n"
        f"  {dst / 'run_pipeline.py'}     ← open in Jupyter/marimo/VS Code\n"
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
