"""Make Behavysis behaviour classifier training notebook (marimo)."""

from pathlib import Path

from behavysis.utils.template_utils import confirm, save_template

_TEMPLATE = "behaviour_classifier/train_behaviour_classifier.py"
_DST = "train_behaviour_classifier.py"
_PROD_TEMPLATE = "behaviour_classifier/production.yaml"
_PROD_DST = "production.yaml"


def main() -> None:
    """Scaffold the marimo training notebook and production.yaml pointer."""
    if not confirm("Create training notebook in current directory?"):
        return
    if Path(_DST).exists() and not confirm(f"Overwrite existing {_DST}?"):
        return
    save_template(_TEMPLATE, Path(_DST))

    if not Path(_PROD_DST).exists() or confirm(f"Overwrite existing {_PROD_DST}?"):
        save_template(_PROD_TEMPLATE, Path(_PROD_DST))

    _print_next_steps(Path(_DST).resolve(), Path(_PROD_DST).resolve())


def _print_next_steps(dst: Path, prod_dst: Path) -> None:
    """Print how to launch the notebook and where outputs land."""
    print(  # noqa: T201
        f"\nTraining notebook created:\n"
        f"  {dst}\n"
        f"  {prod_dst}\n"
        f"\nThis is a marimo notebook (.py). Launch it with:\n"
        f"  marimo edit {dst.name}\n"
        f"\nNext:\n"
        f"  1. Edit the '1. Configure' cell (clf_dir, behaviour_name,\n"
        f"     individuals, bodyparts, source project, BORIS dir).\n"
        f"  2. Run cells top-to-bottom to assemble data, train, and evaluate.\n"
        f"  3. Inspect each version's evaluation/ folder and the leaderboard.\n"
        f"  4. Edit {prod_dst.name} to point at the model_type/version you\n"
        f"     want to deploy (must match a trained version).\n",
    )


if __name__ == "__main__":
    main()
