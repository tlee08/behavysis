"""Make Behavysis behaviour classifier training notebook (marimo)."""

from pathlib import Path

from behavysis.utils.template_utils import confirm, save_template


def main() -> None:
    """Scaffold the marimo training notebook in the current directory."""
    if not confirm("Create training notebook in current directory?"):
        return
    template_fp = Path("behaviour_classifier") / "train_behaviour_classifier.py"
    dst_fp = Path("train_behaviour_classifier.py")
    if Path(dst_fp).exists() and not confirm(f"Overwrite existing {dst_fp}?"):
        return
    save_template(template_fp, dst_fp)
    _print_next_steps(Path(dst_fp).resolve())


def _print_next_steps(dst: Path) -> None:
    """Print how to launch the notebook and where outputs land."""
    print(  # noqa: T201
        f"\nTraining notebook created:\n"
        f"  {dst}\n"
        f"\nThis is a marimo notebook (.py). Launch it with:\n"
        f"  marimo edit {dst.name}\n"
        f"\nNext:\n"
        f"  1. Edit the '1. Configure' cell (clf_dir, behaviour_name,\n"
        f"     individuals, bodyparts, source project, BORIS dir).\n"
        f"  2. Run cells top-to-bottom to assemble data, train, and evaluate.\n"
        f"  3. Inspect each version's evaluation/ folder and the leaderboard.\n"
        f"     The best model is auto-promoted to contract.yaml.\n",
    )


if __name__ == "__main__":
    main()
