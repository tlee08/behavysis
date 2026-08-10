"""Template utilities."""

from pathlib import Path

from jinja2 import Environment, PackageLoader


def confirm(prompt: str, *, default: bool = False) -> bool:
    """Get yes/no confirmation from user."""
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        response = input(f"{prompt} {hint}: ").strip().lower()
        if not response:
            return default
        if response in ("y", "yes"):
            return True
        if response in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'.")  # noqa: T201


def render_template(template_fp: Path, **kwargs: object) -> str:
    """Render a template to a string."""
    env = Environment(
        loader=PackageLoader("behavysis", "templates"),
        autoescape=False,  # noqa: S701
    )
    return env.get_template(str(template_fp)).render(**kwargs)


def save_template(template_fp: Path, dst: Path, **kwargs: object) -> None:
    """Render and save a template."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(render_template(template_fp, **kwargs))
