"""Template utilities."""

import re
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


def render_template(template_name: str, **kwargs: object) -> str:
    """Render a template to a string."""
    env = Environment(
        loader=PackageLoader("behavysis", "templates"),
        autoescape=False,  # noqa: S701
    )
    result = env.get_template(template_name).render(**kwargs)
    return re.sub(r"\n{3,}", "\n\n", result).strip() + "\n"


def save_template(template_name: str, dst: Path, **kwargs: object) -> None:
    """Render and save a template."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(render_template(template_name, **kwargs))
