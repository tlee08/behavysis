from matplotlib import pyplot as plt
from pydantic import BaseModel, field_validator

from behavysis.constants import BODYPARTS, COORDS, INDIVIDUALS, SCORER


def _validate_in_set(v: str, valid_values: list[str]) -> str:
    """Validate that value is in the set of valid values.

    Parameters
    ----------
    v : str
        Value to validate.
    valid_values : list[str]
        List of valid values.

    Returns:
    -------
    str
        The validated value.

    Raises:
    ------
    ValueError
        If value is not in valid_values.
    """
    if v not in valid_values:
        msg = f"Value '{v}' not in valid values: {valid_values[:5]}..."
        raise ValueError(msg)
    return v


class EvaluateVidConfig(BaseModel):
    """EvaluateVidConfig."""

    funcs: list[str] | str = ["keypoints", "analysis"]
    pcutoff: float | str = 0.8
    colour_level: str = INDIVIDUALS
    radius: int | str = 3
    cmap: str = "rainbow"
    padding: int = 30

    @field_validator("cmap")
    @classmethod
    def validate_cmap(cls, v) -> str:
        """validate_cmap."""
        return _validate_in_set(v, plt.colormaps())

    @field_validator("colour_level")
    @classmethod
    def validate_colour_level(cls, v) -> str:
        """validate_colour_level."""
        return _validate_in_set(v, [SCORER, INDIVIDUALS, BODYPARTS, COORDS])
