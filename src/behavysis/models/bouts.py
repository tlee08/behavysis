"""Bout data models for behavioural event tracking."""

from pydantic import BaseModel


class Bout(BaseModel):
    """Single Bout."""

    start: int
    stop: int
    dur: int
    behaviour: str
    actual: int
    sub_behaviour: dict[str, int]


class BoutStruct(BaseModel):
    """Bout Structure."""

    behaviour: str
    sub_behaviour: list[str]


class Bouts(BaseModel):
    """Bouts."""

    start: int
    stop: int
    bouts: list[Bout]
    bout_struct: list[BoutStruct]
