"""Bout data models for behavioral event tracking."""

from pydantic import BaseModel


class Bout(BaseModel):
    start: int
    stop: int
    dur: int
    behav: str
    actual: int
    user_defined: dict[str, int]


class BoutStruct(BaseModel):
    behav: str
    user_defined: list[str]


class Bouts(BaseModel):
    start: int
    stop: int
    bouts: list[Bout]
    bouts_struct: list[BoutStruct]
