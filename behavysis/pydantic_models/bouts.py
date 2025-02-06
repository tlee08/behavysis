"""
_summary_
"""

from typing import Dict

from behavysis.utils.pydantic_base_model import PydanticBaseModel


class Bout(PydanticBaseModel):
    start: int
    stop: int
    dur: int
    behav: str
    actual: int
    user_defined: Dict[str, int]


class BoutStruct(PydanticBaseModel):
    behav: str
    user_defined: list[Bout]


class Bouts(PydanticBaseModel):
    start: int
    stop: int
    bouts: list[Bout]
    bouts_struct: list[BoutStruct]
