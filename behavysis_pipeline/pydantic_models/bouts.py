"""
_summary_
"""

from typing import Dict, List

from behavysis_pipeline.pydantic_models.pydantic_base_model import PydanticBaseModel


class Bout(PydanticBaseModel):
    start: int
    stop: int
    dur: int
    behav: str
    actual: int
    user_defined: Dict[str, int]


class BoutStruct(PydanticBaseModel):
    behav: str
    user_defined: List[Bout]


class Bouts(PydanticBaseModel):
    start: int
    stop: int
    bouts: List[Bout]
    bouts_struct: List[BoutStruct]
