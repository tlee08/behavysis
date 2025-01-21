"""
_summary_
"""

from typing import Dict, List

from pydantic import BaseModel, ConfigDict

from behavysis_pipeline.pydantic_models.pydantic_base_model import PydanticBaseModel


class Bout(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: int
    stop: int
    dur: int
    behav: str
    actual: int
    user_defined: Dict[str, int]


class BoutStruct(BaseModel):
    model_config = ConfigDict(extra="forbid")

    behav: str
    user_defined: List[Bout]


class Bouts(PydanticBaseModel):
    model_config = ConfigDict(extra="forbid")

    start: int
    stop: int
    bouts: List[Bout]
    bouts_struct: List[BoutStruct]
