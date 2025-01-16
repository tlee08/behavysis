"""
_summary_
"""

from typing import Dict, List

from pydantic import BaseModel

from behavysis_pipeline.pydantic_models.pydantic_base_model import PydanticBaseModel


class Bout(BaseModel):
    """__summary__"""

    start: int
    stop: int
    behaviour: str
    actual: int
    user_defined: Dict[str, int]


class Bouts(PydanticBaseModel, BaseModel):
    """__summary__"""

    start: int
    stop: int
    bouts: List[Bout]
