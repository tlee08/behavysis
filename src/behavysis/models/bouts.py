"""_summary_"""

from behavysis.utils.pydantic_base_model import PydanticBaseModel


class Bout(PydanticBaseModel):
    start: int
    stop: int
    dur: int
    behav: str
    actual: int
    user_defined: dict[str, int]


class BoutStruct(PydanticBaseModel):
    behav: str
    user_defined: list[str]


class Bouts(PydanticBaseModel):
    start: int
    stop: int
    bouts: list[Bout]
    bouts_struct: list[BoutStruct]
