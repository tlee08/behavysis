from behavysis.constants import SIMBA_BODYPARTS, SIMBA_INDIVIDUALS
from behavysis.pydantic_models.pydantic_base_model import PydanticBaseModel


class ExtractFeaturesConfigs(PydanticBaseModel):
    individuals: list[str] | str = SIMBA_INDIVIDUALS
    bodyparts: list[str] | str = SIMBA_BODYPARTS
