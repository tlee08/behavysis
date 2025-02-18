from behavysis.constants import BPTS_SIMBA, INDIVS_SIMBA
from behavysis.utils.pydantic_base_model import PydanticBaseModel


class ExtractFeaturesConfigs(PydanticBaseModel):
    individuals: list[str] | str = INDIVS_SIMBA
    bodyparts: list[str] | str = BPTS_SIMBA
