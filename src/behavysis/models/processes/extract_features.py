from pydantic import BaseModel

from behavysis.constants import BPTS_SIMBA, INDIVS_SIMBA


class ExtractFeaturesConfigs(BaseModel):
    individuals: list[str] | str = INDIVS_SIMBA
    bodyparts: list[str] | str = BPTS_SIMBA
