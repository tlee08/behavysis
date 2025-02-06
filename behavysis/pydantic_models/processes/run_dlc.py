import os

from behavysis.utils.pydantic_base_model import PydanticBaseModel


class RunDlcConfigs(PydanticBaseModel):
    model_fp: str = os.path.join("path", "to", "DEEPLABCUT_model", "config.yaml")
