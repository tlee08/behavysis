import os

from behavysis.pydantic_models.pydantic_base_model import PydanticBaseModel


class RunDlcConfigs(PydanticBaseModel):
    model_fp: str = os.path.join("path", "to", "DEEPLABCUT_model", "config.yaml")
