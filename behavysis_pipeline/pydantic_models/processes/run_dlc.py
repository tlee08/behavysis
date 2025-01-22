import os

from behavysis_pipeline.pydantic_models.pydantic_base_model import PydanticBaseModel


class RunDlcConfigs(PydanticBaseModel):
    model_fp: str = os.path.join(".")
