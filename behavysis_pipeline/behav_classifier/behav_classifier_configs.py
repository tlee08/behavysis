"""
_summary_
"""

from behavysis_core.data_models.pydantic_base_model import PydanticBaseModel
from pydantic import ConfigDict


class BehavClassifierConfigs(PydanticBaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    train_fraction: float = 0.8
    undersampling_strategy: float = 0.2
    seed: int = 42
    pcutoff: float = 0.5
    behaviour_name: str = "BehaviourName"

    window_frames: int = 5

    model_template_fp: str = "./model_template"  # Path to the model template
    # model_type: str = "RandomForestClassifier"
    # model_params: dict = {
    #     "n_estimators": 2000,
    #     "learning_rate": 0.1,
    #     "loss": "log_loss",
    #     "criterion": "friedman_mse",
    #     "max_features": "sqrt",
    #     "random_state": seed,
    #     "subsample": 1.0,
    #     "verbose": 1,
    # }
