from pydantic import ConfigDict

from behavysis_core.pydantic_models.pydantic_base_model import PydanticBaseModel


class BehavClassifierConfigs(PydanticBaseModel):
    """_summary_"""

    model_config = ConfigDict(extra="forbid")

    behaviour_name: str = "BehaviourName"
    seed: int = 42
    undersample_ratio: float = 0.2

    clf_structure: str = "clf"  # Classifier type (defined in ClfTemplates)
    pcutoff: float = 0.5
    test_split: float = 0.2
    val_split: float = 0.2
    batch_size: int = 256
    epochs: int = 50
