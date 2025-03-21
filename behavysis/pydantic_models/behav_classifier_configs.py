from behavysis.utils.pydantic_base_model import PydanticBaseModel


class BehavClassifierConfigs(PydanticBaseModel):
    proj_dir: str = "project_dir"
    behav_name: str = "behav_name"
    seed: int = 42
    oversample_ratio: float = 0.2
    undersample_ratio: float = 0.4

    clf_struct: str = "clf"  # Classifier type (defined in ClfTemplates)
    pcutoff: float = 0.2
    test_split: float = 0.2
    val_split: float = 0.2
    batch_size: int = 256
    epochs: int = 100
