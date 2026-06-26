"""Module of example ExperimentConfig setups for different experiments."""
# TODO

from behavysis.constants import (
    BPTS_CENTRE,
    BPTS_CORNERS,
    BPTS_FRONT,
    BPTS_SIMBA,
    INDIVS_SIMBA,
)

from ..experiment_config import ExperimentConfig, RefConfig, UserConfig  # noqa: TID252
from ..funcs import (  # noqa: TID252
    AnalyseConfig,
    CalculateParamsConfig,
    ClassifyBehaviourConfig,
    ExtractFeaturesConfig,
    FormatVidConfig,
    FreezingConfig,
    FromLikelihoodConfig,
    InRoiConfig,
    PreprocessConfig,
    RefineIdsConfig,
    SocialDistanceConfig,
    SpeedConfig,
)


def get_default_config() -> ExperimentConfig:
    """Get default config."""
    return ExperimentConfig(
        user=UserConfig(
            format_vid=FormatVidConfig(width_px=960, height_px=540, fps=15),
            calculate_params=CalculateParamsConfig(
                from_likelihood=FromLikelihoodConfig(bodyparts="--bpts_simba"),
            ),
            preprocess=PreprocessConfig(
                refine_ids=RefineIdsConfig(bodyparts="--bpts_centre"),
            ),
            extract_features=ExtractFeaturesConfig(
                individuals="--indivs_simba",
                bodyparts="--bpts_simba",
            ),
            classify_behaviour=[ClassifyBehaviourConfig()],
            analyse=AnalyseConfig(
                in_roi=[
                    InRoiConfig(roi_corners="--bpts_corners", bodyparts="--bpts_front"),
                ],
                speed=SpeedConfig(bodyparts="--bpts_centre"),
                social_distance=SocialDistanceConfig(bodyparts="--bpts_centre"),
                freezing=FreezingConfig(bodyparts="--bpts_centre"),
            ),
        ),
        ref=RefConfig.model_validate(
            {
                "indivs_simba": INDIVS_SIMBA,
                "bpts_simba": BPTS_SIMBA,
                "bpts_centre": BPTS_CENTRE,
                "bpts_front": BPTS_FRONT,
                "bpts_corners": BPTS_CORNERS,
            },
        ),
    )
