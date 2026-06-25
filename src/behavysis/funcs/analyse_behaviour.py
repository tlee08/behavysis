from pathlib import Path

import numpy as np

from behavysis.constants import FALSE_POS, FBF, OUTCOMES, PRED, UNSURE
from behavysis.df_classes import AnalysisBinnedDf, AnalysisDf, BehaviourScoredDf
from behavysis.models import ExperimentConfig


def analyse_behaviour(
    behaviour_fp: Path,
    dst_dir: Path,
    config_fp: Path,
) -> None:
    """Takes a behaviour df and generates a summary and binned version of the data."""
    name = behaviour_fp.stem
    dst_subdir = dst_dir / "analyse_behaviour"
    # Calculating deltas (changes in bpts) between each frame for the subject
    config = ExperimentConfig.model_validate_json(config_fp.read_text())
    analysis_config = config.get_analysis_config()
    # Loading in dataframe
    behaviour_df = BehaviourScoredDf.read(behaviour_fp)
    # Setting all na and undetermined behav to non-behav
    behaviour_df = behaviour_df.fillna(0).replace(UNSURE, FALSE_POS)
    # Getting the behaviour names and each user_defined for the behaviour
    # Not incl. the `pred` or `prob` (`prob` shouldn't be here anyway) columns
    columns = np.isin(
        behaviour_df.columns.get_level_values(OUTCOMES),
        [PRED],
        invert=True,
    )
    behaviour_df = behaviour_df.loc[:, columns]
    behaviour_df = AnalysisDf.clean_and_validate(behaviour_df)
    # Writing to fbf file
    fbf_fp = dst_subdir / FBF / f"{name}.{AnalysisDf.io_format}"
    AnalysisDf.write(behaviour_df, fbf_fp)
    # Making the summary and binned dataframes
    AnalysisBinnedDf.summary_binned_behaviour(
        behaviour_df,
        dst_subdir,
        name,
        analysis_config.fps,
        analysis_config.bins_sec,
        analysis_config.custom_bins_sec,
    )
