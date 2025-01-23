"""
_summary_
"""

import os

import pandas as pd

from behavysis_pipeline.constants import CACHE_DIR
from behavysis_pipeline.df_classes.features_df import FeaturesDf
from behavysis_pipeline.df_classes.keypoints_df import KeypointsDf
from behavysis_pipeline.pydantic_models.configs import ExperimentConfigs
from behavysis_pipeline.utils.diagnostics_utils import file_exists_msg
from behavysis_pipeline.utils.io_utils import get_name, silent_remove
from behavysis_pipeline.utils.logging_utils import get_io_obj_content, init_logger_with_io_obj
from behavysis_pipeline.utils.multiproc_utils import get_cpid
from behavysis_pipeline.utils.subproc_utils import run_subproc_console
from behavysis_pipeline.utils.template_utils import save_template

# Order of bodyparts is from
# - https://github.com/sgoldenlab/simba/blob/master/docs/Multi_animal_pose.md
# - https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_DLC.md
# - https://github.com/sgoldenlab/simba/blob/master/simba/pose_configurations/bp_names/bp_names.csv
# - https://github.com/sgoldenlab/simba/blob/master/simba/pose_configurations/configuration_names/pose_config_names.csv
# 2 animals; 16 body-parts

#####################################################################
#               FEATURE EXTRACTION FOR SIMBA
#####################################################################


class ExtractFeatures:
    """__summary__"""

    @staticmethod
    def extract_features(
        keypoints_fp: str,
        features_fp: str,
        configs_fp: str,
        overwrite: bool,
    ) -> str:
        """
        Extracting features from preprocessed keypoints dataframe using SimBA
        processes.

        Parameters
        ----------
        keypoints_fp : str
            Preprocessed keypoints filepath.
        dst_fp : str
            Filepath to save extracted_features dataframe.
        configs_fp : str
            Configs JSON filepath.
        overwrite : bool
            Whether to overwrite the dst_fp file (if it exists).

        Returns
        -------
        str
            The outcome of the process.
        """
        logger, io_obj = init_logger_with_io_obj()
        if not overwrite and os.path.exists(features_fp):
            logger.warning(file_exists_msg(features_fp))
            return get_io_obj_content(io_obj)
        # Getting directory and file paths
        name = get_name(keypoints_fp)
        cpid = get_cpid()
        configs_dir = os.path.dirname(configs_fp)
        simba_in_dir = os.path.join(CACHE_DIR, f"input_{cpid}")
        simba_dir = os.path.join(CACHE_DIR, f"simba_proj_{cpid}")
        features_from_dir = os.path.join(simba_dir, "project_folder", "csv", "features_extracted")
        # Preparing keypoints dataframes for input to SimBA project
        os.makedirs(simba_in_dir, exist_ok=True)
        simba_in_fp = os.path.join(simba_in_dir, f"{name}.csv")
        # Selecting bodyparts for SimBA (8 bpts, 2 indivs)
        keypoints_df = KeypointsDf.read(keypoints_fp)
        keypoints_df = select_cols(keypoints_df, configs_fp)
        # Saving keypoints index to use in the SimBA features extraction df
        index = keypoints_df.index
        # Need to remove index name for SimBA to import correctly
        keypoints_df.index.name = None
        # Saving as csv
        keypoints_df.to_csv(simba_in_fp)
        # Removing simba folder (if it exists)
        silent_remove(simba_dir)
        # Running SimBA env and script to run SimBA feature extraction
        logger.info(run_simba_subproc(simba_dir, simba_in_dir, configs_dir, CACHE_DIR, cpid))
        # Exporting SimBA feature extraction csv to disk
        simba_dst_fp = os.path.join(features_from_dir, f"{name}.csv")
        export2df(simba_dst_fp, features_fp, index)
        # Removing temp folders (simba_in_dir, simba_dir)
        silent_remove(simba_in_dir)
        silent_remove(simba_dir)
        return get_io_obj_content(io_obj)


#####################################################################
#               PREPARE FOR SIMBA
#####################################################################


def select_cols(
    keypoints_df: pd.DataFrame,
    configs_fp: str,
) -> pd.DataFrame:
    """
    Selecting given keypoints columns to input to SimBA.

    Parameters
    ----------
    keypoints_df : pd.DataFrame
        Keypoints dataframe.
    configs_fp : str
        Configs dict.

    Returns
    -------
    pd.DataFrame
        Keypoints dataframe with selected columns.
    """
    # Getting necessary config parameters
    configs = ExperimentConfigs.read_json(configs_fp)
    configs_filt = configs.user.extract_features
    indivs = configs.get_ref(configs_filt.individuals)
    bpts = configs.get_ref(configs_filt.bodyparts)
    # Checking that the bodyparts are all valid
    KeypointsDf.check_bpts_exist(keypoints_df, bpts)
    # Selecting given columns
    idx = pd.IndexSlice
    keypoints_df = keypoints_df.loc[:, idx[:, indivs, bpts]]
    return keypoints_df


def run_simba_subproc(
    simba_dir: str,
    keypoints_dir: str,
    configs_dir: str,
    temp_dir: str,
    cpid: int,
) -> str:
    """
    Running the custom SimBA script to take the prepared keypoints dataframe as input and
    create the features extracted dataframe.

    A custom SimBA script must be run in a separate custom conda environment because SimBA
    cannot be installed in the same environment as DEEPLABCUT (and also uses Python 3.6 -
    which is old).

    Parameters
    ----------
    simba_dir : str
        SimBA project directory.
    keypoints_dir : str
        Prepared keypoints dataframes directory. SimBA imports the entire directory.
        If only one file is being processed, put that file in a separate folder.
    configs_dir : str
        Directory path of config files corresponding to keypoints dataframes in keypoints_dir.
        For each keypoints dataframe file, there should be a config file with the same name.
    """
    # Saving the script to a file
    script_fp = os.path.join(temp_dir, f"simba_subproc_{cpid}.py")
    save_template(
        "simba_subproc.py",
        "behavysis_pipeline",
        "templates",
        script_fp,
        simba_dir=simba_dir,
        keypoints_dir=keypoints_dir,
        configs_dir=configs_dir,
    )
    # Running the Simba subprocess in a separate conda env
    cmd = [
        os.environ["CONDA_EXE"],
        "run",
        "--no-capture-output",
        "-n",
        "simba",
        "python",
        script_fp,
    ]
    # run_subproc_fstream(cmd)
    run_subproc_console(cmd)
    # Removing the script file
    silent_remove(script_fp)
    return "Ran SimBA feature extraction script.\n"


def remove_bpts_cols(
    keypoints_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Drops the bodyparts columns from the SimBA features extractions dataframes.
    Because bodypart coordinates should not be a factor in behaviour classification.

    Parameters
    ----------
    keypoints_df : pd.DataFrame
        Features extracted dataframe

    Returns
    -------
    pd.DataFrame
        Features extracted dataframe with the bodyparts columns dropped.
    """
    indivs_n = 2
    bpts_n = 8
    coords_n = 3
    n = indivs_n * bpts_n * coords_n
    return keypoints_df.iloc[:, n:]


def export2df(in_fp: str, dst_fp: str, index: pd.Index) -> str:
    """
    __summary__
    """
    features_df = FeaturesDf.read_csv(in_fp)
    # Setting index to the same as the preprocessed preprocessed df
    features_df = features_df.set_index(index)
    # Saving SimBA extracted features df on disk
    FeaturesDf.write(features_df, dst_fp)
    return "Exported SimBA features to disk.\n"
