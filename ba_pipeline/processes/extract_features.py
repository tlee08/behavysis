"""
_summary_
"""

import os
import shutil

import pandas as pd

from ba_package.utils.funcs import (
    check_bpts_exist,
    get_name,
    read_configs,
    read_feather,
    run_subprocess_fstream,
    warning_msg,
    write_feather,
)

# Order of bodyparts is from
# - https://github.com/sgoldenlab/simba/blob/master/docs/Multi_animal_pose.md
# - https://github.com/sgoldenlab/simba/blob/master/docs/Tutorial_DLC.md
# - https://github.com/sgoldenlab/simba/blob/master/simba/pose_configurations/bp_names/bp_names.csv
# - https://github.com/sgoldenlab/simba/blob/master/simba/pose_configurations/configuration_names/pose_config_names.csv
# And can also double check by looking at the order in "project_folder\csv\input_csv" files.

#####################################################################
#               FEATURE EXTRACTION FOR SIMBA
#####################################################################


class ExtractFeatures:
    @staticmethod
    def extract_features(
        dlc_fp: str,
        out_fp: str,
        configs_fp: str,
        temp_dir: str,
        remove_temp: bool,
        overwrite: bool,
    ) -> str:
        """
        Extracting features from preprocessed DLC dataframe using SimBA
        processes.

        Parameters
        ----------
        dlc_fp : str
            Preprocessed DLC filepath.
        out_fp : str
            Filepath to save extracted_features dataframe.
        configs_fp : str
            Configs JSON filepath.
        temp_dir : str
            Temporary directory path. Used during intermediate SimBA processes.
        remove_temp : bool
            Whether to remove the temp directory.
        overwrite : bool
            Whether to overwrite the out_fp file (if it exists).

        Returns
        -------
        str
            The outcome of the process.
        """
        outcome = ""
        # If overwrite is False, checking if we should skip processing
        if not overwrite and os.path.exists(out_fp):
            return warning_msg()
        # Getting directory and file paths
        name = get_name(dlc_fp)
        configs_dir = os.path.split(configs_fp)[0]
        simba_in_dir = os.path.join(temp_dir, "input")
        simba_dir = os.path.join(temp_dir, "simba_proj")
        features_from_dir = os.path.join(
            simba_dir, "project_folder", "csv", "features_extracted"
        )
        # Preparing dlc dfs for input to SimBA project
        os.makedirs(simba_in_dir, exist_ok=True)
        simba_in_fp = os.path.join(simba_in_dir, f"{name}.csv")
        # TODO: order mousemarked and mouseunmarked correctly as 1 and 2
        # Selecting bodyparts for SimBA (8 bpts, 2 indivs)
        df = read_feather(dlc_fp)
        df = select_cols(df, configs_fp)
        # Saving dlc frame to place in the SimBA features extraction df
        index = df.index
        # Saving as csv
        df.to_csv(simba_in_fp)
        # Running SimBA env and script to run SimBA feature extraction
        outcome += run_extract_features_script(simba_dir, simba_in_dir, configs_dir)
        # Reading SimBA feature extraction csv (to select column and convert to feather)
        simba_out_fp = os.path.join(features_from_dir, f"{name}.csv")
        df = pd.read_csv(simba_out_fp, header=0, index_col=0)
        # Setting index to same as dlc preprocessed df
        df.index = index
        # Saving SimBA extracted features df as feather
        write_feather(df, out_fp)
        # Removing temp dir
        if remove_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
        # Returning outcome
        return outcome


#####################################################################
#               PREPARE FOR SIMBA
#####################################################################


def select_cols(
    df: pd.DataFrame,
    configs_fp: str,
) -> pd.DataFrame:
    """
    Selecting given DLC columns to input to SimBA.

    Parameters
    ----------
    df : pd.DataFrame
        DLC dataframe.
    configs_fp : str
        Configs dict.

    Returns
    -------
    pd.DataFrame
        DLC dataframe with selected columns.
    """
    # Getting necessary config parameters
    configs = read_configs(configs_fp)
    configs_filt = configs.user.extract_features
    indivs = configs_filt.individuals
    bpts = configs_filt.bodyparts
    # Checking that the bodyparts are all valid
    check_bpts_exist(bpts, df)
    # Selecting given columns
    idx = pd.IndexSlice
    df = df.loc[:, idx[:, indivs, bpts]]
    # returning df
    return df


def run_extract_features_script(
    simba_dir: str,
    dlc_dir: str,
    configs_dir: str,
) -> str:
    """
    Running the custom SimBA script to take the prepared DLC dataframe as input and
    create the features extracted dataframe.

    A custom SimBA script must be run in a separate custom conda environment because SimBA
    cannot be installed in the same environment as DEEPLABCUT (and also uses Python 3.6 -
    which is old).

    Parameters
    ----------
    simba_dir : str
        SimBA project directory.
    dlc_dir : str
        Prepared DLC dataframes directory. SimBA imports the entire directory.
        If only one file is being processed, put that file in a separate folder.
    configs_dir : str
        Directory path of config files corresponding to DLC dataframes in dlc_dir.
        For each DLC dataframe file, there should be a config file with the same name.
    """
    cmd = [
        "conda",
        "run",
        "-n",
        "simba_env",
        "python",
        "-m",
        "simba_package.extract_features",
        simba_dir,
        dlc_dir,
        configs_dir,
    ]
    run_subprocess_fstream(cmd)


def remove_bpts_cols(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Drops the bodyparts columns from the SimBA features extractions dataframes.
    Because bodypart coordinates should not be a factor in behavioural classification.

    Parameters
    ----------
    df : pd.DataFrame
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
    return df.iloc[:, n:]
