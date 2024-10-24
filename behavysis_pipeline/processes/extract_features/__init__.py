"""
_summary_
"""

import os

import pandas as pd
from behavysis_core.constants import FeaturesCN, FeaturesIN
from behavysis_core.data_models.experiment_configs import ExperimentConfigs
from behavysis_core.df_mixins.df_io_mixin import DFIOMixin
from behavysis_core.mixins.features_df_mixin import FeaturesDfMixin
from behavysis_core.mixins.io_mixin import IOMixin
from behavysis_core.mixins.keypoints_df_mixin import KeypointsMixin
from behavysis_core.mixins.multiproc_mixin import MultiprocMixin
from behavysis_core.mixins.subproc_mixin import SubprocMixin

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
    """__summary__"""

    @staticmethod
    @IOMixin.overwrite_check()
    def extract_features(
        dlc_fp: str,
        out_fp: str,
        configs_fp: str,
        temp_dir: str,
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
        overwrite : bool
            Whether to overwrite the out_fp file (if it exists).

        Returns
        -------
        str
            The outcome of the process.
        """
        outcome = ""
        # Getting directory and file paths
        name = IOMixin.get_name(dlc_fp)
        cpid = MultiprocMixin.get_cpid()
        configs_dir = os.path.split(configs_fp)[0]
        simba_in_dir = os.path.join(temp_dir, f"input_{cpid}")
        simba_dir = os.path.join(temp_dir, f"simba_proj_{cpid}")
        features_from_dir = os.path.join(
            simba_dir, "project_folder", "csv", "features_extracted"
        )
        # Preparing dlc dfs for input to SimBA project
        os.makedirs(simba_in_dir, exist_ok=True)
        simba_in_fp = os.path.join(simba_in_dir, f"{name}.csv")
        # Selecting bodyparts for SimBA (8 bpts, 2 indivs)
        df = KeypointsMixin.read_feather(dlc_fp)
        df = select_cols(df, configs_fp)
        # Saving dlc frame to place in the SimBA features extraction df
        index = df.index
        # Need to remove index name for SimBA to import correctly
        df.index.name = None
        # Saving as csv
        df.to_csv(simba_in_fp)
        # Removing simba folder (if it exists)
        IOMixin.silent_rm(simba_dir)
        # Running SimBA env and script to run SimBA feature extraction
        outcome += run_simba_subproc(
            simba_dir, simba_in_dir, configs_dir, temp_dir, cpid
        )
        # Exporting SimBA feature extraction csv to feather
        simba_out_fp = os.path.join(features_from_dir, f"{name}.csv")
        export_2_feather(simba_out_fp, out_fp, index)
        # Removing temp folders (simba_in_dir, simba_dir)
        IOMixin.silent_rm(simba_in_dir)
        IOMixin.silent_rm(simba_dir)
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
    configs = ExperimentConfigs.read_json(configs_fp)
    configs_filt = configs.user.extract_features
    indivs = configs.get_ref(configs_filt.individuals)
    bpts = configs.get_ref(configs_filt.bodyparts)
    # Checking that the bodyparts are all valid
    KeypointsMixin.check_bpts_exist(df, bpts)
    # Selecting given columns
    idx = pd.IndexSlice
    df = df.loc[:, idx[:, indivs, bpts]]
    # returning df
    return df


def run_simba_subproc(
    simba_dir: str,
    dlc_dir: str,
    configs_dir: str,
    temp_dir: str,
    cpid: int,
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
    # Saving the script to a file
    script_fp = os.path.join(temp_dir, f"simba_subproc_{cpid}.py")
    IOMixin.save_template(
        "simba_subproc.py",
        "behavysis_pipeline",
        "templates",
        script_fp,
        simba_dir=simba_dir,
        dlc_dir=dlc_dir,
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
    # SubprocMixin.run_subproc_fstream(cmd)
    SubprocMixin.run_subproc_console(cmd)
    # Removing the script file
    IOMixin.silent_rm(script_fp)
    return "Ran SimBA feature extraction script.\n"


def remove_bpts_cols(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Drops the bodyparts columns from the SimBA features extractions dataframes.
    Because bodypart coordinates should not be a factor in behaviour classification.

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


def export_2_feather(in_fp: str, out_fp: str, index: pd.Index) -> str:
    """
    __summary__
    """
    df = pd.read_csv(in_fp, header=0, index_col=0)
    # Setting index to same as dlc preprocessed df
    df.index = index
    # Setting index and column level names
    df.index.names = DFIOMixin.enum2tuple(FeaturesIN)
    df.columns.names = DFIOMixin.enum2tuple(FeaturesCN)
    # Checking df
    FeaturesDfMixin.check_df(df)
    # Saving SimBA extracted features df as feather
    DFIOMixin.write_feather(df, out_fp)
    # Returning outcome
    return "Exported SimBA features to feather.\n"
