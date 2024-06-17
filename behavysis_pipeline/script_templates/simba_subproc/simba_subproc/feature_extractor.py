import configparser
import json
import os
import shutil

import pandas as pd
from simba.feature_extractors.feature_extractor_16bp import ExtractFeaturesFrom16bps
from simba.outlier_tools.outlier_corrector_location import OutlierCorrecterLocation
from simba.outlier_tools.outlier_corrector_movement import OutlierCorrecterMovement
from simba.outlier_tools.skip_outlier_correction import OutlierCorrectionSkipper
from simba.utils.config_creator import ProjectConfigCreator

#################################################
#     UPDATING SIMBA CONFIGS INI FILE
#################################################


class FeatureExtractor:
    """__summary__"""

    @staticmethod
    def simba_update_configs(simba_dir, update_dict):
        """
        Updates project_config.ini file.
        """
        simba_configs_fp = os.path.join(
            simba_dir, "project_folder", "project_config.ini"
        )
        # Making ConfigParser instance
        config = configparser.ConfigParser()
        # Reading in existing simba project configs
        config.read(simba_configs_fp)
        # Updating with given configs
        config.read_dict(update_dict)
        # Writing updated configs to file
        with open(simba_configs_fp, "w", encoding="utf-8") as f:
            config.write(f)

    #################################################
    #       MAKE SIMBA PROJECT
    #################################################

    @staticmethod
    def simba_make_proj(proj_dir, behavs_ls):
        """
        Pose number is from:
            - https://github.com/sgoldenlab/simba/blob/master/simba/pose_configurations/configuration_names/pose_config_names.csv
            - https://github.com/sgoldenlab/simba/blob/master/simba/pose_configurations/bp_names/bp_names.csv
        """
        ProjectConfigCreator(
            project_path=os.path.split(proj_dir)[0],
            project_name=os.path.split(proj_dir)[1],
            target_list=behavs_ls,
            pose_estimation_bp_cnt="16",
            body_part_config_idx=6,  # bp_names.csv or pose_config_names.csv row minus 1
            animal_cnt=2,
            file_type="csv",  # "parquet", "csv"
        )

    #################################################
    #            IMPORTING DATA TO SIMBA
    #################################################

    @staticmethod
    def simba_import_file(simba_dir, dlc_dir):
        """
        DLC csv must already be in simba csv readable format.
        Similar to simba.import_multiple_dlc_tracking_csv_file()
        """
        for fp in os.listdir(dlc_dir):
            name = FeatureExtractor.get_name(fp)
            src_fp = os.path.join(dlc_dir, f"{name}.csv")
            dst_fp = os.path.join(
                simba_dir, "project_folder", "csv", "input_csv", f"{name}.csv"
            )
            os.makedirs(os.path.split(dst_fp)[0], exist_ok=True)
            # Copying video mp4 and dlc csv to simba project dir
            shutil.copyfile(src_fp, dst_fp)

    #################################################
    #    READING VIDEO INFO (DIMENSIONS) FOR SIMBA
    #################################################

    @staticmethod
    def simba_set_dims(simba_dir, configs_dir):
        """
        Similar to `simba.set_video_parameters()` but gets specific vals from each config file.
        """
        # simba_configs_fp = os.path.join(
        #     simba_dir, "project_folder", "project_config.ini"
        # )
        # set_video_parameters(config_path=simba_configs_fp, px_per_mm=PX_PER_MM, fps=FPS, resolution=RESOLUTION)

        # Initialising video dims df
        df = pd.DataFrame(
            columns=[
                "Video",
                "fps",
                "Resolution_width",
                "Resolution_height",
                "Distance_in_mm",
                "pixels/mm",
            ]
        )
        input_csv_dir = os.path.join(simba_dir, "project_folder", "csv", "input_csv")
        # Getting and saving the px/mm values to the df
        for fp in os.listdir(input_csv_dir):
            name = FeatureExtractor.get_name(fp)
            # Getting configs JSON
            configs_fp = os.path.join(configs_dir, f"{name}.json")
            with open(configs_fp, "r", encoding="utf-8") as f:
                configs = json.load(f)
            vid_configs = configs["auto"]["formatted_vid"]
            row = (
                pd.Series(
                    {
                        "Video": name,
                        "fps": vid_configs["fps"],
                        "Resolution_width": vid_configs["width_px"],
                        "Resolution_height": vid_configs["height_px"],
                        "Distance_in_mm": configs["user"]["calculate_params"][
                            "px_per_mm"
                        ]["dist_mm"],
                        "pixels/mm": configs["auto"]["px_per_mm"],
                    }
                )
                .to_frame()
                .transpose()
            )
            df = pd.concat([df, row], axis=0, ignore_index=True)
        # Storing the df in the simba project
        logs_fp = os.path.join(simba_dir, "project_folder", "logs", "video_info.csv")
        os.makedirs(os.path.split(logs_fp)[0], exist_ok=True)
        df.to_csv(logs_fp, index=None)
        return df

    #################################################
    #     SIMBA OUTLIER CORRECTION
    #################################################

    @staticmethod
    def simba_run_outlier_correction(simba_dir):
        """
        Movement and location criterion is from nose to tail-base of each mouse.
        Movement is delta of points between frames.
        Location is delta of points.
        The threshold is the criterion, C, multiplied by the median (or mean) of all frames.
        Any points above the threshold are set as the previously "valid" point.
        """
        simba_configs_fp = os.path.join(
            simba_dir, "project_folder", "project_config.ini"
        )
        OutlierCorrecterMovement(config_path=simba_configs_fp).run()
        OutlierCorrecterLocation(config_path=simba_configs_fp).run()

    @staticmethod
    def simba_skip_outlier_correction(simba_dir):
        """
        Skipping
        """
        simba_configs_fp = os.path.join(
            simba_dir, "project_folder", "project_config.ini"
        )
        OutlierCorrectionSkipper(config_path=simba_configs_fp).run()

    #################################################
    #     SIMBA EXTRACT FEATURES
    #################################################

    @staticmethod
    def simba_extract_features(simba_dir):
        """
        Extracting features
        """
        simba_configs_fp = os.path.join(
            simba_dir, "project_folder", "project_config.ini"
        )
        ExtractFeaturesFrom16bps(config_path=simba_configs_fp).run()

    #################################################
    #                LABEL FRAMES
    #################################################

    @staticmethod
    def simba_label_scoring(scored_fp, features_fp, targets_inserted_fp, behavs_ls):
        """
        Adding behaviour labels to features_extracted csv.
        DEPRICATED: store behav frame outcomes in different df.
        """
        # Reading in features_extracted csv
        df = pd.read_csv(features_fp, header=0, index_col=0)
        scored_df = pd.read_csv(scored_fp, header=0, index_col=0)
        # Adding labelled behaviour columns (if a column does not exist, imputes with 0)
        for behav in behavs_ls:
            try:
                df[behav] = scored_df[behav]
            except KeyError:
                df[behav] = 0
        # Writing to csv
        df.to_csv(targets_inserted_fp)

    @staticmethod
    def get_name(fp: str) -> str:
        """
        Given the filepath, returns the name of the file.
        The name is:
        ```
        <path_to_file>/<name>.<ext>
        ```
        """
        return os.path.splitext(os.path.basename(fp))[0]


def main() -> None:
    """
    Batch processes a BA_Project (without using the BA_project_env - simBA doesn't work with it)

    Assumes input is column-selected csv, and output is csv in simba_proj >> features_extracted dir.
    """
    # Getting directories from cmd args
    simba_dir = "{{ simba_dir }}"
    dlc_dir = "{{ dlc_dir }}"
    configs_dir = "{{ configs_dir }}"
    # Making SimBA project
    if not os.path.exists(simba_dir):
        FeatureExtractor.simba_make_proj(simba_dir, ["placeholder"])
    # Importing DLC dfs to SimBA project
    FeatureExtractor.simba_import_file(simba_dir, dlc_dir)
    # Setting video px per mm
    FeatureExtractor.simba_set_dims(simba_dir, configs_dir)
    # Outlier correction (skipping)
    FeatureExtractor.simba_skip_outlier_correction(simba_dir)
    # Extracting features
    FeatureExtractor.simba_extract_features(simba_dir)


if __name__ == "__main__":
    main()
