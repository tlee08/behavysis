import logging
from io import BytesIO

import numpy as np
import pandas as pd
from behavysis_core.constants import KeypointsCN, KeypointsIN
from behavysis_core.data_models.experiment_configs import ExperimentConfigs
from behavysis_core.mixins.df_io_mixin import DFIOMixin

from behavysis_pipeline.processes.calculate_params import (
    CalculateParams,
    Model_check_exists,
    Model_stop_frame,
)


def make_dlc_df_for_dur(sections_params_ls, columns):
    sections_ls = []
    for size, loc, scale in sections_params_ls:
        section = np.random.normal(loc=loc, scale=scale, size=(size, len(columns)))
        section = np.clip(section, 0, 1)
        sections_ls.append(section)
    # Making each df for (x, y, l)
    dlc_df_l = pd.DataFrame(np.concatenate(sections_ls, axis=0), columns=columns)
    dlc_df_x = pd.DataFrame(0, index=dlc_df_l.index, columns=columns)
    dlc_df_y = pd.DataFrame(0, index=dlc_df_l.index, columns=columns)
    # Combining each (x, y, l) df
    dlc_df = pd.concat(
        (dlc_df_x, dlc_df_y, dlc_df_l),
        axis=1,
        keys=("x", "y", "likelihood"),
        names=["coords", "bodyparts"],
    )
    # Wrangling column names and order
    columns_df = dlc_df.columns.to_frame(index=False)
    columns_df["scorer"] = "scorer"
    cols_subsect = np.random.choice(columns, 3)
    columns_df["individuals"] = columns_df["bodyparts"].apply(
        lambda x: ("animal" if np.isin(x, cols_subsect) else "single")
    )
    columns_df = columns_df[["scorer", "individuals", "bodyparts", "coords"]]
    dlc_df.columns = pd.MultiIndex.from_frame(columns_df)
    dlc_df = dlc_df.sort_index(level=["individuals", "bodyparts"], axis=1)
    # Setting index and column level names
    dlc_df.index.name = DFIOMixin.enum_to_list(KeypointsIN)
    dlc_df.columns.name = DFIOMixin.enum_to_list(KeypointsCN)
    # Returning dlc_df
    return dlc_df


def test_start_frame():
    # Defining testing params
    fps = 15
    before_size = 500

    # Making configs
    configs = ExperimentConfigs()
    configs.user.calculate_params.start_frame = Model_check_exists(
        bodyparts=["b", "c", "e"],
        window_sec=1,
        pcutoff=0.9,
    )
    configs.auto.formatted_vid.fps = fps
    # Writing to BytesIO to mimmick file API
    configs_io = "configs.json"
    configs.write_json(configs_io)

    # Making random normal values in a given range (above and below pcutoff)
    dlc_df = make_dlc_df_for_dur(
        [
            (before_size, 0.75, 0.1),
            (1000, 0.95, 0.1),
            (200, 0.6, 0.1),
        ],
        ["a", "b", "c", "d", "e", "f", "g"],
    )
    dlc_df_io_in = BytesIO()
    dlc_df.to_feather(dlc_df_io_in)

    # Testing start_frame func
    output = CalculateParams.start_frame(dlc_df_io_in, configs_io)

    # Getting updated configs
    configs = ExperimentConfigs.read_json(configs_io)
    logging.info(configs.auto.model_dump_json(indent=2))

    # Asserting
    assert np.abs(configs.auto.start_frame - before_size) < 10


def test_stop_frame():
    # Defining testing params
    fps = 15
    start_frame = 500
    dur_sec = 60
    total_frames = 1000

    # Making configs
    configs = ExperimentConfigs()
    configs.user.calculate_params.stop_frame = Model_stop_frame(
        dur_sec=dur_sec,
    )
    configs.auto.formatted_vid.fps = fps
    configs.auto.start_frame = start_frame
    configs.auto.formatted_vid.total_frames = total_frames
    # Writing to BytesIO to mimmick file API
    configs_io = "configs.json"
    configs.write_json(configs_io)

    # Making random normal values in a given range (above and below pcutoff)
    dlc_df = make_dlc_df_for_dur(
        [
            (500, 0.75, 0.1),
            (1000, 0.95, 0.1),
            (200, 0.6, 0.1),
        ],
        ["a", "b", "c", "d", "e", "f", "g"],
    )
    dlc_df_io_in = BytesIO()
    dlc_df.to_feather(dlc_df_io_in)

    # Testing start_frame func
    output = CalculateParams.stop_frame(dlc_df_io_in, configs_io)

    # Getting updated configs
    configs = ExperimentConfigs.read_json(configs_io)
    logging.info(configs.auto.model_dump_json(indent=2))

    # Asserting
    assert configs.auto.stop_frame == start_frame + dur_sec * fps


def test_exp_dur():
    # Defining testing params
    fps = 15
    before_size = 500
    during_size = 1000
    after_size = 200

    # Making configs
    configs = ExperimentConfigs()
    configs.user.calculate_params.exp_dur = Model_check_exists(
        bodyparts=["b", "c", "e"],
        window_sec=1,
        pcutoff=0.9,
    )
    configs.auto.formatted_vid.fps = fps
    # Writing to BytesIO to mimmick file API
    configs_io = "configs.json"
    configs.write_json(configs_io)

    # Making random normal values in a given range (above and below pcutoff)
    dlc_df = make_dlc_df_for_dur(
        [
            (before_size, 0.75, 0.1),
            (during_size, 0.95, 0.1),
            (after_size, 0.6, 0.1),
        ],
        ["a", "b", "c", "d", "e", "f", "g"],
    )
    dlc_df_io_in = BytesIO()
    dlc_df.to_feather(dlc_df_io_in)

    # Testing start_frame func
    output = CalculateParams.exp_dur(dlc_df_io_in, configs_io)

    # Getting updated configs
    configs = ExperimentConfigs.read_json(configs_io)
    logging.info(configs.auto.model_dump_json(indent=2))

    # Asserting
    assert np.abs(configs.auto.exp_dur_frames - during_size) < 20


# def test_format_vid():
#     proj_dir = os.path.join(".")
#     proj = Project(proj_dir)
#     proj.format_vid(
#         funcs=(
#             FormatVid.format_vid,
#             FormatVid.vid_metadata,
#         ),
#         overwrite=True,
#     )
#     # Add assertions here based on what you expect to happen


# def test_run_dlc():
#     proj_dir = os.path.join(".")
#     proj = Project(proj_dir)
#     proj.run_dlc(
#         gputouse=None,
#         overwrite=True,
#     )
#     # Add assertions here based on what you expect to happen


# def test_calculate_params():
#     proj_dir = os.path.join(".")
#     proj = Project(proj_dir)
#     proj.calculate_params(
#         (
#             CalculateParams.start_frame,
#             CalculateParams.stop_frame,
#             CalculateParams.px_per_mm,
#         )
#     )
#     # Add assertions here based on what you expect to happen


# def test_preprocess():
#     proj_dir = os.path.join(".")
#     proj = Project(proj_dir)
#     proj.preprocess(
#         (
#             Preprocess.start_stop_trim,
#             Preprocess.interpolate,
#             Preprocess.refine_ids,
#         ),
#         overwrite=True,
#     )
#     # Add assertions here based on what you expect to happen
