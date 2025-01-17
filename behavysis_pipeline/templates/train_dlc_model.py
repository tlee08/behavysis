# %% [markdown]
# # DLC Model Creation and Training Script
# 
# **RUNNING THIS SCRIPT:** Run this script with the DEEPLABCUT conda environment.
# 
# This script generates a DLC multi-animal model, which can even be used for single animals (in fact, training and inference is both faster too).
# 
# 
# The script follows the below steps, which are almost identical to the prescribed process [here](https://deeplabcut.github.io/DeepLabCut/docs/maDLC_UserGuide.html):
# 1. Create a DLC project folder and config file. The config file stores the model's data, training, and inference configurations.
# 1. Manually change the following parameters in the config file:
#     <!-- * `identity: true` (as we can identify each animal uniquely across frames) -->
#     * `individuals`: name for each animal (e.g. `mouse1`)
#     * `uniquebodyparts`: parts in the arena that are NOT the animal (e.g. `TopLeft`, `ColourMarking`)
#     * `multianimalbodyparts`: bodyparts for an animal (e.g. `Nose`)
#     * `numframes2pick`: The number of frames to extract from each video for labeling. A rule of thumb is ~500 frames overall is sufficient to train a model well.
#     <!-- * `batch_size: 32` (Speeds up computation for better GPUs). -->
# 1. Load videos to be used for training into the project's `videos` folder and update the config file with a list of these videos.
# 1. Randomly extract `n` (user specified) frames from each video and store in the `labeled-data` folder.
#     * NOTE: it can be useful to trim videos and import these to the project to get frames that you'd particularly like to label (e.g. close interaction in social experiments).
# 1. Downsample all frames to `960 x 540 px` (or the resolution you'd like). Also update config file to reflect this resolution.
# 1. Manually label frames
# 1. Create combined training dataset from labeled frames
# 1. Run training. The following training parameters are usually ideal:
#     * `saveiter = 5000`
#     * `maxiter = 50000`
# 1. Evaluate statistic (optional - gives MAE but difficult to interrogate this single statistic).
# 1. Test on novel video(s) and manually inspect tracking.
# 
# NOTE: for experiments with multiple animals, the following parameters are usually ideal:
# * TODO: in pose_inference.yaml
# 

# %%
import os
import re

import cv2
import deeplabcut
import yaml
import numpy as np
import pandas as pd

# %% [markdown]
# ## Specify project folder and name
# 
# DLC models are usually stored in the `Z:\PRJ-BowenLab\PRJ-BowenLab\DeepLabCut-Projects` folder.
# 
# 

# %%
# Don't need to change
root_dir = r"/home/linux1/Desktop/models_training/EPM"

# Change
proj_name = "EPM_BLACK_MICE_960px"

# Don't need to change
experimenter = "BowenLab"

# Can modify if running multiple GPU's at once
gputouse = 1

# Is a placeholder video
# This is needed to make the project
# but will be replaced by actual videos
placeholder_videos = [r"Z:\PRJ-BowenLab\PRJ-BowenLab\DeepLabCut-Projects\placeholder.mp4"]

# Downsampling size for training frames
# 960 x 540 is a good size
res_width = 960
res_height = 540

# DON'T CHANGE BELOW THIS LINE
proj_dir = os.path.join(root_dir, proj_name)
config_fp = os.path.join(proj_dir, "config.yaml")

os.path.isfile(config_fp)


# %% [markdown]
# ## Creating project
# 
# NOTE: don't need to run if project is already created.
# 

# %%
# Only run if porject doesn't exist yet
if os.path.exists(proj_dir):
    print(f"NOT making project because it already exists: {proj_dir}")
else:
    # Creating project
    temp_config_fp = deeplabcut.create_new_project(
        project=proj_name,
        experimenter=experimenter,
        videos=placeholder_videos,
        working_directory=root_dir,
        copy_videos=True,
        multianimal=True
    )
    # Renaming project to just proj_name
    os.rename(
        src=os.path.dirname(temp_config_fp),
        dst=proj_dir
    )
    # Updating config file with:
    # animals are identifiable, updated project path, and batch size
    deeplabcut.auxiliaryfunctions.edit_config(
        config_fp,
        {
            "identity": True,
            "project_path": proj_dir,
            "batch_size": 32,
        },
    )
    # Remove placeholding videos from project videos folder
    for vid_fp in placeholder_videos:
        os.remove(os.path.join(proj_dir, "videos", vid_fp))


# %% [markdown]
# ## 1. Manually change config file parameters
# 
# **ATTENTION**
# 
# Manually update the following parameters in the `config.yaml` file.
# * `individuals`: name for each animal (e.g. `mouse1`)
# * `uniquebodyparts`: parts in the arena that are NOT the animal (e.g. `TopLeft`, `ColourMarking`)
# * `multianimalbodyparts`: bodyparts for an animal (e.g. `Nose`)
# * `numframes2pick`: The number of frames to extract from each video for labeling. A rule of thumb is ~500 frames overall is sufficient to train a model well.
# 

# %% [markdown]
# # Import training videos
# 
# **ATTENTION**
# 
# Copy training videos to the `<proj_dir>\videos` folder.
# 
# Then run the following code block to update the config file with the videos.
# 

# %%
def update_config_videos(proj_dir):
    # Getting folder names for videos and labeled data
    videos_dir = os.path.join(proj_dir, "videos")
    labeled_dir = os.path.join(proj_dir, "labeled-data")
    video_sets = {}
    # For each video, store vid dims in video_configs
    for j in os.listdir(videos_dir):
        vid_fp = os.path.join(videos_dir, j)
        # Getting video dimensions
        cap = cv2.VideoCapture(vid_fp)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Adding to video_sets dict
        video_sets[vid_fp] = {"crop": f"0, {width}, 0, {height}"}
        # Closing video
        cap.release()
    # For all labeled-data frames (corresponding to videos)
    # Overwrites the video dimensions because these are the actual frames
    # Used for training
    for i in os.listdir(labeled_dir):
        # Not considering labeled data
        if re.search("_labeled$", i):
            continue
        # Getting one of the image frames in the
        # labeled data dict (to get video dimensions)
        fp_ls = [
            j
            for j in os.listdir(os.path.join(labeled_dir, i))
            if re.search("\.png$", j)
        ]
        # If no frames, skip
        if len(fp_ls) == 0:
            continue
        # Getting frame fp and video fp
        vid_fp = os.path.join(videos_dir, f"{i}.mp4")
        png_fp = os.path.join(labeled_dir, i, fp_ls[0])
        # Getting video dimensions
        height, width, ch = cv2.imread(png_fp).shape
        video_sets[vid_fp] = {"crop": f"0, {width}, 0, {height}"}
    # Updating configs file with video_sets
    deeplabcut.auxiliaryfunctions.edit_config(config_fp, {"video_sets": video_sets})

update_config_videos(proj_dir)

# # Regular DLC implementation (does not consider extracted frame dimensions)
# deeplabcut.add_new_videos(
#     config_path,
#     r"z:\PRJ-BowenLab\PRJ-BowenLab\DeepLabCut-Projects\SFC_WHITE_MICE\videos",
#     copy_videos=True,
# )


# %% [markdown]
# ## Extract frames
# 
# Randomly extract `n` (user specified) frames from each video and store in the `labeled-data` folder.
# 
# NOTE: edit the `numframes2pick` value in `configs.yaml` to change the number of frames extracted.
# ~500 frames overall is sufficient to train a model well.
# 
# NOTE: it can be useful to trim videos and import these to the project to get frames that you'd particularly like to label (e.g. close interaction in social experiments).
# 

# %%
# EXTRACTING FRAMES

# Getting folder names for videos and labeled data
videos_dir = os.path.join(proj_dir, "videos")
labeled_dir = os.path.join(proj_dir, "labeled-data")
# Get `n` from config file
with open(config_fp, "r") as f:
    config = yaml.safe_load(f)
n = config["numframes2pick"]
# For each video, extract `n` frames
for i in os.listdir(videos_dir):
    # Getting video fp
    vid_name = os.path.splitext(i)[0]
    vid_fp = os.path.join(videos_dir, i)
    vid_labeled_dir = os.path.join(labeled_dir, vid_name)
    # Opening video
    print(f"Extracting {n} frames from {vid_name}")
    vid = cv2.VideoCapture(vid_fp)
    # Getting total frames in video
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    # Getting `k` random frames
    frame_ids = np.random.choice(total_frames, n, replace=False).astype(int)
    # Creating folder for video
    os.makedirs(vid_labeled_dir, exist_ok=True)
    # For each frame in randomly selected set, save frame
    for j in sorted(frame_ids):
        print(f"    saving frame {j:06} ... ")
        # Seeking to frame
        vid.set(cv2.CAP_PROP_POS_FRAMES, j)
        # Reading frame
        ret, frame = vid.read()
        # Saving frame
        if ret:
            cv2.imwrite(os.path.join(vid_labeled_dir, f"img{j:06}.png"), frame)
    # Closing video
    vid.release()

# # Regular DLC implementation (any error will entirely halt the process)
# deeplabcut.extract_frames(
#     config_path,
#     mode="automatic",  # "automatic"/"manual"
#     algo='uniform',  # "uniform"/"kmeans"
#     userfeedback=False,  # True/False
#     crop=False  # keep as False
# )


# %% [markdown]
# ## Downsample frames
# 
# Resolution to downsample to is specified at start of script with `res_width` and `res_height` variables.
# 
# Usually, `960 x 540 px` is good for downsampling.

# %%
# DOWNSAMPLING FRAMES
# NOTE: also resizes the labels if necessary

# Defining DLC HDF key
# Sometimes changes from "keypoints" to "with_missing"
# Check keys with pd.HDFStore(fp, "r").keys()
HDF_KEY = "keypoints"

idx = pd.IndexSlice
labeled_dir = os.path.join(proj_dir, "labeled-data")
# Reading each video folder of image
for i in os.listdir(labeled_dir):
    vid_labeled_dir = os.path.join(labeled_dir, i)
    print(f"- {i}")
    # Reading CollectedData keypoints (if exists)
    h5_fp = os.path.join(vid_labeled_dir, f"CollectedData_{experimenter}.h5")
    h5 = None
    if os.path.isfile(h5_fp):
        h5 = pd.read_hdf(h5_fp, key=HDF_KEY)
    # For each image in video folder, resizing (and keypoints if necessary)
    for j in os.listdir(vid_labeled_dir):
        # Skip if not .png
        if not os.path.splitext(j)[1] == ".png":
            continue
        # Getting image fp
        img_fp = os.path.join(vid_labeled_dir, j)
        # Reading image
        img = cv2.imread(img_fp)
        # Getting scale factors (for keypoints resizing)
        x_scale = res_width / img.shape[1]
        y_scale = res_height / img.shape[0]
        # Resizing image
        img = cv2.resize(img, (res_width, res_height), interpolation=cv2.INTER_AREA)
        # Saving image
        cv2.imwrite(img_fp, img)
        # Skip keypoints resizing if no h5 data
        if h5 is None:
            continue
        # Getting row index
        row_idx = ("labeled-data", i, j)
        # Skip if row_idx not in h5
        if row_idx not in h5.index:
            continue
        # Resizing image's keypoint coords in h5_in
        h5_row = h5.loc[row_idx]
        # Note header is ["scorer", "individuals", "bodyparts", "coords"]
        # Resizing all x coords
        h5_row.loc[idx[:, :, :, "x"]] = (
            h5_row.loc[idx[:, :, :, "x"]] * x_scale
        )
        # Resizing all y coords
        h5_row.loc[idx[:, :, :, "y"]] = (
            h5_row.loc[idx[:, :, :, "y"]] * y_scale
        )
    # Saving h5 keypoints data (intermediately within loop)
    h5.to_hdf(h5_fp, key=HDF_KEY, mode="w")

# Updating config file with frame/video downsampled widths and heights
update_config_videos(proj_dir)


# %% [markdown]
# ## Manually label frames
# 

# %% [markdown]
# deeplabcut.label_frames(config_fp)
# 

# %% [markdown]
# ## Create training dataset
# 

# %%
deeplabcut.create_training_dataset(config_fp)

# deeplabcut.create_multianimaltraining_dataset(config_fp)


# %% [markdown]
# ## Editing training configs
# Training configs are in the `dlc-models-pytorch/.../train/pytorch_config.yaml` file.
# 

# %%
# At the end of the file, ensure the following parameters
"""
snapshots:
    max_snapshots: 5
    save_epochs: 50
    save_optimizer_state: false
train_settings:
  batch_size: 16
  dataloader_workers: 1
  dataloader_pin_memory: true
  display_iters: 100
  epochs: 400
  seed: 42
"""



# %% [markdown]
# ## Train model
# 

# %%
deeplabcut.train_network(
    config_fp,
    shuffle=1,
    trainingsetindex=0,
    gputouse=gputouse,
    max_snapshots_to_keep=5,
    autotune=False,
    displayiters=100,
    # saveiters=5000,
    save_epochs=50,
    # maxiters=50000, # Can change - 50000 is good
    epochs=400,
    allow_growth=True,
)

# %% [markdown]
# ## Evaluate model
# 
# Optional - this gives a Mean Absolute Error, which is difficult to interrogate.
# 
# It is advisable to instead run the model on some novel videos and inspect its performance by eye.
# 

# %%
deeplabcut.evaluate_network(config_fp, plotting=False)


# %% [markdown]
# ## Test on novel video(s) and manually inspect tracking
# 
# Firstly, make a folder in `proj_dir` called `novel_videos` and add some novel videos.
# 
# Then run the following code block, which runs the model on these video.
# 
# Inspect these videos and if performance is not satisfactory, label more frames and rerun training.
# 
# Notes for inspection:
# * Importantly, do bodypoints track well.
# * For multi-animal experiments, do points assemble to a single animal well (even if the identity is incorrect),
# * For multi-animal experiments, don't worry about swapping identities - a postprocessing step is done in our pipeline which fixes the identities to the markings/non-markings of each animal.
# 

# %%
# ProcessVidMixin.process_vid(
#     "Z:PRJ-BowenLab\\PRJ-BowenLab\\DeepLabCut-Projects\\RAT_3C_960px\\novel_videos\\13.mp4",
#     "Z:PRJ-BowenLab\\PRJ-BowenLab\\DeepLabCut-Projects\\RAT_3C_960px\\novel_videos\\13_d.mp4",
#     width_px=350,
#     height_px=960,
# )

deeplabcut.analyze_videos(
    config=config_fp,
    videos=os.path.join(proj_dir, "novel_videos"),
    videotype=".mp4",
    destfolder=os.path.join(proj_dir, "novel_videos"),
    auto_track=True,
    gputouse=gputouse,
    save_as_csv=False,
    calibrate=False,
    identity_only=False,
    allow_growth=True,
)


# %%
deeplabcut.create_labeled_video(
    config=config_fp,
    videos=os.path.join(proj_dir, "novel_videos"),
    videotype=".mp4",
    color_by="individual",
    destfolder=os.path.join(proj_dir, "novel_videos"),
)


# %% [markdown]
# ## EXTRA CODE SNIPPETS
# 
# * Impute all points from first for corners
# 

# %%
import os
import pandas as pd

HDF_KEY = "keypoints"

labeled_dir = os.path.join(proj_dir, "labeled-data")
idx = pd.IndexSlice
corner_bpts = [
    "FarTopLeft",
    "FarTopRight",
    "MiddleUpperLeft",
    "MiddleUpperRight",
    "MiddleLowerLeft",
    "MiddleLowerRight",
    "FarBottomLeft",
    "FarBottomRight",
]
for i in os.listdir(labeled_dir):
    h5_fp = os.path.join(labeled_dir, i, f"CollectedData_{experimenter}.h5")
    if not os.path.isfile(h5_fp):
        continue
    print(f"- {i}")
    h5 = pd.read_hdf(h5_fp, key=HDF_KEY)
    # FOR EACH CORNER BODYPOINT
    # # Making values nan
    # h5.loc[:, idx[:, :, corner_bpts]] = np.nan
    # # imputing missing values (ffill)
    # h5.loc[:, idx[:, :, corner_bpts]] = h5.loc[:, idx[:, :, corner_bpts]].ffill()
    # Bounding the values to (0+5, res_height-5) and (0+5, res_width-5)
    # THIS SEEMS TO WORK BEST BECAUSE IT JUST ENSURES THE POINTS ARE WITHIN THE FRAME (AUTOMATICALLY)
    h5.loc[:, idx[:, :, corner_bpts, "y"]] = h5.loc[:, idx[:, :, corner_bpts, "y"]].clip(
        5, res_height - 5
    )
    h5.loc[:, idx[:, :, corner_bpts, "x"]] = h5.loc[:, idx[:, :, corner_bpts, "x"]].clip(
        5, res_width - 5
    )
    # saving h5
    h5.to_hdf(h5_fp, key=HDF_KEY, mode="w")


# %%



