import marimo

__generated_with = "0.23.10"
app = marimo.App()

with app.setup:
    import contextlib
    import os
    import re
    import shutil
    import subprocess
    from pathlib import Path

    import cv2
    import deeplabcut
    import marimo as mo
    import numpy as np


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # DLC Model Creation and Training Script

    **RUNNING THIS SCRIPT:** Run this script with the DEEPLABCUT conda environment.

    This script generates a DLC multi-animal model, which can even be used for single animals (in fact, training and inference is both faster too).

    The script follows the below steps, which are almost identical to the prescribed process [here](https://deeplabcut.github.io/DeepLabCut/docs/maDLC_UserGuide.html):

    1. Create a DLC project folder and config file. The config file stores the model's data, training, and inference configurations.
    1. Manually change the following parameters in the config file:
       <!-- * `identity: true` (as we can identify each animal uniquely across frames) -->
       - `individuals`: name for each animal (e.g. `mouse1`)
       - `uniquebodyparts`: parts in the arena that are NOT the animal (e.g. `TopLeft`, `ColourMarking`)
       - `multianimalbodyparts`: bodyparts for an animal (e.g. `Nose`)
       - `numframes2pick`: The number of frames to extract from each video for labeling. A rule of thumb is ~500 frames overall is sufficient to train a model well.
       <!-- * `batch_size: 32` (Speeds up computation for better GPUs). -->
    1. Load videos to be used for training into the project's `videos` folder and update the config file with a list of these videos.
    1. Randomly extract `n` (user specified) frames from each video and store in the `labeled-data` folder.
       - NOTE: it can be useful to trim videos and import these to the project to get frames that you'd particularly like to label (e.g. close interaction in social experiments).
    1. Downsample all frames to `960 x 540 px` (or the resolution you'd like). Also update config file to reflect this resolution.
    1. Manually label frames
    1. Create combined training dataset from labeled frames
    1. Run training. The following training parameters are usually ideal:
       - `saveiter = 5000`
       - `maxiter = 50000`
    1. Evaluate statistic (optional - gives MAE but difficult to interrogate this single statistic).
    1. Test on novel video(s) and manually inspect tracking.

    NOTE: for experiments with multiple animals, the following parameters are usually ideal:

    - TODO: in pose_inference.yaml
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Specify project folder and name

    DLC models are usually stored in the `Z:\PRJ-BowenLab\PRJ-BowenLab\DeepLabCut-Projects` folder.
    """)
    return


@app.cell
def _():
    # CHANGE
    root_dir = "set/me/here"
    # CHANGE
    proj_name = "set me"
    # Don't need to change
    experimenter = "BowenLab"
    # Can modify if running multiple GPU's at once
    gputouse = 0
    # Downsampling size for training frames
    # 960 x 540 is a good size
    res_width = 960
    res_height = 540

    # DON'T CHANGE
    proj_dir = Path(root_dir) / proj_name
    config_fp = Path(proj_dir) / "config.yaml"

    config_fp.is_file()
    return (
        config_fp,
        experimenter,
        gputouse,
        proj_dir,
        proj_name,
        res_height,
        res_width,
        root_dir,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Creating project

    NOTE: don't need to run if project is already created.
    """)
    return


@app.cell
def _(
    config_fp,
    experimenter,
    proj_dir,
    proj_name,
    res_height,
    res_width,
    root_dir,
):
    # Only run if project doesn't exist yet
    if proj_dir.exists():
        print(f"NOT making project because it already exists: {proj_dir}")
    else:
        placeholder_fp = (
            Path(root_dir) / "placeholder_vid.mp4"
        )  # Making placeholder vid
        _cap = cv2.VideoWriter(
            placeholder_fp,
            cv2.VideoWriter_fourcc(*"mp4v"),
            15,
            (res_width, res_height),
        )
        black_frame = np.zeros((res_height, res_width, 3), dtype=np.uint8)
        for _ in range(15):
            _cap.write(black_frame)
        _cap.release()
        temp_config_fp = deeplabcut.create_new_project(
            project=proj_name,
            experimenter=experimenter,
            videos=[placeholder_fp],
            working_directory=root_dir,
            copy_videos=True,
            multianimal=True,
        )
        os.rename(src=Path(temp_config_fp).parent, dst=proj_dir)
        deeplabcut.auxiliaryfunctions.edit_config(
            config_fp,
            {"identity": True, "project_path": proj_dir, "batch_size": 8},
        )
        os.remove(placeholder_fp)
        os.remove(proj_dir / "videos" / "placeholder_vid.mp4")
        (proj_dir / "videos_raw").mkdir(parents=True, exist_ok=True)
        (proj_dir / "test_on_novels").mkdir(parents=True, exist_ok=True)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Manually change config file parameters

    **ATTENTION**

    Manually update the following parameters in the `config.yaml` file.

    - `individuals`: name for each animal (e.g. `mouse1`)
    - `uniquebodyparts`: parts in the arena that are NOT the animal (e.g. `TopLeft`, `ColourMarking`)
    - `multianimalbodyparts`: bodyparts for an animal (e.g. `Nose`)
    - `numframes2pick`: The number of frames to extract from each video for labeling. A rule of thumb is ~500 frames overall is sufficient to train a model well.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Import raw training videos

    **ATTENTION**

    Copy training videos to the `<proj_dir>\videos_raw` folder.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Downsampling videos and saving to the `videos` folder

    Uses ffmpeg to downsample videos.

    Set the `res_width` and `res_height` values in the 2nd top Python cell (that has all the other settings for training this model).
    """)
    return


@app.cell
def _(proj_dir, res_height, res_width):
    def downsample_vid(in_fp, out_fp, res_width, res_height) -> None:
        cmd = [
            "ffmpeg",
            "-i",
            in_fp,
            "-vf",
            f"scale={res_width}:{res_height}",
            "-c:v",
            "h264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-y",
            out_fp,
        ]
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd)

    for _vid_fp in os.listdir(proj_dir / "videos_raw"):
        downsample_vid(
            in_fp=proj_dir / "videos_raw" / _vid_fp,
            out_fp=proj_dir / "videos" / _vid_fp,
            res_width=res_width,
            res_height=res_height,
        )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Updating config file with the filepaths of our training videos

    This is required for DLC's extract frames step.
    It looks at the video filepaths in the config file and extracts frames from those videos for labeling.
    """)
    return


@app.cell
def _(config_fp, proj_dir):
    def update_config_videos(proj_dir) -> None:
        videos_dir = (
            proj_dir / "videos"
        )  # Getting folder names for videos and labeled data
        labeled_dir = proj_dir / "labeled-data"
        video_sets = {}
        for j in os.listdir(videos_dir):
            _vid_fp = videos_dir / j  # For each video, store vid dims in video_config
            _cap = cv2.VideoCapture(_vid_fp)
            width = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(
                _cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            )  # Getting video dimensions
            video_sets[_vid_fp] = {"crop": f"0, {width}, 0, {height}"}
            _cap.release()
        for i in os.listdir(labeled_dir):
            if re.search("_labeled$", i):  # Adding to video_sets dict
                continue
            fp_ls = [
                j for j in (labeled_dir / i).iterdir() if re.search("\\.png$", j)
            ]  # Closing video
            if len(fp_ls) == 0:
                continue  # For all labeled-data frames (corresponding to videos)
            _vid_fp = (
                videos_dir / f"{i}.mp4"
            )  # Overwrites the video dimensions because these are the actual frames
            png_fp = labeled_dir / i / fp_ls[0]  # Used for training
            height, width, _ch = cv2.imread(png_fp).shape
            video_sets[_vid_fp] = {
                "crop": f"0, {width}, 0, {height}",
            }  # Not considering labeled data
        deeplabcut.auxiliaryfunctions.edit_config(
            config_fp,
            {"video_sets": video_sets},
        )

    # # Regular DLC implementation (does not consider extracted frame dimensions)
    # deeplabcut.add_new_videos(
    #     config_path,
    #     r"z:\PRJ-BowenLab\PRJ-BowenLab\DeepLabCut-Projects\SFC_WHITE_MICE\videos",
    #     copy_videos=True,
    # )
    update_config_videos(proj_dir)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Extract frames

    Randomly extract `n` (user specified) frames from each video and store in the `labeled-data` folder.

    NOTE: edit the `numframes2pick` value in `config.yaml` to change the number of frames extracted.
    ~500 frames overall is sufficient to train a model well.

    NOTE: it can be useful to trim videos and import these to the project to get frames that you'd particularly like to label (e.g. close interaction in social experiments).
    """)
    return


@app.cell
def _(proj_dir):
    # EXTRACTING FRAMES
    videos_dir = proj_dir / "videos"
    # Getting folder names for videos and labeled data
    labeled_dir = proj_dir / "labeled-data"

    n = 5
    for i in videos_dir.iterdir():
        # For each video, extract `n` frames
        vid_name = i.stem
        _vid_fp = videos_dir / i  # Getting video fp
        vid_labeled_dir = labeled_dir / vid_name
        print(f"Extracting {n} frames from {vid_name}")
        vid = cv2.VideoCapture(_vid_fp)
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))  # Opening video
        frame_ids = np.random.choice(total_frames, n, replace=False).astype(int)
        vid_labeled_dir.mkdir(parents=True, exist_ok=True)
        for j in sorted(frame_ids):  # Getting total frames in video
            print(f"    saving frame {j:06} ... ")
            vid.set(cv2.CAP_PROP_POS_FRAMES, j)  # Getting `k` random frames
            ret, frame = vid.read()
            if ret:  # Creating folder for video
                cv2.imwrite(vid_labeled_dir / f"img{j:06}.png", frame)
        # # Regular DLC implementation (any error will entirely halt the process)
        # deeplabcut.extract_frames(
        #     config_path,
        #     mode="automatic",  # "automatic"/"manual"
        #     algo='uniform',  # "uniform"/"kmeans"
        #     userfeedback=False,  # True/False
        #     crop=False  # keep as False
        # )
        vid.release()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Label frames
    """)
    return


@app.cell
def _(config_fp):
    deeplabcut.label_frames(config_fp)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    TODO: check that all frames are labelled without deleting rows and images
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Create training dataset
    """)
    return


@app.cell
def _(config_fp):
    # deeplabcut.create_training_dataset(config_fp)

    deeplabcut.create_multianimaltraining_dataset(config_fp)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Train model

    Note the pytorch training config. These particular config are stored in the `dlc-models-pytorch/.../train/pytorch_config.yaml` file.
    """)
    return


@app.cell
def _(config_fp, gputouse):
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
        epochs=1000,
        allow_growth=True,
        pytorch_cfg_updates={
            "runner.gpus": [gputouse],
            "runner.snapshots.max_snapshots": 5,
            "runner.snapshots.save_epochs": 50,
            "runner.snapshots.save_optimizer_state": False,
            "train_settings.batch_size": 8,
            "train_settings.dataloader_workers": 1,
            "train_settings.dataloader_pin_memory": True,
            "train_settings.display_iters": 100,
            "train_settings.epochs": 1000,
            "train_settings.seed": 42,
        },
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Evaluate model

    Optional - this gives a Mean Absolute Error, which is difficult to interrogate.

    It is advisable to instead run the model on some novel videos and inspect its performance by eye.
    """)
    return


@app.cell
def _():
    # deeplabcut.evaluate_network(config_fp, plotting=False)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Test on novel video(s) and manually inspect tracking

    Firstly, make a folder in `proj_dir` called `novel_videos` and add some novel videos.

    Then run the following code block, which runs the model on these video.

    Inspect these videos and if performance is not satisfactory, label more frames and rerun training.

    Notes for inspection:

    - Importantly, do bodypoints track well.
    - For multi-animal experiments, do points assemble to a single animal well (even if the identity is incorrect),
    - For multi-animal experiments, don't worry about swapping identities - a postprocessing step is done in our pipeline which fixes the identities to the markings/non-markings of each animal.
    """)
    return


@app.cell
def _(config_fp, gputouse, proj_dir):
    novel_vids_dir = proj_dir / "test_on_novels"
    assert novel_vids_dir.exists()

    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree(novel_vids_dir / "out")
    (novel_vids_dir / "out").mkdir(parents=True, exist_ok=True)

    deeplabcut.analyze_videos(
        config=config_fp,
        videos=novel_vids_dir / "in",
        videotype=".mp4",
        destfolder=novel_vids_dir / "out",
        auto_track=True,
        gputouse=gputouse,
        save_as_csv=False,
        calibrate=False,
        identity_only=False,
        allow_growth=True,
        # torch_kwargs={
        #     "device": [gputouse],
        # },
    )
    return (novel_vids_dir,)


@app.cell
def _(config_fp, novel_vids_dir):
    deeplabcut.create_labeled_video(
        config=config_fp,
        videos=novel_vids_dir / "in",
        videotype=".mp4",
        color_by="individual",
        destfolder=novel_vids_dir / "out",
        overwrite=True,
    )
    return


if __name__ == "__main__":
    app.run()
