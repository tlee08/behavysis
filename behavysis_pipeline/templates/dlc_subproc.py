import deeplabcut

for video in {{ in_fp_ls }}:
    try:
        deeplabcut.analyze_videos(
            config=r"{{ model_fp}}",
            videos=[video],
            videotype="mp4",
            destfolder=r"{{ dlc_out_dir }}",
            gputouse={{ gputouse }},
            save_as_csv=False,
            calibrate=False,
            identity_only=False,
            allow_growth=False,
        )
    except Exception as e:
        print('Error', e)
