import os

from behavysis_pipeline.utils.io_utils import IOMixin


# TODO: what is happening here? Fix up for make_classifier script
def main(root_dir: str = ".", overwrite: bool = False) -> None:
    """
    Makes a script to build a BehavClassifier.

    Copies the `train_behav_classifier.py` script to `root_dir/behav_models`.
    """
    # Making the project root folder
    os.makedirs(os.path.join(root_dir, "behav_models"), exist_ok=True)
    # Copying the files to the project folder
    for i in ["train_behav_classifier.py"]:
        # Getting the file path
        dst_fp = os.path.join(root_dir, i)
        # If not overwrite and file exists, then don't overwrite
        if not overwrite and os.path.exists(dst_fp):
            continue
        # Saving the template to the file
        IOMixin.save_template(
            i,
            "behavysis_pipeline",
            "templates",
            dst_fp,
        )


if __name__ == "__main__":
    main()
