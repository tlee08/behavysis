import os

from behavysis_pipeline.utils.diagnostics_utils import file_exists_msg
from behavysis_pipeline.utils.io_utils import save_template


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
        if not overwrite and os.path.exists(dst_fp):
            print(file_exists_msg(dst_fp))
            continue
        # Saving the template to the file
        save_template(
            i,
            "behavysis_pipeline",
            "templates",
            dst_fp,
        )


if __name__ == "__main__":
    main()
