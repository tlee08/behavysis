from behavysis_pipeline.utils.template_utils import import_static_templates_script


def main(root_dir: str = ".", overwrite: bool = False) -> None:
    """
    Makes a script to build a DeepLabCut model.
    """
    import_static_templates_script(
        description="Make DLC Model Script",
        templates_ls=["train_dlc_model.py"],
        pkg_name="behavysis_pipeline",
        pkg_subdir="templates",
        root_dir=".",
        overwrite=False,
        dialogue=True,
    )


if __name__ == "__main__":
    main()
