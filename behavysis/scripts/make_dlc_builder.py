from behavysis.utils.template_utils import import_static_templates_script


def main() -> None:
    """
    Makes a script to build a DeepLabCut model.
    """
    import_static_templates_script(
        description="Make DLC Model Script",
        templates_ls=["dlc_builder.ipynb"],
        pkg_name="behavysis",
        pkg_subdir="templates",
        root_dir=".",
        to_overwrite=False,
        dialogue=True,
    )


if __name__ == "__main__":
    main()
