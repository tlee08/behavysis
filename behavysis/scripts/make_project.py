from behavysis.utils.template_utils import import_static_templates_script


def main() -> None:
    """
    Makes a script to run a behavysis analysis project.
    """
    import_static_templates_script(
        description="Make Behavysis Pipeline Project",
        templates_ls=["run_pipeline.py", "default_configs.json"],
        pkg_name="behavysis_pipeline",
        pkg_subdir="templates",
        root_dir=".",
        overwrite=False,
        dialogue=True,
    )


if __name__ == "__main__":
    main()
