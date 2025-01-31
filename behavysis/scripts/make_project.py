from behavysis.pydantic_models.experiment_configs import get_default_configs
from behavysis.utils.template_utils import import_static_templates_script


def main() -> None:
    """
    Makes a script to run a behavysis analysis project.
    """
    import_static_templates_script(
        description="Make Behavysis Pipeline Project",
        templates_ls=["run_pipeline.py"],
        pkg_name="behavysis",
        pkg_subdir="templates",
        root_dir=".",
        overwrite=False,
        dialogue=True,
    )
    default_configs = get_default_configs()
    default_configs.write_json("default_configs.json")


if __name__ == "__main__":
    main()
