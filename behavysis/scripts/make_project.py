import os

from behavysis.constants import Folders
from behavysis.pydantic_models.configs import get_default_configs
from behavysis.utils.diagnostics_utils import file_exists_msg
from behavysis.utils.template_utils import save_template


def main() -> None:
    """
    Makes a script to run a behavysis analysis project.
    """
    # Very similar structure ot the import_static_templates_script function
    # Updated to include the default_configs.json file and the Folders enum
    description = "Make Behavysis Pipeline Project"
    templates_ls = ["run_pipeline.py"]
    pkg_name = "behavysis"
    pkg_subdir = "templates"
    root_dir = "."
    overwrite = False
    dialogue = True

    if dialogue:
        # Dialogue to check if the user wants to make the files
        to_continue = input(f"Running {description} in current directory. Continue? [y/N]: ").lower() + " "
        if to_continue[0] != "y":
            print("Exiting.")
            return
        # Dialogue to check if the user wants to overwrite the files
        to_overwrite = input("Overwrite existing files? [y/N]: ").lower() + " "
        overwrite = to_overwrite[0] == "y"
    # Making the root folder
    os.makedirs(root_dir, exist_ok=True)
    # Copying the Python files to the project folder
    for template_fp in templates_ls:
        dst_fp = os.path.join(root_dir, template_fp)
        if not overwrite and os.path.exists(dst_fp):
            # Check if we should skip importing (i.e. overwrite is False and file exists)
            print(file_exists_msg(dst_fp))
            continue
        save_template(template_fp, pkg_name, pkg_subdir, dst_fp)
    # Adding the default configs
    if not overwrite and os.path.exists("default_configs.json"):
        print(file_exists_msg("default_configs.json"))
    else:
        default_configs = get_default_configs()
        default_configs.write_json("default_configs.json")
    # Adding the folders
    for folder in Folders:
        os.makedirs(os.path.join(root_dir, folder.value), exist_ok=True)


if __name__ == "__main__":
    main()
