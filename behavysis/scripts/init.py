from importlib.util import find_spec


def main() -> None:
    """
    Sets up the behavysis environment.
    - Installs DEEPLABCUT conda env
    - Installs SimBA conda env
    """
    behavysis_dir = find_spec("behavysis").submodule_search_locations[0]
    templates_dir = os.path.join(behavysis_dir, templates)
    # Installing DEEPLABCUT env
    run_subproc_console(f"cd {templates_dir} && conda env create -f DEEPLABCUT.yaml")
    # Installing simba env
    run_subproc_console(f"cd {templates_dir} && conda env create -f simba_env.yaml")
    
    


if __name__ == "__main__":
    main()
