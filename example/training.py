import os

from behavysis_core.mixins.behav_mixin import BehavMixin

from behavysis_pipeline.behav_classifier import BehavClassifier
from behavysis_pipeline.behav_classifier.clf_templates import DNN1
from behavysis_pipeline.pipeline import Project

if __name__ == "__main__":
    root_dir = "."
    overwrite = True

    # Choose GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    # Option 1: From BORIS
    # Define behaviours in BORIS
    behavs_ls = ["potential huddling", "huddling"]
    # Paths
    configs_dir = os.path.join(root_dir, "0_configs")
    boris_dir = os.path.join(root_dir, "boris")
    out_dir = os.path.join(root_dir, "7_scored_behavs")
    # Getting names of all files
    names = [os.path.splitext(i)[0] for i in os.listdir(boris_dir)]
    for name in names:
        # Paths
        boris_fp = os.path.join(boris_dir, f"{name}.tsv")
        configs_fp = os.path.join(configs_dir, f"{name}.json")
        out_fp = os.path.join(out_dir, f"{name}.feather")
        # Making df from BORIS
        df = BehavMixin.import_boris_tsv(boris_fp, configs_fp, behavs_ls)
        # Saving df
        df.to_feather(out_fp)
    # Making BehavClassifier objects
    for behav in behavs_ls:
        BehavClassifier.create_new_model(os.path.join(root_dir, "behav_models"), behav)

    # Option 2: From previous behavysis project
    proj = Project(root_dir)
    proj.import_experiments()
    # Making BehavClassifier objects
    BehavClassifier.create_from_project(proj)

    # Loading a BehavModel
    behav = "fight"
    model = BehavClassifier.load(
        os.path.join(root_dir, "behav_models", f"{behav}.json")
    )
    # Testing all different classifiers
    model.clf_eval_compare_all()
    # MANUALLY LOOK AT THE BEST CLASSIFIER AND SELECT
    # Example
    model.pipeline_build(DNN1)

    # Example of using model for inference
    # Loading a BehavModel
    model = BehavClassifier.load(
        os.path.join(root_dir, "behav_models", f"{behav}.json")
    )
    # Getting data
    x, y = model.prepare_data_training()
    # Splitting into train and test sets
    x_train, x_test, y_train, y_test = model.train_test_split(x, y)
    # Loading classifier
    model.clf_load()
    # Evaluating classifier (results stored in "eval" folder)
    model.clf_eval(x_test, y_test)
