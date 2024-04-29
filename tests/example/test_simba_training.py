import os

from behavysis_pipeline import Experiment
from behavysis_pipeline.behav_classifier import BehavClassifier
from behavysis_pipeline.processes import CalculateParams, FormatVid, Preprocess

if __name__ == "__main__":
    overwrite = True

    proj_dir = r"."
    proj = BehavysisProject(proj_dir)
    proj.importExperiments()
    # exp = proj.getExperiments()[1]

    proj.updateConfigFile(
        os.path.join(proj_dir, "default.json"),
        overwrite="set",
    )
    proj.formatVid(
        funcs=(
            FormatVid.formatVid,
            FormatVid.getVidMetadata,
        ),
        overwrite=overwrite,
    )
    proj.runDLC(
        gputouse=None,
        overwrite=True,
    )
    proj.calculateParams(
        (
            CalculateParams.start_frame,
            CalculateParams.stop_frame,
            CalculateParams.px_per_mm,
        )
    )
    proj.preprocess(
        (
            Preprocess.start_stop_trim,
            Preprocess.interpolate_points,
            Preprocess.bodycentre,
            Preprocess.refine_identities,
        ),
        overwrite=overwrite,
    )
    proj.extractFeatures(True, True)
    # ALL saved in `model` folder.
    # Steps:
    # * Data prep
    #     * Make a configs_json, which holds train/test experiment split
    #     * Prepare attributes df in single large DF (extra index midx level "experiment")
    #     * Prepare behaviour labels df (each column has a separate classifier though) in single large DF (extra index midx level "experiment", and extra column midx level "outcome" ("actual"))
    #     * Ensure that both X_all and y_all have the same rows (midx is ("experiment", "Frame"))
    #     * DO THE SAME FOR X_train and X_test SETS. Split videos into subset for each.
    # * Make individual folders for each behaviour we want to train a classifier for (in `behavs_ls`):
    #     * Copy X dfs to each folder
    #     * Save relevant columns of y dfs to each folder
    #     * Copy configs json
    # * For each behav folder, X_all preprocessing
    #     * Random undersampler (select subset of majority class): this seems to work best. Alternatives are random oversampling (repeat minority class instances).
    #         * Undersample X_all and X_train. DO NOT do for X_test.
    # * Define Classifier
    #     * GradientBoost(): This seems to work best. Alternatives are RF, XGBoost, and Keras MLP.
    # * Run Classifier
    #     * From saved classifier hyper-params.
    # * Evaluate
    #     * Using novel videos (from where though?? Maybe save some videos for a X_train and X_test dataset) to create:
    #         * Sorted probability results logistic graph (line for probabilities, points for actuals, vline for threshold).
    #         * Accuracy, Precision, Recall, F1 graph for range of threshold (from 0 to 1).
    #         * Timeseries probabilities against actuals lineplot for each video.
    #         * Annotated video with predicted vs actual behavs.

    # Making root classifier folder (stores all classifiers and data)
    # root_clf = BehavClassifier.from_BehavysisProject(proj)
    # # Combining dfs into x_all and y_all
    # root_clf.combine_dfs(
    #     os.path.join(proj.dir, "5_features_extracted"),
    #     os.path.join(proj.dir, "7_scored_behavs"),
    # )
    # # Making train/test split
    # root_clf.make_train_test_split()
    # # Making individual behav classifiers
    # behav_ls = ["fight", "marked_fight", "unmarked_fight"]
    # behav_clf_ls = [root_clf.make_behav_model_subdir(behav) for behav in behav_ls]
    # for behav_clf in behav_clf_ls:
    #     # Undersampling majority class (helps a LOT with classifying positive behaviour)
    #     behav_clf.make_random_undersample()
    #     # Initialising and preparing classifiers
    #     behav_clf.init_behav_classifier()
    #     # Training classifiers
    #     behav_clf.train_behav_classifier()
    #     # Evaluating classifiers
    #     behav_clf.model_eval()

    # Evaluate in behavysis_viewer

    # import shutil
    # import os

    # for i in [
    #     "0_configs",
    #     "2_formatted_vid",
    #     "3_dlc_csv",
    #     "4_preprocessed_csv",
    #     "5_features_extracted",
    #     "6_predicted_behavs",
    #     "analysis",
    #     "diagnostics",
    #     "evaluate",
    # ]:
    #     if os.path.exists(os.path.join(proj_dir, i)):
    #         shutil.rmtree(os.path.join(proj_dir, i))
