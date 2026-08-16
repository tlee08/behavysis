"""CNN experiment: train a 1D temporal CNN and compare to tabular baselines.

Trains the ``CNN`` architecture via ``TorchAdapter`` on the held-out-video
split, then reports frame-level and bout-level PR-AUC on the test set
(against the XGB/TabPFN baseline of ~0.80 frame / ~0.84 bout).

Follows the same train/val/downsample/optimise contract as ``train_model``.
"""

from __future__ import annotations

import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from behavysis.behaviour_classifier.adapter import TorchAdapter
from behavysis.behaviour_classifier.config import ModelRecipe
from behavysis.behaviour_classifier.data import (
    ACTUAL,
    agg_eval_df_by_bouts,
    df_stride_sample,
    df_under_sample_by_group,
    stratified_split_by_group,
)
from behavysis.behaviour_classifier.torch.architectures import CNN
from behavysis.constants import BOUT_ID, EXPERIMENT
from behavysis.funcs.extract_features.extract_rearing import REARING_FEATURES

from .common import EDA_OUT_DIR, load_features_labels, load_model_eval, write_json

BEHAVIOUR = "FR"
_META = [EXPERIMENT, "frame", ACTUAL, BOUT_ID]


def main() -> None:
    """Train the CNN and report test metrics."""
    df = load_features_labels()
    test_exps = set(load_model_eval("xgb", "test")[EXPERIMENT].unique().to_list())
    # Use only the primitive features: the CNN learns temporal context itself,
    # replacing the hand-crafted rolling aggregates the tabular models need.
    trainval = df.filter(~pl.col(EXPERIMENT).is_in(test_exps)).select(
        _META + REARING_FEATURES
    )
    test = df.filter(pl.col(EXPERIMENT).is_in(test_exps)).select(
        _META + REARING_FEATURES
    )

    recipe_fp = EDA_OUT_DIR / "cnn_recipe.yaml"
    recipe = ModelRecipe(
        behaviour_name=BEHAVIOUR,
        model_name="cnn",
        model_type="torch",
        stride_frames=1,
        under_sampling_strategy=1.0,
    )
    recipe.write_yaml(recipe_fp)

    # Split val from the train experiments, then downsample the train rows.
    train_idx, val_idx = stratified_split_by_group(
        trainval, recipe.val_split, EXPERIMENT, recipe.seed
    )
    idx_df = (
        trainval.gather(train_idx).select(EXPERIMENT, ACTUAL, BOUT_ID).with_row_index()
    )
    idx_df = df_stride_sample(idx_df, recipe.stride_frames)
    if recipe.under_sampling_strategy is not None:
        idx_df = df_under_sample_by_group(
            idx_df, recipe.under_sampling_strategy, seed=recipe.seed
        )
    train_downsample_idx = train_idx[idx_df.get_column("index").to_numpy()]

    adapter = TorchAdapter(
        recipe_fp,
        model_cls=CNN,
        window_frames=25,
        batch_size=512,
        epochs=10,
    )
    history = adapter.fit(trainval.gather(train_downsample_idx))
    print("training loss per epoch:")  # noqa: T201
    print(history.to_string())  # noqa: T201

    adapter.optimise_postprocessing_parameters(trainval.gather(val_idx))

    y_test = adapter.predict(test).with_columns(
        test.get_column(ACTUAL),
        test.get_column(BOUT_ID),
    )
    bouts = agg_eval_df_by_bouts(y_test)

    report = {
        "frame_pr_auc": average_precision_score(y_test[ACTUAL], y_test["prob"]),
        "frame_roc_auc": roc_auc_score(y_test[ACTUAL], y_test["prob"]),
        "bout_pr_auc": average_precision_score(bouts[ACTUAL], bouts["prob"]),
        "bout_roc_auc": roc_auc_score(bouts[ACTUAL], bouts["prob"]),
        "bout_detection_rate_any": float(
            bouts.filter(pl.col(ACTUAL) == 1)["pred"].mean()
        ),
    }
    write_json(report, EDA_OUT_DIR / "cnn_experiment.json")
    for k, v in report.items():
        print(f"{k}: {v:.3f}")  # noqa: T201


if __name__ == "__main__":
    main()
