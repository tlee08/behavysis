"""Behaviour classifier — training, versioning, promotion, and inference.

A classifier is fully self-contained in its own directory (``clf_dir``). The
directory name is arbitrary; the behaviour it classifies is authored in the
shared ``contract.yaml`` (``ClassifierContract.behaviour_name``) and is the
single source of truth. Training data lives inside the classifier at
``{clf_dir}/training_data/`` and mirrors the inference pipeline's stage folders
(``5_features_extracted/``, ``7_behaviour_scored/``, …). See ``storage`` for the
full on-disk layout.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import joblib
import numpy as np
import polars as pl
from loguru import logger
from sklearn.metrics import classification_report

from behavysis.constants import (
    ACTUAL,
    BEHAVIOUR,
    EXPERIMENT,
    FRAME,
    PRED,
    PROB,
)
from behavysis.schemas import BEHAVIOUR_PREDICTED_SCHEMA

from .adapter import BaseAdapter, SklearnAdapter, TorchAdapter
from .config import (
    ActivePointer,
    ClassifierContract,
    DatasetManifest,
    DataSummary,
    EvalSummary,
    Leaderboard,
    LeaderboardEntry,
    ProductionPointer,
    TrainingRecipe,
    TrainingSummary,
    VersionMetadata,
)
from .data import load_feature_names, load_training_data, stratified_split_by_video
from .evaluation import (
    save_evaluation_results,
    save_feature_importance,
    save_feature_report,
    save_training_history,
)
from .registry import MODEL_REGISTRY
from .storage import (
    active_fp,
    config_fp,
    contract_fp,
    dataset_manifest_fp,
    eval_dir,
    features_dir,
    labels_dir,
    leaderboard_fp,
    metadata_fp,
    model_fp,
    production_fp,
    versions_dir,
)

if TYPE_CHECKING:
    from pathlib import Path

# ── version string generation ────────────────────────────────────────

_VERSION_RE = re.compile(r"^v(\d+)_")


def _next_version(clf_dir: Path, model_type: str) -> str:
    """Generate the next version string."""
    vd = versions_dir(clf_dir, model_type)
    seq = 1
    if vd.exists():
        nums = [
            int(m.group(1))
            for d in vd.iterdir()
            if d.is_dir() and (m := _VERSION_RE.match(d.name))
        ]
        seq = max(nums) + 1 if nums else 1
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%S")
    return f"v{seq:03d}_{ts}"


# ── initialise ───────────────────────────────────────────────────────


def set_contract(
    clf_dir: Path,
    behaviour_name: str,
    individuals: list[str],
    bodyparts: list[str],
) -> None:
    """Set model's contract.yaml."""
    cofp = contract_fp(clf_dir)
    if not cofp.exists():
        ClassifierContract(
            behaviour_name=behaviour_name,
            individuals=individuals,
            bodyparts=bodyparts,
        ).write_yaml(cofp)


# ── training ─────────────────────────────────────────────────────────


def train(clf_dir: Path, model_type: str) -> str:
    """Train a classifier and persist all versioned artifacts.

    ``clf_dir`` is the self-contained classifier directory (its name is
    arbitrary); training data is read from ``clf_dir/training_data/``. The
    shared ``contract.yaml`` (a ``ClassifierContract`` declaring
    ``behaviour_name`` and the ``individuals``/``bodyparts`` feature contract)
    and a human-authored ``config.yaml`` (a ``TrainingRecipe`` of
    hyperparameters) must already exist — neither is auto-created here.

    Auto-promotes within model_type if the new version improves
    ``test_f1_behav`` over the currently active version.

    Returns the version string.
    """
    clf_dir = clf_dir.resolve()

    # 1. Load the shared contract and the authored recipe
    contract = _read_contract(clf_dir)
    cfp = config_fp(clf_dir, model_type)
    if not cfp.exists():
        msg = (
            f"No config.yaml for '{model_type}' in {clf_dir}. "
            "Author a TrainingRecipe first (via train_all_models or "
            "TrainingRecipe(...).write_yaml(config_fp(clf_dir, model_type)))."
        )
        raise FileNotFoundError(msg)
    config = TrainingRecipe.read_yaml(cfp)

    logger.info(
        "Training {} (model_type={})",
        contract.behaviour_name,
        model_type,
    )

    # 2. Load and align data
    df = load_training_data(
        features_dir(clf_dir),
        labels_dir(clf_dir),
        contract.behaviour_name,
    )

    # 3. Two-way split: test (grouped by experiment), rest is train
    train_mask, test_mask = stratified_split_by_video(
        df, config.test_split, config.split_seed
    )

    # 4. Train
    start = datetime.now(UTC)
    factory, _ = MODEL_REGISTRY[model_type]
    adapter: BaseAdapter = factory()
    history = adapter.fit(df, train_mask, config)
    duration = (datetime.now(UTC) - start).total_seconds()

    # 5. Generate version & create output dir
    version = _next_version(clf_dir, model_type)
    ed = eval_dir(clf_dir, model_type, version)
    ed.mkdir(parents=True, exist_ok=True)

    # 6. Save model
    adapter.save(model_fp(clf_dir, model_type, version))

    # 7. Evaluate both splits
    train_acc, train_f1 = _eval_split(
        adapter,
        df,
        train_mask,
        config,
        contract.behaviour_name,
        ed,
        "train",
    )
    test_acc, test_f1 = _eval_split(
        adapter,
        df,
        test_mask,
        config,
        contract.behaviour_name,
        ed,
        "test",
    )

    # 8. Training history plot
    if not history.empty:
        save_training_history(history, ed)

    # 9. Diagnostics
    _run_diagnostics(adapter, clf_dir, ed)

    # 10. Metadata (resolve hyperparameters from grid search if used)
    if isinstance(adapter, SklearnAdapter):
        resolved_recipe = config.model_copy(
            update={"hyperparameters": adapter.resolved_hyperparameters}
        )
    else:
        resolved_recipe = config
    data_summary = _make_data_summary(
        df,
        train_mask,
        test_mask,
        n_features_selected=adapter.feature_mask.shape[0],
    )
    meta = VersionMetadata(
        version=version,
        framework=adapter.framework,
        model_type=model_type,
        created_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        recipe=resolved_recipe,
        data=data_summary,
        training=TrainingSummary(duration_seconds=round(duration, 1)),
        evaluation=EvalSummary(
            train_accuracy=round(train_acc, 4) if train_acc is not None else None,
            train_f1_behav=round(train_f1, 4) if train_f1 is not None else None,
            test_accuracy=round(test_acc, 4) if test_acc is not None else None,
            test_f1_behav=round(test_f1, 4) if test_f1 is not None else None,
        ),
    )
    meta.write_yaml(metadata_fp(clf_dir, model_type, version))

    # 11. Dataset manifest
    train_ids = sorted(df.filter(train_mask)[EXPERIMENT].unique().to_list())
    test_ids = sorted(df.filter(test_mask)[EXPERIMENT].unique().to_list())
    manifest = DatasetManifest(
        version=version,
        train_ids=train_ids,
        test_ids=test_ids,
        n_train=int(train_mask.sum()),
        n_test=int(test_mask.sum()),
    )
    manifest.write_yaml(dataset_manifest_fp(clf_dir, model_type, version))

    # 12. Promote to best
    promote_to_best(clf_dir, model_type)

    logger.info(
        "Training complete: {} {} (test_f1={:.4f})",
        model_type,
        version,
        test_f1,
    )
    return version


def train_all_models(clf_dir: Path) -> list[str]:
    """Author the shared contract + a default recipe per model_type, then train.

    ``behaviour_name`` and the feature contract (``individuals``/``bodyparts``)
    are written once to ``contract.yaml`` (if missing). A default
    ``config.yaml`` of hyperparameters is written per model_type (if missing);
    existing files are left untouched. Edit them and re-run ``train`` for finer
    control.
    """
    results: list[str] = []
    for mt in MODEL_REGISTRY:
        cfp = config_fp(clf_dir, mt)
        if not cfp.exists():
            _, defaults = MODEL_REGISTRY[mt]
            TrainingRecipe(
                model_type=mt,
                hyperparameters=dict(defaults),
            ).write_yaml(cfp)
        results.append(train(clf_dir, mt))

    _ = regenerate_leaderboard(clf_dir)
    return results


# ── promotion ────────────────────────────────────────────────────────


def promote(clf_dir: Path, model_type: str, version: str) -> None:
    """Set the active version for a model_type."""
    vd = model_fp(clf_dir, model_type, version)
    if not vd.exists():
        msg = f"Version {version} not found for {model_type} in {clf_dir.name}"
        raise FileNotFoundError(msg)

    ptr = ActivePointer(
        version=version,
        promoted_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    ptr.write_yaml(active_fp(clf_dir, model_type))
    logger.info(
        "Promoted {}/{} to {}",
        clf_dir.name,
        model_type,
        version,
    )


def promote_to_best(clf_dir: Path, model_type: str | None = None) -> None:
    """Scan versions and promote the best by test_f1_behav.

    If model_type is None, does this for every model_type.
    """
    types = [model_type] if model_type else list(_model_types(clf_dir))

    for mt in types:
        best = _find_best_version(clf_dir, mt)
        if best is None:
            logger.warning("No versions found for {}/{}", clf_dir.name, mt)
            continue
        promote(clf_dir, mt, best)


def _find_best_version(clf_dir: Path, model_type: str) -> str | None:
    """Return the version with highest test_f1_behav for a model_type."""
    vd = versions_dir(clf_dir, model_type)
    if not vd.exists():
        return None

    best_f1: float = -1.0
    best_ver: str | None = None

    for d in sorted(vd.iterdir()):
        if not d.is_dir():
            continue
        mfp = metadata_fp(clf_dir, model_type, d.name)
        if not mfp.exists():
            continue
        meta = VersionMetadata.read_yaml(mfp)
        f1 = meta.evaluation.test_f1_behav
        if f1 is not None and f1 > best_f1:
            best_f1 = f1
            best_ver = d.name

    return best_ver


# ── leaderboard ──────────────────────────────────────────────────────


def regenerate_leaderboard(clf_dir: Path) -> Leaderboard:
    """Rebuild leaderboard.yaml from all model_types' active versions."""
    clf_dir = clf_dir.resolve()
    rankings: list[LeaderboardEntry] = []

    for mt in _model_types(clf_dir):
        afp = active_fp(clf_dir, mt)
        if not afp.exists():
            continue

        active = ActivePointer.read_yaml(afp)
        mfp = metadata_fp(clf_dir, mt, active.version)
        if not mfp.exists():
            continue

        meta = VersionMetadata.read_yaml(mfp)
        test_f1 = meta.evaluation.test_f1_behav
        train_f1 = meta.evaluation.train_f1_behav
        test_acc = meta.evaluation.test_accuracy

        overfit = None
        if train_f1 is not None and test_f1 is not None and train_f1 > 0:
            overfit = round(train_f1 - test_f1, 4)

        rankings.append(
            LeaderboardEntry(
                model_type=mt,
                version=active.version,
                test_f1_behav=round(test_f1, 4) if test_f1 is not None else None,
                test_accuracy=round(test_acc, 4) if test_acc is not None else None,
                train_f1_behav=round(train_f1, 4) if train_f1 is not None else None,
                overfit_ratio=overfit,
            )
        )

    rankings.sort(
        key=lambda e: e.test_f1_behav if e.test_f1_behav is not None else -1.0,
        reverse=True,
    )

    board = Leaderboard(
        behaviour_name=_behaviour_name(clf_dir),
        generated_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        rankings=rankings,
    )
    board.write_yaml(leaderboard_fp(clf_dir))
    logger.info("Regenerated leaderboard for {}", clf_dir.name)
    return board


def _behaviour_name(clf_dir: Path) -> str:
    """Return the behaviour name from the shared contract, or empty if absent."""
    cofp = contract_fp(clf_dir)
    if cofp.exists():
        return ClassifierContract.read_yaml(cofp).behaviour_name
    return ""


def _read_contract(clf_dir: Path) -> ClassifierContract:
    """Read the shared classifier contract, raising if absent."""
    cofp = contract_fp(clf_dir)
    if not cofp.exists():
        msg = (
            f"No contract.yaml in {clf_dir}. Author a ClassifierContract first "
            "(via train_all_models or "
            "ClassifierContract(...).write_yaml(contract_fp(clf_dir)))."
        )
        raise FileNotFoundError(msg)
    return ClassifierContract.read_yaml(cofp)


# ── production ───────────────────────────────────────────────────────


def promote_to_production(clf_dir: Path, model_type: str) -> None:
    """Set the production pointer to a ``model_type``.

    Written only here — ``production.yaml`` is a pure pointer to the model_type.
    The deployed version is resolved from that model_type's ``active.yaml``; the
    behaviour and feature contract are resolved from ``contract.yaml``.
    """
    clf_dir = clf_dir.resolve()
    afp = active_fp(clf_dir, model_type)
    if not afp.exists():
        msg = (
            f"No active version for {model_type} in {clf_dir.name}. "
            "Train first or set active.yaml."
        )
        raise FileNotFoundError(msg)

    ptr = ProductionPointer(
        model_type=model_type,
        promoted_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    ptr.write_yaml(production_fp(clf_dir))
    logger.info("Promoted to production: {}", model_type)


# ── inference ────────────────────────────────────────────────────────


class BehaviourClassifier:
    """Thin wrapper around a trained adapter for inference.

    Obtained via ``BehaviourClassifier.load()``, not constructed directly.
    """

    def __init__(
        self,
        config: TrainingRecipe,
        contract: ClassifierContract,
        adapter: BaseAdapter,
    ) -> None:
        self.config = config
        self.contract = contract
        self.adapter = adapter

    @classmethod
    def load(
        cls,
        clf_dir: Path,
        *,
        model_type: str | None = None,
        version: str | None = None,
    ) -> BehaviourClassifier:
        """Load a trained classifier for inference.

        Resolution order:
        1. If model_type AND version given → load that exact version
        2. If only model_type given → read active.yaml for that type
        3. If neither → read production.yaml
        """
        clf_dir = clf_dir.resolve()

        if model_type and version:
            return cls._load_version(clf_dir, model_type, version)

        if model_type:
            return cls._load_active(clf_dir, model_type)

        return cls._load_production(clf_dir)

    @classmethod
    def _load_version(
        cls,
        clf_dir: Path,
        model_type: str,
        version: str,
    ) -> BehaviourClassifier:
        vd = model_fp(clf_dir, model_type, version)
        if not vd.exists():
            msg = f"Version {version} not found for {model_type} in {clf_dir.name}"
            raise FileNotFoundError(msg)

        cfp = config_fp(clf_dir, model_type)
        if not cfp.exists():
            msg = f"Missing config.yaml for {model_type} in {clf_dir}."
            raise FileNotFoundError(msg)
        config = TrainingRecipe.read_yaml(cfp)
        contract = _read_contract(clf_dir)

        adapter = cls._load_adapter(clf_dir, model_type, version)
        return cls(config, contract, adapter)

    @classmethod
    def _load_active(
        cls,
        clf_dir: Path,
        model_type: str,
    ) -> BehaviourClassifier:
        afp = active_fp(clf_dir, model_type)
        if not afp.exists():
            msg = (
                f"No active version for {model_type} in {clf_dir.name}. "
                "Train first or set active.yaml."
            )
            raise FileNotFoundError(msg)

        active = ActivePointer.read_yaml(afp)
        return cls._load_version(clf_dir, model_type, active.version)

    @classmethod
    def _load_production(cls, clf_dir: Path) -> BehaviourClassifier:
        pfp = production_fp(clf_dir)
        if not pfp.exists():
            msg = (
                f"No production model for {clf_dir.name}. "
                "Train and promote_to_production first."
            )
            raise FileNotFoundError(msg)

        prod = ProductionPointer.read_yaml(pfp)
        return cls._load_active(clf_dir, prod.model_type)

    @staticmethod
    def _load_adapter(
        clf_dir: Path,
        model_type: str,
        version: str,
    ) -> BaseAdapter:
        vd = model_fp(clf_dir, model_type, version)
        factory, _ = MODEL_REGISTRY[model_type]
        adapter: BaseAdapter = factory()

        if isinstance(adapter, SklearnAdapter):
            loaded = joblib.load(vd / "model.joblib")
            if isinstance(loaded, BaseAdapter):
                return loaded
            return adapter  # fallback, shouldn't happen

        if isinstance(adapter, TorchAdapter):
            adapter.load_state(vd)
            return adapter

        msg = f"Unknown adapter type: {type(adapter)}"
        raise TypeError(msg)

    def predict(self, features_df: pl.DataFrame) -> pl.DataFrame:
        """Run inference on a wide features DataFrame.

        ``features_df`` is the wide feature table used at training time: a
        ``frame`` column plus one column per feature. Returns a long-form
        DataFrame conforming to ``BEHAVIOUR_PREDICTED_SCHEMA`` — one row per
        frame with ``(frame, behaviour, prob, pred)``.
        """
        frames = features_df.get_column(FRAME)
        x = features_df.drop(FRAME).to_numpy()
        y_prob = self.adapter.predict(
            x,
            self.config.batch_size,
        )
        y_pred = (y_prob > self.config.pcutoff).astype(int)
        return pl.DataFrame(
            {
                FRAME: frames,
                BEHAVIOUR: [self.contract.behaviour_name] * len(frames),
                PROB: y_prob,
                PRED: y_pred,
            },
            schema=BEHAVIOUR_PREDICTED_SCHEMA,
        )


# ── internal helpers ─────────────────────────────────────────────────


def _eval_split(
    adapter: BaseAdapter,
    df: pl.DataFrame,
    mask: np.ndarray,
    config: TrainingRecipe,
    behaviour_name: str,
    eval_d: Path,
    name: str,
) -> tuple[float | None, float | None]:
    """Evaluate on a data split and save artifacts."""
    y_true = df.filter(mask)[ACTUAL].to_numpy()
    x_subset = df.filter(mask).drop([EXPERIMENT, FRAME, ACTUAL]).to_numpy()
    y_prob = adapter.predict(x_subset, config.batch_size)

    y_pred = (y_prob > config.pcutoff).astype(int)

    report = classification_report(
        y_true,
        y_pred,
        target_names=["nil", "behav"],
        output_dict=True,
    )
    accuracy: float | None = report.get("accuracy")
    f1_behav: float | None = report["behav"]["f1-score"]

    _ = save_evaluation_results(
        y_true,
        y_prob,
        y_pred,
        behaviour_name,
        config.pcutoff,
        eval_d,
        name,
        [np.arange(len(y_true))],
    )

    return accuracy, f1_behav


def _make_data_summary(
    df: pl.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    n_features_selected: int,
) -> DataSummary:
    y_train = df.filter(train_mask)[ACTUAL].to_numpy()
    y_test = df.filter(test_mask)[ACTUAL].to_numpy()
    n_features = len(df.columns) - 3  # experiment, frame, actual
    return DataSummary(
        n_samples=len(df),
        n_features=n_features,
        n_features_selected=n_features_selected,
        n_train=int(train_mask.sum()),
        n_test=int(test_mask.sum()),
        train_pos_ratio=round(float(np.mean(y_train)), 4),
        test_pos_ratio=round(float(np.mean(y_test)), 4),
    )


def _model_types(clf_dir: Path) -> list[str]:
    """List model_type directories that exist for this classifier."""
    clf_dir = clf_dir.resolve()
    if not clf_dir.exists():
        return []
    return sorted(
        d.name
        for d in clf_dir.iterdir()
        if d.is_dir()
        and (config_fp(clf_dir, d.name).exists() or d.name in MODEL_REGISTRY)
    )


def _run_diagnostics(
    adapter: BaseAdapter,
    clf_dir: Path,
    eval_d: Path,
) -> None:
    """Run feature importance and SHAP diagnostics."""
    feature_names = load_feature_names(features_dir(clf_dir))
    if not feature_names:
        logger.warning("No feature names found for diagnostics.")
        return

    n_features_total = len(feature_names)

    importances: np.ndarray | None = None
    if isinstance(adapter, SklearnAdapter):
        if adapter.feature_mask is not None:
            feature_names = [feature_names[i] for i in adapter.feature_mask]
        importances = np.zeros(len(feature_names), dtype=np.float64)
        est = adapter.estimator
        if hasattr(est, "feature_importances_"):
            importances = est.feature_importances_
        elif hasattr(est, "coef_"):
            importances = np.abs(est.coef_).flatten()  # type: ignore[assignment]

    if importances is not None:
        save_feature_importance(feature_names, importances, eval_d)
        save_feature_report(feature_names, importances, eval_d, n_features_total)
