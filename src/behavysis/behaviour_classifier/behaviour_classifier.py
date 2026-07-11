"""Behaviour classifier — training, versioning, promotion, and inference.

A classifier is fully self-contained in its own directory (``clf_dir``). The
directory name is arbitrary; the behaviour it classifies is authored in each
model_type's ``config.yaml`` (``TrainingRecipe.behaviour_name``) and is the
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
from sklearn.model_selection import train_test_split

from behavysis.constants import (
    BEHAVIOUR,
    FRAME,
    PRED,
    PROB,
)
from behavysis.schemas import BEHAVIOUR_PREDICTED_SCHEMA

from .adapter import BaseAdapter, SklearnAdapter, TorchAdapter
from .config import (
    ActivePointer,
    DatasetManifest,
    DataSummary,
    EvalSummary,
    Leaderboard,
    LeaderboardEntry,
    ProductionPointer,
    ResolvedHyperparams,
    TrainingRecipe,
    TrainingSummary,
    VersionMetadata,
)
from .data import (
    align_features_labels,
    load_feature_names,
    load_features,
    load_labels,
    stratified_split_by_video,
)
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


# ── training ─────────────────────────────────────────────────────────


def train(clf_dir: Path, model_type: str) -> str:
    """Train a classifier and persist all versioned artifacts.

    ``clf_dir`` is the self-contained classifier directory (its name is
    arbitrary); training data is read from ``clf_dir/training_data/``. A
    human-authored ``config.yaml`` (a ``TrainingRecipe`` declaring
    ``behaviour_name`` and the ``individuals``/``bodyparts`` feature contract)
    must already exist for ``model_type`` — configs are never auto-created.

    Auto-promotes within model_type if the new version improves
    ``test_f1_behav`` over the currently active version.

    Returns the version string.
    """
    clf_dir = clf_dir.resolve()

    # 1. Load the authored recipe (config.yaml is the single source of truth)
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
        config.behaviour_name,
        model_type,
    )

    # 2. Load and align data
    x_ls, x_names = load_features(features_dir(clf_dir))
    y_ls, y_names = load_labels(labels_dir(clf_dir), config.behaviour_name)
    x_ls, y_ls, exp_names = align_features_labels(
        x_ls,
        y_ls,
        x_names,
        y_names,
    )

    # 3. Three-way split: test (grouped), then val from remaining
    train_idx, val_idx, test_idx = _three_way_split(
        x_ls,
        y_ls,
        config.test_split,
        config.val_split,
        config.seed,
    )

    # 4. Train
    start = datetime.now(UTC)
    factory = MODEL_REGISTRY[model_type]
    adapter: BaseAdapter = factory()
    history = adapter.fit(x_ls, y_ls, train_idx, config)
    duration = (datetime.now(UTC) - start).total_seconds()

    # 5. Generate version & create output dir
    version = _next_version(clf_dir, model_type)
    ed = eval_dir(clf_dir, model_type, version)
    ed.mkdir(parents=True, exist_ok=True)

    # 6. Save model
    adapter.save(model_fp(clf_dir, model_type, version))

    # 7. Evaluate all splits
    train_acc, train_f1 = _eval_split(
        adapter,
        x_ls,
        y_ls,
        train_idx,
        config,
        ed,
        "train",
    )
    val_acc, val_f1 = _eval_split(
        adapter,
        x_ls,
        y_ls,
        val_idx,
        config,
        ed,
        "val",
    )
    test_acc, test_f1 = _eval_split(
        adapter,
        x_ls,
        y_ls,
        test_idx,
        config,
        ed,
        "test",
    )

    # 8. Training history plot
    if not history.empty:
        save_training_history(history, ed)

    # 9. Diagnostics
    _run_diagnostics(adapter, clf_dir, ed)

    # 10. Metadata
    data_summary = _make_data_summary(
        x_ls,
        y_ls,
        train_idx,
        val_idx,
        test_idx,
        n_features_selected=len(adapter.feature_mask),
    )
    meta = VersionMetadata(
        version=version,
        framework=adapter.framework,
        model_type=model_type,
        created_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        resolved=ResolvedHyperparams(
            seed=config.seed,
            batch_size=config.batch_size,
            epochs=config.epochs,
            oversample_ratio=config.oversample_ratio,
            undersample_ratio=config.undersample_ratio,
            test_split=config.test_split,
            val_split=config.val_split,
        ),
        data=data_summary,
        training=TrainingSummary(duration_seconds=round(duration, 1)),
        evaluation=EvalSummary(
            train_accuracy=round(train_acc, 4) if train_acc is not None else None,
            train_f1_behav=round(train_f1, 4) if train_f1 is not None else None,
            val_accuracy=round(val_acc, 4) if val_acc is not None else None,
            val_f1_behav=round(val_f1, 4) if val_f1 is not None else None,
            test_accuracy=round(test_acc, 4) if test_acc is not None else None,
            test_f1_behav=round(test_f1, 4) if test_f1 is not None else None,
        ),
    )
    meta.write_yaml(metadata_fp(clf_dir, model_type, version))

    # 11. Dataset manifest
    train_ids = [exp_names[i] for i in range(len(exp_names)) if len(train_idx[i]) > 0]
    val_ids = [exp_names[i] for i in range(len(exp_names)) if len(val_idx[i]) > 0]
    test_ids = [exp_names[i] for i in range(len(exp_names)) if len(test_idx[i]) > 0]
    manifest = DatasetManifest(
        version=version,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        n_train=sum(len(idx) for idx in train_idx),
        n_val=sum(len(idx) for idx in val_idx),
        n_test=sum(len(idx) for idx in test_idx),
    )
    manifest.write_yaml(dataset_manifest_fp(clf_dir, model_type, version))

    # 12. Auto-promote if better
    _auto_promote(clf_dir, model_type, version, test_f1)

    logger.info(
        "Training complete: {} {} (test_f1={:.4f})",
        model_type,
        version,
        test_f1,
    )
    return version


def train_all_models(
    clf_dir: Path,
    behaviour_name: str,
    individuals: list[str],
    bodyparts: list[str],
) -> list[str]:
    """Author a default recipe per model_type (if missing) and train them all.

    ``behaviour_name`` and the feature contract (``individuals``/``bodyparts``)
    are written into each freshly created ``config.yaml``; existing configs are
    left untouched. Hyperparameters use ``TrainingRecipe`` defaults — edit the
    written ``config.yaml`` and re-run ``train`` for finer control.
    """
    clf_dir = clf_dir.resolve()
    results: list[str] = []
    for mt in MODEL_REGISTRY:
        cfp = config_fp(clf_dir, mt)
        if not cfp.exists():
            TrainingRecipe(
                model_type=mt,
                behaviour_name=behaviour_name,
                individuals=individuals,
                bodyparts=bodyparts,
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


def _auto_promote(
    clf_dir: Path,
    model_type: str,
    new_version: str,
    new_test_f1: float | None,
) -> None:
    """Promote new version if it beats the current active."""
    if new_test_f1 is None:
        return

    afp = active_fp(clf_dir, model_type)
    if not afp.exists():
        promote(clf_dir, model_type, new_version)
        return

    active = ActivePointer.read_yaml(afp)
    mfp = metadata_fp(clf_dir, model_type, active.version)
    if not mfp.exists():
        promote(clf_dir, model_type, new_version)
        return

    cur_meta = VersionMetadata.read_yaml(mfp)
    cur_f1 = cur_meta.evaluation.test_f1_behav

    if cur_f1 is None or new_test_f1 > cur_f1:
        promote(clf_dir, model_type, new_version)


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
    """Return the behaviour name from the first available model_type config."""
    for mt in _model_types(clf_dir):
        cfp = config_fp(clf_dir, mt)
        if cfp.exists():
            return TrainingRecipe.read_yaml(cfp).behaviour_name
    return ""


# ── production ───────────────────────────────────────────────────────


def promote_to_production(clf_dir: Path, model_type: str, version: str) -> None:
    """Set the production pointer, copying the model's contract from its config.

    ``behaviour_name`` and the feature contract (``individuals``/``bodyparts``)
    are read from the model_type's ``config.yaml`` and recorded in
    ``production.yaml`` so downstream inference resolves them from one place.
    """
    clf_dir = clf_dir.resolve()
    vd = model_fp(clf_dir, model_type, version)
    if not vd.exists():
        msg = f"Version {version} not found for {model_type} in {clf_dir.name}"
        raise FileNotFoundError(msg)

    cfp = config_fp(clf_dir, model_type)
    if not cfp.exists():
        msg = f"Missing config.yaml for {model_type} in {clf_dir}"
        raise FileNotFoundError(msg)
    config = TrainingRecipe.read_yaml(cfp)

    ptr = ProductionPointer(
        behaviour_name=config.behaviour_name,
        model_type=model_type,
        version=version,
        individuals=config.individuals,
        bodyparts=config.bodyparts,
        promoted_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    ptr.write_yaml(production_fp(clf_dir))
    logger.info(
        "Promoted {} to production: {}/{}",
        config.behaviour_name,
        model_type,
        version,
    )


# ── inference ────────────────────────────────────────────────────────


class BehaviourClassifier:
    """Thin wrapper around a trained adapter for inference.

    Obtained via ``BehaviourClassifier.load()``, not constructed directly.
    """

    def __init__(
        self,
        config: TrainingRecipe,
        adapter: BaseAdapter,
    ) -> None:
        self._config = config
        self._adapter = adapter

    @property
    def config(self) -> TrainingRecipe:
        return self._config

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
            msg = (
                f"Missing config.yaml for {model_type} in {clf_dir}; "
                "cannot resolve the behaviour name."
            )
            raise FileNotFoundError(msg)
        config = TrainingRecipe.read_yaml(cfp)

        adapter = cls._load_adapter(clf_dir, model_type, version)
        return cls(config, adapter)

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
        return cls._load_version(clf_dir, prod.model_type, prod.version)

    @staticmethod
    def _load_adapter(
        clf_dir: Path,
        model_type: str,
        version: str,
    ) -> BaseAdapter:
        vd = model_fp(clf_dir, model_type, version)
        factory = MODEL_REGISTRY[model_type]
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
        y_prob = self._adapter.predict(
            x,
            np.arange(x.shape[0]),
            self._config.batch_size,
        )
        y_pred = (y_prob > self._config.pcutoff).astype(int)
        return pl.DataFrame(
            {
                FRAME: frames,
                BEHAVIOUR: [self._config.behaviour_name] * len(frames),
                PROB: y_prob,
                PRED: y_pred,
            },
            schema=BEHAVIOUR_PREDICTED_SCHEMA,
        )


# ── internal helpers ─────────────────────────────────────────────────


def _eval_split(
    adapter: BaseAdapter,
    x_ls: list[np.ndarray],
    y_ls: list[np.ndarray],
    index_ls: list[np.ndarray],
    config: TrainingRecipe,
    eval_d: Path,
    name: str,
) -> tuple[float | None, float | None]:
    """Evaluate on a data split and save artifacts."""
    y_true_ls = [y[idx] for y, idx in zip(y_ls, index_ls, strict=True)]
    y_prob_ls = [
        adapter.predict(x, idx, config.batch_size)
        for x, idx in zip(x_ls, index_ls, strict=True)
    ]

    y_true = np.concatenate(y_true_ls)
    y_prob = np.concatenate(y_prob_ls)
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
        config.behaviour_name,
        config.pcutoff,
        eval_d,
        name,
        index_ls,
    )

    return accuracy, f1_behav


def _make_data_summary(
    x_ls: list[np.ndarray],
    y_ls: list[np.ndarray],
    train_idx: list[np.ndarray],
    val_idx: list[np.ndarray],
    test_idx: list[np.ndarray],
    n_features_selected: int,
) -> DataSummary:
    y_train = np.concatenate([y[idx] for y, idx in zip(y_ls, train_idx, strict=True)])
    y_test = np.concatenate([y[idx] for y, idx in zip(y_ls, test_idx, strict=True)])
    return DataSummary(
        n_samples=sum(x.shape[0] for x in x_ls),
        n_features=x_ls[0].shape[1] if x_ls else 0,
        n_features_selected=n_features_selected,
        n_train=sum(len(idx) for idx in train_idx),
        n_val=sum(len(idx) for idx in val_idx),
        n_test=sum(len(idx) for idx in test_idx),
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


def _three_way_split(
    x_ls: list[np.ndarray],
    y_ls: list[np.ndarray],
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Split into train/val/test, respecting video groups for test."""
    train_val_idx, test_idx = stratified_split_by_video(
        x_ls,
        y_ls,
        test_size,
        seed,
    )

    x_train_val = np.concatenate(
        [x[idx] for x, idx in zip(x_ls, train_val_idx, strict=True)]
    )
    y_train_val = np.concatenate(
        [y[idx] for y, idx in zip(y_ls, train_val_idx, strict=True)]
    )

    val_ratio = val_size / (1 - test_size)
    tv_flat, val_flat = train_test_split(
        np.arange(x_train_val.shape[0]),
        stratify=y_train_val,
        test_size=val_ratio,
        random_state=seed,
    )

    t_offsets = np.cumsum([0] + [len(idx) for idx in train_val_idx[:-1]])
    train_idx: list[np.ndarray] = []
    val_idx: list[np.ndarray] = []
    for i in range(len(train_val_idx)):
        lo = t_offsets[i]
        hi = t_offsets[i] + len(train_val_idx[i])

        tr = tv_flat[(tv_flat >= lo) & (tv_flat < hi)] - lo
        train_idx.append(train_val_idx[i][tr])

        vr = val_flat[(val_flat >= lo) & (val_flat < hi)] - lo
        val_idx.append(train_val_idx[i][vr])

    return train_idx, val_idx, test_idx
