"""Candidate model registry for the Phase 0.5 bake-off.

Every candidate is a scikit-learn ``Pipeline`` wrapping the *shared* preprocessor
(so the tournament is apples-to-apples) plus a classifier. Gradient-boosted
backends (XGBoost, LightGBM, CatBoost) are optional imports — if a library is not
installed, that candidate is silently skipped rather than crashing the run.

Imbalance handling is built into each model (class weights / ``scale_pos_weight``)
so the comparison reflects realistic, tuned-for-imbalance behaviour.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fraud_detection.config import settings
from fraud_detection.logging_utils import get_logger
from fraud_detection.models.preprocessing import build_preprocessor

log = get_logger(__name__)


def _scale_pos_weight(y: np.ndarray) -> float:
    """Ratio of negatives to positives — the standard imbalance weight."""
    y = np.asarray(y).astype(int)
    pos = max(int(y.sum()), 1)
    neg = int((y == 0).sum())
    return neg / pos


def _pipe(classifier, scale_numeric: bool) -> Pipeline:
    return Pipeline(
        [("prep", build_preprocessor(scale_numeric=scale_numeric)), ("clf", classifier)]
    )


# Each builder takes y_train (for imbalance weighting) and returns a Pipeline.
def _build_dummy(_y) -> Pipeline:
    return _pipe(
        DummyClassifier(strategy="stratified", random_state=settings.random_seed),
        scale_numeric=False,
    )


def _build_logistic(_y) -> Pipeline:
    return _pipe(
        LogisticRegression(
            max_iter=2000, class_weight="balanced", C=1.0, random_state=settings.random_seed
        ),
        scale_numeric=True,
    )


def _build_random_forest(_y) -> Pipeline:
    return _pipe(
        RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=settings.random_seed,
        ),
        scale_numeric=False,
    )


def _build_xgboost(y):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None
    return _pipe(
        XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=_scale_pos_weight(y),
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=settings.random_seed,
        ),
        scale_numeric=False,
    )


def _build_lightgbm(y):
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        return None
    return _pipe(
        LGBMClassifier(
            n_estimators=500,
            num_leaves=63,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=_scale_pos_weight(y),
            n_jobs=-1,
            random_state=settings.random_seed,
            verbose=-1,
        ),
        scale_numeric=False,
    )


def _build_catboost(_y):
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        return None
    return _pipe(
        CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            auto_class_weights="Balanced",
            random_seed=settings.random_seed,
            verbose=0,
            allow_writing_files=False,
        ),
        scale_numeric=False,
    )


# Ordered registry: baseline floor first, then increasingly powerful models.
_BUILDERS: dict[str, Callable] = {
    "dummy": _build_dummy,
    "logistic_regression": _build_logistic,
    "random_forest": _build_random_forest,
    "xgboost": _build_xgboost,
    "lightgbm": _build_lightgbm,
    "catboost": _build_catboost,
}


def available_models() -> list[str]:
    """Names of all registered candidates (including optional ones)."""
    return list(_BUILDERS)


def build_candidates(
    y_train: np.ndarray, include: list[str] | None = None
) -> dict[str, Pipeline]:
    """Instantiate candidate pipelines, skipping any whose library is missing.

    Parameters
    ----------
    y_train:
        Training labels — used to set imbalance weights.
    include:
        Optional subset of model names to build. Defaults to all available.
    """
    names = include or available_models()
    built: dict[str, Pipeline] = {}
    for name in names:
        builder = _BUILDERS.get(name)
        if builder is None:
            log.warning("Unknown candidate '%s' — skipping", name)
            continue
        pipe = builder(y_train)
        if pipe is None:
            log.warning("Candidate '%s' unavailable (library not installed) — skipping", name)
            continue
        built[name] = pipe
    return built
