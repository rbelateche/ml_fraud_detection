"""Optuna hyperparameter tuning for the bake-off.

Each tunable model gets an *equal, small* trial budget so the comparison stays
fair. Tuning optimises PR-AUC on a time-ordered validation split (no leakage).
Models without a meaningful search space (dummy, logistic) are returned as-is.

Tuning is optional — the bake-off runs with sensible defaults unless ``--tune``
is passed. This keeps the default run fast while still demonstrating the
capability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from fraud_detection.config import settings
from fraud_detection.logging_utils import get_logger
from fraud_detection.models.candidates import _pipe, _scale_pos_weight

log = get_logger(__name__)


def _suggest_xgb(trial, y) -> Pipeline | None:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None
    return _pipe(
        XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 200, 600, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 9),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            scale_pos_weight=_scale_pos_weight(y),
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=settings.random_seed,
        ),
        scale_numeric=False,
    )


def _suggest_lgbm(trial, y) -> Pipeline | None:
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        return None
    return _pipe(
        LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 300, 800, step=100),
            num_leaves=trial.suggest_int("num_leaves", 31, 127),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            scale_pos_weight=_scale_pos_weight(y),
            n_jobs=-1,
            random_state=settings.random_seed,
            verbose=-1,
        ),
        scale_numeric=False,
    )


def _suggest_catboost(trial, _y) -> Pipeline | None:
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        return None
    return _pipe(
        CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 300, 800, step=100),
            depth=trial.suggest_int("depth", 4, 9),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            auto_class_weights="Balanced",
            random_seed=settings.random_seed,
            verbose=0,
            allow_writing_files=False,
        ),
        scale_numeric=False,
    )


def _suggest_random_forest(trial, _y) -> Pipeline:
    from sklearn.ensemble import RandomForestClassifier

    return _pipe(
        RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 200, 500, step=100),
            max_depth=trial.suggest_categorical("max_depth", [None, 8, 12, 20]),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=settings.random_seed,
        ),
        scale_numeric=False,
    )


_SEARCHERS = {
    "random_forest": _suggest_random_forest,
    "xgboost": _suggest_xgb,
    "lightgbm": _suggest_lgbm,
    "catboost": _suggest_catboost,
}


def tunable_models() -> list[str]:
    return list(_SEARCHERS)


def build_tuned(name: str, params: dict, y_train: np.ndarray) -> Pipeline | None:
    """Reconstruct a pipeline for ``name`` from Optuna ``best_params``."""
    import optuna

    searcher = _SEARCHERS.get(name)
    if searcher is None:
        return None
    return searcher(optuna.trial.FixedTrial(params), y_train)


def tune_model(
    name: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
    n_trials: int,
) -> dict | None:
    """Run Optuna for one model. Returns ``{best_params, best_value}`` or None.

    Returns None when the model is not tunable or its library is unavailable.
    """
    searcher = _SEARCHERS.get(name)
    if searcher is None:
        return None

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        pipe = searcher(trial, y_train)
        if pipe is None:
            raise optuna.TrialPruned()
        pipe.fit(x_train, y_train)
        scores = pipe.predict_proba(x_valid)[:, 1]
        return average_precision_score(y_valid, scores)

    # Probe once to detect an unavailable library before creating a study.
    probe = optuna.trial.FixedTrial(
        {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "num_leaves": 63,
            "iterations": 300,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "min_samples_leaf": 5,
        }
    )
    if searcher(probe, y_train) is None:
        return None

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=settings.random_seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info("Tuned %s: PR-AUC=%.4f params=%s", name, study.best_value, study.best_params)
    return {"best_params": study.best_params, "best_value": float(study.best_value)}
