"""Probability calibration comparison.

A calibrated probability is what makes a cost-based threshold meaningful: if the
model says 0.3, roughly 30% of such transactions should truly be fraud. We do not
pre-decide a method — we compare three on a held-out set and let the data choose:

- **raw**       — the model's own ``predict_proba`` (no calibration).
- **sigmoid**   — Platt scaling: a 1-D logistic fit on the scores.
- **isotonic**  — a non-parametric monotonic fit (flexible, but needs enough
                  positives or it overfits).

Selection is by **Brier score** on the calibration/validation split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from fraud_detection.logging_utils import get_logger

log = get_logger(__name__)


class Calibrator:
    """A picklable probability calibrator.

    Stores the fitted mapping (``None`` for raw, a logistic model for sigmoid, or
    an isotonic model for isotonic) so the whole object can be persisted with the
    winning model bundle for serving.
    """

    def __init__(self, method: str, model=None):
        self.method = method
        self.model = model

    def predict(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if self.method == "raw":
            out = scores
        elif self.method == "sigmoid":
            out = self.model.predict_proba(scores.reshape(-1, 1))[:, 1]
        elif self.method == "isotonic":
            out = self.model.predict(scores)
        else:
            raise ValueError(f"Unknown calibration method '{self.method}'.")
        return np.clip(out, 0.0, 1.0)


def _fit_sigmoid(scores: np.ndarray, y: np.ndarray) -> Calibrator:
    lr = LogisticRegression(max_iter=1000)
    lr.fit(scores.reshape(-1, 1), y)
    return Calibrator("sigmoid", lr)


def _fit_isotonic(scores: np.ndarray, y: np.ndarray) -> Calibrator:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(scores, y)
    return Calibrator("isotonic", iso)


def _identity(scores: np.ndarray) -> Calibrator:
    return Calibrator("raw", None)


def fit_calibrator(method: str, scores: np.ndarray, y: np.ndarray) -> Calibrator:
    """Fit a single named calibrator (``raw`` | ``sigmoid`` | ``isotonic``)."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y).astype(int)
    if method == "raw":
        return _identity(scores)
    if method == "sigmoid":
        return _fit_sigmoid(scores, y)
    if method == "isotonic":
        return _fit_isotonic(scores, y)
    raise ValueError(f"Unknown calibration method '{method}'.")


@dataclass
class CalibrationResult:
    """Outcome of the calibration comparison."""

    best_method: str
    brier_by_method: dict[str, float]
    calibrator: Calibrator


# Heuristic floor: isotonic needs enough positives to avoid overfitting.
_MIN_POS_FOR_ISOTONIC = 50


def compare_calibration(
    cal_scores: np.ndarray,
    cal_y: np.ndarray,
    *,
    eval_scores: np.ndarray | None = None,
    eval_y: np.ndarray | None = None,
) -> CalibrationResult:
    """Fit and compare calibration methods; return the best by Brier score.

    Calibrators are *fit* on ``cal_scores``/``cal_y`` and *scored* on the eval set
    (defaults to the calibration set itself if no eval set is supplied).
    """
    cal_scores = np.asarray(cal_scores, dtype=float)
    cal_y = np.asarray(cal_y).astype(int)
    es = eval_scores if eval_scores is not None else cal_scores
    ey = eval_y if eval_y is not None else cal_y
    es = np.asarray(es, dtype=float)
    ey = np.asarray(ey).astype(int)

    n_pos = int(cal_y.sum())
    candidates = {
        "raw": _identity(cal_scores),
        "sigmoid": _fit_sigmoid(cal_scores, cal_y),
    }
    # Only offer isotonic when there are enough positives to support it.
    if n_pos >= _MIN_POS_FOR_ISOTONIC:
        candidates["isotonic"] = _fit_isotonic(cal_scores, cal_y)
    else:
        log.info(
            "Skipping isotonic calibration: only %d positives (< %d).",
            n_pos,
            _MIN_POS_FOR_ISOTONIC,
        )

    brier_by_method = {
        name: float(brier_score_loss(ey, cal.predict(es))) for name, cal in candidates.items()
    }
    best_method = min(brier_by_method, key=brier_by_method.get)
    log.info("Calibration Brier by method: %s -> best=%s", brier_by_method, best_method)

    return CalibrationResult(
        best_method=best_method,
        brier_by_method=brier_by_method,
        calibrator=candidates[best_method],
    )
