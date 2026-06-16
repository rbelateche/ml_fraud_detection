"""Fraud-appropriate evaluation metrics and the business cost model.

Accuracy is meaningless at ~1% prevalence, so this module centres on:
- PR-AUC (average precision) — primary ranking metric for rare positives.
- ROC-AUC — secondary, optimistic under imbalance.
- Recall at a fixed precision — mirrors a review-team constraint.
- Precision at top-k% — "if we review the riskiest 1%, how many are fraud?".
- Brier score — calibration quality, feeds the calibration step.
- Cost-weighted loss — the actual business objective.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

from fraud_detection.config import CostModel, settings


@dataclass
class ClassificationMetrics:
    """Container for threshold-independent ranking/calibration metrics."""

    pr_auc: float
    roc_auc: float
    brier: float
    recall_at_90_precision: float
    precision_at_1pct: float
    precision_at_5pct: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> ClassificationMetrics:
    """Compute the full set of ranking/calibration metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    return ClassificationMetrics(
        pr_auc=float(average_precision_score(y_true, y_score)),
        roc_auc=float(roc_auc_score(y_true, y_score)),
        brier=float(brier_score_loss(y_true, y_score)),
        recall_at_90_precision=recall_at_precision(y_true, y_score, 0.90),
        precision_at_1pct=precision_at_top_k(y_true, y_score, 0.01),
        precision_at_5pct=precision_at_top_k(y_true, y_score, 0.05),
    )


def recall_at_precision(y_true: np.ndarray, y_score: np.ndarray, min_precision: float) -> float:
    """Highest recall achievable while keeping precision >= ``min_precision``."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    mask = precision[:-1] >= min_precision  # last point has no threshold
    if not mask.any():
        return 0.0
    return float(recall[:-1][mask].max())


def precision_at_top_k(y_true: np.ndarray, y_score: np.ndarray, k_frac: float) -> float:
    """Precision among the top ``k_frac`` fraction of highest-scoring rows."""
    y_true = np.asarray(y_true).astype(int)
    n = len(y_true)
    k = max(1, int(np.ceil(n * k_frac)))
    top_idx = np.argsort(-y_score)[:k]
    return float(y_true[top_idx].mean())


# --------------------------------------------------------------------------- #
# Cost model
# --------------------------------------------------------------------------- #
@dataclass
class CostResult:
    """Outcome of evaluating a decision threshold under the business cost model."""

    threshold: float
    expected_cost: float
    cost_per_txn: float
    n_false_positives: int
    n_false_negatives: int
    n_true_positives: int
    fraud_dollars_caught: float
    fraud_dollars_missed: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def total_cost(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amount: np.ndarray,
    threshold: float,
    cost: CostModel | None = None,
) -> CostResult:
    """Expected business cost if we block transactions scoring >= ``threshold``.

    - False positive (block legit): ``false_positive_cost`` per event.
    - False negative (miss fraud): ``amount * fn_amount_fraction + fn_fixed_cost``.
    - True positive (catch fraud): ``review_cost`` (we still pay to review/act).
    """
    cost = cost or settings.cost
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    amount = np.asarray(amount, dtype=float)

    flagged = y_score >= threshold
    tp = flagged & (y_true == 1)
    fp = flagged & (y_true == 0)
    fn = (~flagged) & (y_true == 1)

    fp_cost = fp.sum() * cost.false_positive_cost
    fn_cost = (
        amount[fn].sum() * cost.false_negative_amount_fraction
        + fn.sum() * cost.false_negative_fixed_cost
    )
    tp_cost = tp.sum() * cost.review_cost
    expected = float(fp_cost + fn_cost + tp_cost)

    return CostResult(
        threshold=float(threshold),
        expected_cost=expected,
        cost_per_txn=expected / max(len(y_true), 1),
        n_false_positives=int(fp.sum()),
        n_false_negatives=int(fn.sum()),
        n_true_positives=int(tp.sum()),
        fraud_dollars_caught=float(amount[tp].sum()),
        fraud_dollars_missed=float(amount[fn].sum()),
    )


def optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amount: np.ndarray,
    cost: CostModel | None = None,
    n_grid: int = 200,
) -> CostResult:
    """Find the threshold minimising expected business cost via a score grid."""
    y_score = np.asarray(y_score, dtype=float)
    lo, hi = float(y_score.min()), float(y_score.max())
    grid = np.unique(np.concatenate([np.linspace(lo, hi, n_grid), [0.5]]))

    best: CostResult | None = None
    for t in grid:
        res = total_cost(y_true, y_score, amount, t, cost)
        if best is None or res.expected_cost < best.expected_cost:
            best = res
    assert best is not None
    return best


def cost_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    amount: np.ndarray,
    cost: CostModel | None = None,
    n_grid: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (thresholds, expected_cost) arrays for plotting the cost curve."""
    y_score = np.asarray(y_score, dtype=float)
    grid = np.linspace(float(y_score.min()), float(y_score.max()), n_grid)
    costs = np.array([total_cost(y_true, y_score, amount, t, cost).expected_cost for t in grid])
    return grid, costs
