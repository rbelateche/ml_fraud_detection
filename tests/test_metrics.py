"""Tests for fraud metrics and the business cost model."""

from __future__ import annotations

import numpy as np

from fraud_detection.config import CostModel
from fraud_detection.metrics import (
    compute_metrics,
    optimal_threshold,
    precision_at_top_k,
    recall_at_precision,
    total_cost,
)


def _toy():
    # Perfectly separable scores: fraud rows score high.
    y = np.array([0, 0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.15, 0.3, 0.9, 0.85])
    amt = np.array([10.0, 20.0, 5.0, 8.0, 100.0, 200.0])
    return y, s, amt


def test_compute_metrics_perfect_separation():
    y, s, _ = _toy()
    m = compute_metrics(y, s)
    assert m.pr_auc > 0.99
    assert m.roc_auc > 0.99


def test_precision_at_top_k():
    y, s, _ = _toy()
    # Top 2 scores are both fraud -> precision 1.0.
    assert precision_at_top_k(y, s, 2 / 6) == 1.0


def test_recall_at_precision_perfect():
    y, s, _ = _toy()
    assert recall_at_precision(y, s, 0.9) == 1.0


def test_total_cost_counts_outcomes():
    y, s, amt = _toy()
    # Threshold 0.5 flags both fraud rows, no legit -> no FP/FN.
    res = total_cost(y, s, amt, threshold=0.5)
    assert res.n_false_negatives == 0
    assert res.n_false_positives == 0
    assert res.n_true_positives == 2


def test_missing_fraud_is_expensive():
    y, s, amt = _toy()
    cost = CostModel(false_positive_cost=5, false_negative_amount_fraction=1.0, review_cost=1.0)
    # Very high threshold flags nothing -> both frauds missed.
    high = total_cost(y, s, amt, threshold=0.99, cost=cost)
    assert high.n_false_negatives == 2
    assert high.fraud_dollars_missed == 300.0


def test_optimal_threshold_minimises_cost():
    y, s, amt = _toy()
    best = optimal_threshold(y, s, amt)
    # Optimal should catch the fraud (separable), so missed dollars are 0.
    assert best.fraud_dollars_missed == 0.0
