"""Bake-off plots — the artifacts reviewers actually look at.

All figures are written headlessly to ``reports/figures``:
- PR curves overlaid for every candidate.
- Reliability (calibration) diagram before vs after calibration.
- Latency (p99) vs PR-AUC scatter — the trade-off plot that justifies the pick.
- Cost curve for the chosen model.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from fraud_detection.config import settings  # noqa: E402
from fraud_detection.metrics import cost_curve  # noqa: E402


def _fig_dir() -> Path:
    d = settings.paths.figures
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_pr_curves(y_true: np.ndarray, scores_by_model: dict[str, np.ndarray]) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, scores in scores_by_model.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ax.plot(recall, precision, label=name, linewidth=1.5)
    baseline = float(np.mean(y_true))
    ax.axhline(baseline, color="grey", linestyle="--", linewidth=1,
               label=f"prevalence = {baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves (test set)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = _fig_dir() / "bakeoff_pr_curves.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_reliability(
    y_true: np.ndarray, raw_scores: np.ndarray, calibrated_scores: np.ndarray, model_name: str
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfectly calibrated")
    for scores, label in [(raw_scores, "raw"), (calibrated_scores, "calibrated")]:
        frac_pos, mean_pred = calibration_curve(y_true, scores, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=1.5, label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction of fraud")
    ax.set_title(f"Reliability diagram — {model_name}")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out = _fig_dir() / "bakeoff_reliability.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_latency_vs_pr_auc(rows: list[dict]) -> Path:
    """Scatter of p99 latency vs PR-AUC; the SLA line makes the trade-off explicit."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in rows:
        ax.scatter(r["p99_ms"], r["pr_auc"], s=60)
        ax.annotate(r["model"], (r["p99_ms"], r["pr_auc"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.axvline(50, color="red", linestyle="--", linewidth=1, label="50 ms SLA")
    ax.set_xlabel("p99 inference latency (ms)")
    ax.set_ylabel("PR-AUC (test)")
    ax.set_title("Latency vs PR-AUC trade-off")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = _fig_dir() / "bakeoff_latency_vs_prauc.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_cost_curve(
    y_true: np.ndarray, scores: np.ndarray, amount: np.ndarray, chosen_threshold: float,
    model_name: str,
) -> Path:
    thresholds, costs = cost_curve(y_true, scores, amount)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, costs, color="#4c72b0", linewidth=1.5)
    ax.axvline(chosen_threshold, color="green", linestyle="--", linewidth=1,
               label=f"chosen threshold = {chosen_threshold:.3f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected business cost ($)")
    ax.set_title(f"Cost vs threshold — {model_name}")
    ax.legend(loc="upper center", fontsize=9)
    fig.tight_layout()
    out = _fig_dir() / "bakeoff_cost_curve.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
