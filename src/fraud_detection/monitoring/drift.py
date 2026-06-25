"""Population Stability Index (PSI) — the drift metric.

PSI measures how much a distribution has shifted between a **reference** window
(what the model knows — e.g. the training/recent-baseline period) and a
**current** window (fresh production traffic). For each bin::

    PSI = Σ (current_pct - reference_pct) * ln(current_pct / reference_pct)

Standard reading of the result (industry convention):

- ``PSI < 0.10``  → no meaningful shift (stable).
- ``0.10–0.25``   → moderate shift (watch / investigate).
- ``PSI ≥ 0.25``  → major shift (likely retrain / alert).

We compute it in pure numpy/pandas — no Evidently — so it is fast, transparent
and easy to reason about. Numeric features are binned on **reference quantiles**
(equal-frequency bins), which is robust to skew; categorical features compare
category frequencies directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Industry-standard PSI thresholds.
PSI_MODERATE = 0.10
PSI_MAJOR = 0.25

# Default number of equal-frequency bins for numeric features.
DEFAULT_BINS = 10

# Floor applied to bin proportions so empty bins never blow up the log term.
_EPS = 1e-6


def severity(psi: float) -> str:
    """Map a PSI value to ``"none"`` | ``"moderate"`` | ``"major"``."""
    if psi >= PSI_MAJOR:
        return "major"
    if psi >= PSI_MODERATE:
        return "moderate"
    return "none"


def _numeric_edges(reference: np.ndarray, bins: int) -> np.ndarray:
    """Equal-frequency bin edges from the reference distribution.

    Outer edges are pushed to ±inf so out-of-range values in the current window
    still fall into the first/last bin instead of being dropped.
    """
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if edges.size < 2:
        # Constant reference → a single catch-all bin.
        return np.array([-np.inf, np.inf])
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Per-bin proportions, floored at ``_EPS`` to keep the log finite."""
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    if total == 0:
        return np.full(counts.shape, _EPS)
    return np.clip(counts / total, _EPS, None)


def numeric_psi(reference, current, bins: int = DEFAULT_BINS) -> float:
    """PSI between two numeric samples using reference-quantile bins."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if ref.size == 0 or cur.size == 0:
        return 0.0
    edges = _numeric_edges(ref, bins)
    ref_p = _proportions(ref, edges)
    cur_p = _proportions(cur, edges)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def categorical_psi(reference, current) -> float:
    """PSI between two categorical samples comparing category frequencies."""
    ref = pd.Series(reference).dropna().astype("object")
    cur = pd.Series(current).dropna().astype("object")
    categories = sorted(set(ref.unique()) | set(cur.unique()), key=str)
    if not categories:
        return 0.0
    ref_total = max(len(ref), 1)
    cur_total = max(len(cur), 1)
    ref_counts = ref.value_counts()
    cur_counts = cur.value_counts()

    psi = 0.0
    for cat in categories:
        ref_p = max(ref_counts.get(cat, 0) / ref_total, _EPS)
        cur_p = max(cur_counts.get(cat, 0) / cur_total, _EPS)
        psi += (cur_p - ref_p) * np.log(cur_p / ref_p)
    return float(psi)
