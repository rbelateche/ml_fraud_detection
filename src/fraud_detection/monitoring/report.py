"""Drift report — per-feature + score PSI rolled into one verdict.

``compute_drift`` runs PSI over every model feature (and, optionally, the model
score distribution) and packages the result as a ``DriftReport`` that knows how
to render a human summary, serialise to a dict/JSON, and raise an **alert** when
any signal shows a *major* shift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraud_detection.data import schema
from fraud_detection.monitoring.drift import (
    DEFAULT_BINS,
    categorical_psi,
    numeric_psi,
    severity,
)


@dataclass(frozen=True)
class FeatureDrift:
    """PSI result for a single feature (or the model score)."""

    feature: str
    psi: float
    severity: str  # "none" | "moderate" | "major"
    kind: str  # "numeric" | "categorical"

    def as_dict(self) -> dict:
        return {
            "feature": self.feature,
            "psi": round(self.psi, 4),
            "severity": self.severity,
            "kind": self.kind,
        }


@dataclass
class DriftReport:
    """The full drift picture for one reference-vs-current comparison."""

    features: list[FeatureDrift]
    score_psi: float | None
    score_severity: str | None
    n_reference: int
    n_current: int

    @property
    def drifted(self) -> list[FeatureDrift]:
        """Features that moved at least moderately, worst first."""
        return [f for f in self.features if f.severity != "none"]

    @property
    def max_psi(self) -> float:
        return max((f.psi for f in self.features), default=0.0)

    @property
    def alert(self) -> bool:
        """True when any feature *or* the score distribution shifted majorly."""
        feature_major = any(f.severity == "major" for f in self.features)
        return feature_major or self.score_severity == "major"

    def as_dict(self) -> dict:
        return {
            "n_reference": self.n_reference,
            "n_current": self.n_current,
            "alert": self.alert,
            "max_psi": round(self.max_psi, 4),
            "score_psi": None if self.score_psi is None else round(self.score_psi, 4),
            "score_severity": self.score_severity,
            "features": [f.as_dict() for f in self.features],
        }

    def summary(self) -> str:
        """A compact, aligned table for the terminal."""
        lines = [
            "================ DRIFT REPORT ================",
            f"  reference rows : {self.n_reference}",
            f"  current rows   : {self.n_current}",
            f"  max feature PSI: {self.max_psi:.4f}",
        ]
        if self.score_psi is not None:
            lines.append(f"  score PSI      : {self.score_psi:.4f}  ({self.score_severity})")
        lines.append("  ---------------------------------------------")
        lines.append(f"  {'feature':<22}{'PSI':>10}  severity")
        for f in self.features:
            flag = "" if f.severity == "none" else ("  ⚠️" if f.severity == "moderate" else "  🚨")
            lines.append(f"  {f.feature:<22}{f.psi:>10.4f}  {f.severity}{flag}")
        lines.append("  ---------------------------------------------")
        verdict = "🚨 ALERT — major drift detected" if self.alert else "✅ no major drift"
        lines.append(f"  {verdict}")
        lines.append("==============================================")
        return "\n".join(lines)


def compute_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    bins: int = DEFAULT_BINS,
    scores_reference=None,
    scores_current=None,
) -> DriftReport:
    """Compute PSI for every feature (and optionally the score) → ``DriftReport``.

    Parameters
    ----------
    reference, current:
        Frames sharing the canonical schema. ``reference`` is the baseline the
        model knows; ``current`` is fresh traffic.
    numeric_features, categorical_features:
        Override the columns to check. Default to the canonical feature lists,
        intersected with the columns actually present.
    scores_reference, scores_current:
        Optional model-score arrays for the two windows → adds score drift,
        the single most actionable signal (the model's *output* is moving).
    """
    numeric_features = numeric_features or [
        c for c in schema.NUMERIC_FEATURES if c in reference.columns
    ]
    categorical_features = categorical_features or [
        c for c in schema.CATEGORICAL_FEATURES if c in reference.columns
    ]

    features: list[FeatureDrift] = []
    for col in numeric_features:
        psi = numeric_psi(reference[col].to_numpy(), current[col].to_numpy(), bins=bins)
        features.append(FeatureDrift(col, psi, severity(psi), "numeric"))
    for col in categorical_features:
        psi = categorical_psi(reference[col], current[col])
        features.append(FeatureDrift(col, psi, severity(psi), "categorical"))

    score_psi: float | None = None
    score_sev: str | None = None
    if scores_reference is not None and scores_current is not None:
        score_psi = numeric_psi(np.asarray(scores_reference), np.asarray(scores_current), bins=bins)
        score_sev = severity(score_psi)

    features.sort(key=lambda f: f.psi, reverse=True)
    return DriftReport(
        features=features,
        score_psi=score_psi,
        score_severity=score_sev,
        n_reference=len(reference),
        n_current=len(current),
    )
