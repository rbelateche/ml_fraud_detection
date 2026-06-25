"""Phase 4 — monitoring.

Two complementary capabilities:

- **Drift detection** (``drift.py`` + ``report.py``): Population Stability Index
  (PSI) over input features and model scores, computed in pure numpy/pandas so
  it is fast, dependency-light and CI-stable. PSI answers *"has the live data
  moved away from what the model was trained on?"* — the first warning sign that
  a model is going stale.
- **Live serving metrics** (``metrics.py``): Prometheus instrumentation for the
  inference API (request count, latency histogram, score distribution,
  block/allow counts) exposed at ``GET /metrics``.

Both share the project's golden rule: one schema, computed the same way offline
and online (see ``data/schema.py``).
"""

from __future__ import annotations

from fraud_detection.monitoring.drift import (
    PSI_MAJOR,
    PSI_MODERATE,
    categorical_psi,
    numeric_psi,
    severity,
)
from fraud_detection.monitoring.report import DriftReport, FeatureDrift, compute_drift

__all__ = [
    "PSI_MAJOR",
    "PSI_MODERATE",
    "categorical_psi",
    "numeric_psi",
    "severity",
    "DriftReport",
    "FeatureDrift",
    "compute_drift",
]
