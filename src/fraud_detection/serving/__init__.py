"""Phase 1 — low-latency inference service.

Loads the winning model bundle produced by the bake-off
(``artifacts/best_model.joblib``) and exposes it behind a FastAPI app that
returns a calibrated fraud probability and a cost-based block/allow decision.
"""

from __future__ import annotations

from fraud_detection.serving.bundle import Decision, ModelBundle, default_model_path

__all__ = ["Decision", "ModelBundle", "default_model_path"]
