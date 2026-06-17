"""Model bundle: load the winning pipeline and score a transaction.

The bake-off persists the chosen model as a self-contained bundle::

    {model_name, pipeline, calibrator, threshold, feature_columns}

This module wraps that bundle so both the FastAPI app and the latency
benchmark share the *exact same* scoring path — preprocessing → model →
calibration → cost-based threshold. Keeping a single scoring path is what
prevents training/serving skew.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fraud_detection.config import settings
from fraud_detection.data import schema
from fraud_detection.logging_utils import get_logger

log = get_logger(__name__)

DEFAULT_MODEL_FILENAME = "best_model.joblib"


def default_model_path() -> Path:
    """Where to load the model bundle from.

    ``FRAUD_MODEL_PATH`` overrides the default (``artifacts/best_model.joblib``),
    e.g. to point the container at a mounted volume or a downloaded artifact.
    """
    env = os.environ.get("FRAUD_MODEL_PATH")
    if env:
        return Path(env)
    return settings.paths.artifacts / DEFAULT_MODEL_FILENAME


@dataclass(frozen=True)
class Decision:
    """The outcome of scoring one transaction."""

    probability: float
    is_fraud: bool
    decision: str  # "block" | "allow"
    threshold: float
    model_name: str


class ModelBundle:
    """A loaded, ready-to-serve model bundle."""

    def __init__(
        self,
        model_name: str,
        pipeline,
        calibrator,
        threshold: float,
        feature_columns: list[str],
    ) -> None:
        self.model_name = model_name
        self.pipeline = pipeline
        self.calibrator = calibrator
        self.threshold = float(threshold)
        self.feature_columns = list(feature_columns)

    @classmethod
    def load(cls, path: str | Path | None = None) -> ModelBundle:
        """Load the bundle from disk (warm-loaded once at service startup)."""
        path = Path(path) if path is not None else default_model_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Model bundle not found at {path}. Run `fraud-bakeoff` to produce "
                "artifacts/best_model.joblib, or set FRAUD_MODEL_PATH."
            )
        raw = joblib.load(path)
        bundle = cls(
            model_name=raw["model_name"],
            pipeline=raw["pipeline"],
            calibrator=raw["calibrator"],
            threshold=raw["threshold"],
            feature_columns=raw.get("feature_columns") or schema.feature_columns(),
        )
        log.info(
            "Loaded model '%s' (threshold=%.4f, %d features) from %s",
            bundle.model_name,
            bundle.threshold,
            len(bundle.feature_columns),
            path,
        )
        return bundle

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Calibrated fraud probabilities for a frame of transactions."""
        x = frame[self.feature_columns]
        raw = self.pipeline.predict_proba(x)[:, 1]
        return np.clip(self.calibrator.predict(raw), 0.0, 1.0)

    def score(self, features: dict) -> Decision:
        """Score a single transaction dict and apply the operating threshold."""
        frame = pd.DataFrame(
            [{col: features.get(col) for col in self.feature_columns}],
            columns=self.feature_columns,
        )
        prob = float(self.predict_proba(frame)[0])
        is_fraud = prob >= self.threshold
        return Decision(
            probability=prob,
            is_fraud=is_fraud,
            decision="block" if is_fraud else "allow",
            threshold=self.threshold,
            model_name=self.model_name,
        )
