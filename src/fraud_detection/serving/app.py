"""FastAPI inference service.

Endpoints
---------
- ``GET  /health`` — liveness + whether a model is loaded.
- ``GET  /model``  — metadata about the serving model.
- ``POST /score``  — score one transaction → calibrated probability + decision.

The model bundle is warm-loaded once at startup (lifespan) and kept on
``app.state`` so each request avoids disk I/O. Tests inject a bundle directly.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from fraud_detection.logging_utils import get_logger
from fraud_detection.serving.bundle import ModelBundle
from fraud_detection.serving.schemas import (
    HealthResponse,
    ModelInfo,
    ScoreResponse,
    Transaction,
)

log = get_logger(__name__)


def _require_bundle(request: Request) -> ModelBundle:
    bundle = request.app.state.bundle
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `fraud-bakeoff` to produce "
            "artifacts/best_model.joblib, or set FRAUD_MODEL_PATH.",
        )
    return bundle


def create_app(bundle: ModelBundle | None = None) -> FastAPI:
    """Build the FastAPI app. Pass a ``bundle`` to inject one (tests); otherwise
    it is loaded from disk at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.bundle is None:
            try:
                app.state.bundle = ModelBundle.load()
            except FileNotFoundError as exc:
                log.warning("Starting without a model: %s", exc)
                app.state.bundle = None
        yield

    app = FastAPI(
        title="Fraud Detection — Serving",
        version="0.1.0",
        summary="Real-time fraud scoring with a cost-based decision threshold.",
        lifespan=lifespan,
    )
    app.state.bundle = bundle

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(request: Request) -> HealthResponse:
        b = request.app.state.bundle
        return HealthResponse(
            status="ok",
            model_loaded=b is not None,
            model_name=b.model_name if b is not None else None,
        )

    @app.get("/model", response_model=ModelInfo, tags=["ops"])
    def model_info(request: Request) -> ModelInfo:
        b = _require_bundle(request)
        return ModelInfo(
            model_name=b.model_name,
            threshold=b.threshold,
            n_features=len(b.feature_columns),
            feature_columns=b.feature_columns,
        )

    @app.post("/score", response_model=ScoreResponse, tags=["inference"])
    def score(txn: Transaction, request: Request) -> ScoreResponse:
        b = _require_bundle(request)
        d = b.score(txn.model_dump())
        return ScoreResponse(
            probability=d.probability,
            is_fraud=d.is_fraud,
            decision=d.decision,
            threshold=d.threshold,
            model_name=d.model_name,
        )

    return app


# Module-level app for `uvicorn fraud_detection.serving.app:app`.
app = create_app()
