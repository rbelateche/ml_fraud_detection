"""Tests for the Phase 1 serving layer: bundle, API contract, and benchmark."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fraud_detection.data import schema
from fraud_detection.models.calibration import fit_calibrator
from fraud_detection.models.candidates import build_candidates
from fraud_detection.models.preprocessing import split_xy
from fraud_detection.serving.app import create_app
from fraud_detection.serving.benchmark import benchmark
from fraud_detection.serving.bundle import Decision, ModelBundle
from fraud_detection.serving.schemas import Transaction


@pytest.fixture(scope="module")
def bundle(small_dataset) -> ModelBundle:
    """A small, fast, fitted bundle (no full bake-off needed)."""
    x, y = split_xy(small_dataset)
    pipe = build_candidates(y.to_numpy(), include=["logistic_regression"])["logistic_regression"]
    pipe.fit(x, y)
    raw = pipe.predict_proba(x)[:, 1]
    calibrator = fit_calibrator("sigmoid", raw, y.to_numpy())
    return ModelBundle(
        model_name="logistic_regression",
        pipeline=pipe,
        calibrator=calibrator,
        threshold=0.5,
        feature_columns=schema.feature_columns(),
    )


@pytest.fixture(scope="module")
def example_payload(small_dataset) -> dict:
    """A JSON-serialisable transaction taken from the synthetic dataset."""
    row = small_dataset[schema.feature_columns()].iloc[0]
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}


# --------------------------------------------------------------------------- #
# Schema parity — the API contract must mirror the model's feature schema.
# --------------------------------------------------------------------------- #
def test_request_schema_matches_feature_columns():
    assert set(Transaction.model_fields) == set(schema.feature_columns())


# --------------------------------------------------------------------------- #
# Bundle scoring + decision logic.
# --------------------------------------------------------------------------- #
def test_bundle_score_returns_valid_decision(bundle, example_payload):
    d = bundle.score(example_payload)
    assert isinstance(d, Decision)
    assert 0.0 <= d.probability <= 1.0
    assert d.decision in {"block", "allow"}
    assert d.is_fraud == (d.probability >= bundle.threshold)


def test_decision_follows_threshold(bundle, example_payload):
    bundle.threshold = 0.0
    assert bundle.score(example_payload).decision == "block"
    bundle.threshold = 1.0001
    assert bundle.score(example_payload).decision == "allow"
    bundle.threshold = 0.5  # restore


# --------------------------------------------------------------------------- #
# API contract.
# --------------------------------------------------------------------------- #
def test_health_and_model_endpoints(bundle):
    with TestClient(create_app(bundle)) as client:
        h = client.get("/health").json()
        assert h["status"] == "ok"
        assert h["model_loaded"] is True
        assert h["model_name"] == "logistic_regression"

        m = client.get("/model").json()
        assert m["n_features"] == len(schema.feature_columns())
        assert set(m["feature_columns"]) == set(schema.feature_columns())


def test_score_endpoint_matches_bundle(bundle, example_payload):
    """The HTTP path must produce the same decision as the in-process bundle."""
    expected = bundle.score(example_payload)
    with TestClient(create_app(bundle)) as client:
        resp = client.post("/score", json=example_payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_name"] == expected.model_name
        assert body["decision"] == expected.decision
        assert body["is_fraud"] == expected.is_fraud
        assert body["probability"] == pytest.approx(expected.probability, rel=1e-6)


def test_score_rejects_invalid_input(bundle, example_payload):
    with TestClient(create_app(bundle)) as client:
        bad = dict(example_payload)
        bad["hour"] = 99  # out of range -> 422
        assert client.post("/score", json=bad).status_code == 422

        missing = dict(example_payload)
        del missing["amount"]
        assert client.post("/score", json=missing).status_code == 422


def test_score_returns_503_without_model(monkeypatch, tmp_path, example_payload):
    # Point loading at a missing path so startup leaves the model unloaded.
    monkeypatch.setenv("FRAUD_MODEL_PATH", str(tmp_path / "missing.joblib"))
    with TestClient(create_app(bundle=None)) as client:
        assert client.get("/health").json()["model_loaded"] is False
        assert client.post("/score", json=example_payload).status_code == 503


# --------------------------------------------------------------------------- #
# Latency benchmark.
# --------------------------------------------------------------------------- #
def test_benchmark_orders_percentiles(bundle):
    res = benchmark(bundle, n_samples=200, warmup=10)
    assert res.n_samples == 200
    assert res.p50_ms <= res.p95_ms <= res.p99_ms
    assert res.max_ms >= res.p99_ms
    assert isinstance(res.meets_sla, bool)
