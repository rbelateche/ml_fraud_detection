"""Tests for the Phase 4 monitoring layer: PSI drift + Prometheus metrics."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fraud_detection.data import schema
from fraud_detection.models.calibration import fit_calibrator
from fraud_detection.models.candidates import build_candidates
from fraud_detection.models.preprocessing import split_xy
from fraud_detection.monitoring.cli import inject_drift
from fraud_detection.monitoring.drift import (
    PSI_MAJOR,
    PSI_MODERATE,
    categorical_psi,
    numeric_psi,
    severity,
)
from fraud_detection.monitoring.metrics import build_serving_metrics, prometheus_available
from fraud_detection.monitoring.report import DriftReport, compute_drift
from fraud_detection.serving.app import create_app
from fraud_detection.serving.bundle import ModelBundle


# --------------------------------------------------------------------------- #
# PSI maths.
# --------------------------------------------------------------------------- #
def test_numeric_psi_is_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    b = rng.normal(size=5000)
    assert numeric_psi(a, b) < PSI_MODERATE


def test_numeric_psi_is_large_for_shifted_distribution():
    rng = np.random.default_rng(1)
    a = rng.normal(loc=0.0, size=5000)
    b = rng.normal(loc=3.0, size=5000)  # big mean shift
    assert numeric_psi(a, b) >= PSI_MAJOR


def test_categorical_psi_detects_a_frequency_shift():
    ref = ["a"] * 500 + ["b"] * 500
    cur = ["a"] * 950 + ["b"] * 50  # mix flipped
    assert categorical_psi(ref, cur) >= PSI_MAJOR
    assert categorical_psi(ref, ref) < PSI_MODERATE


def test_severity_thresholds():
    assert severity(0.0) == "none"
    assert severity(PSI_MODERATE) == "moderate"
    assert severity(PSI_MAJOR) == "major"


# --------------------------------------------------------------------------- #
# Drift report.
# --------------------------------------------------------------------------- #
def test_compute_drift_no_alert_on_same_data(small_dataset):
    mid = len(small_dataset) // 2
    ref = small_dataset.iloc[:mid].reset_index(drop=True)
    cur = small_dataset.iloc[mid:].reset_index(drop=True)
    report = compute_drift(ref, cur)
    assert isinstance(report, DriftReport)
    # Same generator, two halves → no major drift expected.
    assert report.alert is False
    # One FeatureDrift per canonical feature.
    assert len(report.features) == len(schema.feature_columns())


def test_compute_drift_fires_alert_on_injected_drift(small_dataset):
    mid = len(small_dataset) // 2
    ref = small_dataset.iloc[:mid].reset_index(drop=True)
    cur = inject_drift(small_dataset.iloc[mid:].reset_index(drop=True))
    report = compute_drift(ref, cur)
    assert report.alert is True
    assert report.max_psi >= PSI_MAJOR
    assert any(f.feature == "amount" for f in report.drifted)


def test_drift_report_serialises(small_dataset):
    mid = len(small_dataset) // 2
    report = compute_drift(
        small_dataset.iloc[:mid].reset_index(drop=True),
        small_dataset.iloc[mid:].reset_index(drop=True),
    )
    d = report.as_dict()
    assert {"alert", "max_psi", "features"} <= set(d)
    assert "DRIFT REPORT" in report.summary()


# --------------------------------------------------------------------------- #
# Prometheus serving metrics.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def bundle(small_dataset) -> ModelBundle:
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


def test_serving_metrics_observe_and_render():
    metrics = build_serving_metrics()
    assert prometheus_available() is True
    assert metrics is not None
    metrics.observe(probability=0.8, decision="block", latency_seconds=0.003)
    payload, content_type = metrics.render()
    text = payload.decode()
    assert "fraud_score_requests_total" in text
    assert 'fraud_decisions_total{decision="block"}' in text
    assert "text/plain" in content_type


def test_metrics_endpoint_reports_after_scoring(bundle, small_dataset):
    app = create_app(bundle=bundle)
    client = TestClient(app)
    payload = {
        k: (v.item() if hasattr(v, "item") else v)
        for k, v in small_dataset[schema.feature_columns()].iloc[0].items()
    }
    assert client.post("/score", json=payload).status_code == 200

    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "fraud_score_requests_total" in body
    assert "fraud_score_latency_seconds" in body
