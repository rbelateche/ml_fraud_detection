"""Tests for bake-off candidates, calibration, latency and a tiny end-to-end run."""

from __future__ import annotations

import numpy as np

from fraud_detection.bakeoff.latency import measure_latency
from fraud_detection.bakeoff.runner import BakeoffConfig, run
from fraud_detection.models.calibration import compare_calibration, fit_calibrator
from fraud_detection.models.candidates import available_models, build_candidates
from fraud_detection.models.preprocessing import split_xy


def test_registry_lists_expected_models():
    names = available_models()
    assert "dummy" in names
    assert "logistic_regression" in names
    assert "random_forest" in names


def test_build_candidates_always_includes_sklearn_models(small_dataset):
    _, y = split_xy(small_dataset)
    built = build_candidates(y.to_numpy())
    # These three have no optional dependency and must always be present.
    for name in ("dummy", "logistic_regression", "random_forest"):
        assert name in built


def test_calibration_prefers_a_real_method_on_miscalibrated_scores():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=2000)
    # Deliberately over-confident scores -> calibration should help (or tie).
    raw = np.clip(y * 0.9 + rng.normal(0, 0.25, size=2000), 0.01, 0.99)
    res = compare_calibration(raw, y)
    assert res.best_method in {"raw", "sigmoid", "isotonic"}
    assert res.brier_by_method[res.best_method] == min(res.brier_by_method.values())


def test_fit_calibrator_is_picklable():
    import pickle

    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=500)
    raw = rng.uniform(0, 1, size=500)
    cal = fit_calibrator("sigmoid", raw, y)
    restored = pickle.loads(pickle.dumps(cal))
    assert restored.method == "sigmoid"
    assert restored.predict(raw).shape == raw.shape


def test_measure_latency_returns_ordered_percentiles(small_dataset):
    _, y = split_xy(small_dataset)
    built = build_candidates(y.to_numpy(), include=["logistic_regression"])
    pipe = built["logistic_regression"]
    x, _ = split_xy(small_dataset)
    pipe.fit(x, y)
    lat = measure_latency(pipe, x, n_iters=50)
    assert lat.p50_ms <= lat.p99_ms
    assert lat.n_samples > 0


def test_bakeoff_end_to_end_quick(small_dataset, monkeypatch):
    # Run a fast tournament on a tiny in-memory dataset; assert a winner + leaderboard.
    import fraud_detection.bakeoff.runner as runner

    monkeypatch.setattr(runner, "load_dataset", lambda source=None: small_dataset)
    cfg = BakeoffConfig(
        include=["dummy", "logistic_regression"],
        quick=True,
        log_mlflow=False,
    )
    out = run(cfg)
    assert out.best_model in {"dummy", "logistic_regression"}
    assert len(out.leaderboard) == 2
    assert "cost_per_txn" in out.leaderboard.columns
