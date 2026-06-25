"""Latency benchmark for the serving path.

Times the *single-transaction* scoring path used in production
(preprocess → model → calibrate → threshold) and reports p50/p95/p99, then
checks them against the 50 ms SLA. Single-row timing (not batched) mirrors how
the model is actually called when a customer is waiting at checkout.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraud_detection.data import schema
from fraud_detection.data.synthetic import generate
from fraud_detection.logging_utils import get_logger
from fraud_detection.serving.bundle import ModelBundle

log = get_logger(__name__)

LATENCY_SLA_MS = 50.0


@dataclass
class BenchmarkResult:
    n_samples: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    max_ms: float
    meets_sla: bool

    def as_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "meets_sla": self.meets_sla,
            "sla_ms": LATENCY_SLA_MS,
        }


def _sample_transactions(n: int) -> pd.DataFrame:
    """A small batch of realistic transactions to replay through the model."""
    df = generate(n_transactions=max(n, 200), fraud_rate=0.02, seed=123)
    return df[schema.feature_columns()].head(n).reset_index(drop=True)


def benchmark(
    bundle: ModelBundle,
    n_samples: int = 2000,
    warmup: int = 50,
) -> BenchmarkResult:
    """Measure per-transaction scoring latency over ``n_samples`` calls."""
    rows = _sample_transactions(n_samples)
    records = rows.to_dict("records")

    # Warm up (JIT/caches/first-call costs excluded from the measurement).
    for rec in records[: min(warmup, len(records))]:
        bundle.score(rec)

    timings: list[float] = []
    for rec in records:
        start = time.perf_counter()
        bundle.score(rec)
        timings.append((time.perf_counter() - start) * 1000.0)

    arr = np.asarray(timings)
    p99 = float(np.percentile(arr, 99))
    return BenchmarkResult(
        n_samples=len(arr),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=p99,
        mean_ms=float(arr.mean()),
        max_ms=float(arr.max()),
        meets_sla=p99 < LATENCY_SLA_MS,
    )
