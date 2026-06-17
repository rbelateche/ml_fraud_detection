"""Inference latency measurement.

A model that wins PR-AUC but blows the 50 ms SLA loses. We measure single-row
``predict_proba`` latency (the online path) and report p50/p99 in milliseconds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LatencyResult:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    n_samples: int


def measure_latency(model, x_sample: pd.DataFrame, n_iters: int = 500) -> LatencyResult:
    """Time per-row ``predict_proba`` over ``n_iters`` single-row calls.

    Single-row timing reflects the real online serving path (one transaction at a
    time), which is what the <50 ms SLA applies to.
    """
    rows = [x_sample.iloc[[i % len(x_sample)]] for i in range(min(n_iters, 1000))]

    # Warm-up so we don't time first-call overhead / lazy init.
    for r in rows[:5]:
        model.predict_proba(r)

    timings = np.empty(len(rows), dtype=float)
    for i, r in enumerate(rows):
        t0 = time.perf_counter()
        model.predict_proba(r)
        timings[i] = (time.perf_counter() - t0) * 1000.0  # ms

    return LatencyResult(
        p50_ms=float(np.percentile(timings, 50)),
        p95_ms=float(np.percentile(timings, 95)),
        p99_ms=float(np.percentile(timings, 99)),
        mean_ms=float(timings.mean()),
        n_samples=len(rows),
    )
