"""Prometheus metrics for the inference service.

Exposes the four numbers an on-call engineer actually wants while transactions
are flowing:

- ``fraud_score_requests_total``     — throughput (a counter).
- ``fraud_decisions_total{decision}``— block vs allow split (a counter).
- ``fraud_score_latency_seconds``    — scoring latency (a histogram → percentiles).
- ``fraud_score_probability``        — predicted-probability distribution
  (a histogram → the earliest, cheapest signal of score drift in real time).

``prometheus_client`` is an optional dependency (the ``monitoring`` extra). If it
is not installed, :func:`build_serving_metrics` returns ``None`` and the serving
app simply answers ``503`` on ``/metrics`` — the API itself keeps working. Each
``ServingMetrics`` owns a private ``CollectorRegistry`` so building several apps
(e.g. in tests) never collides on the global default registry.
"""

from __future__ import annotations

try:  # optional dependency — only present with the `monitoring` extra.
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Histogram,
        generate_latest,
    )

    _HAVE_PROMETHEUS = True
except ImportError:  # pragma: no cover - exercised only when extra is absent.
    _HAVE_PROMETHEUS = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"


# Latency buckets in seconds, centred around the 50 ms SLA.
_LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25)
# Probability buckets across [0, 1], denser near the low operating thresholds.
_SCORE_BUCKETS = (0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)


def prometheus_available() -> bool:
    """Whether ``prometheus_client`` is importable."""
    return _HAVE_PROMETHEUS


class ServingMetrics:
    """A self-contained set of Prometheus collectors for the serving app."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        if not _HAVE_PROMETHEUS:  # pragma: no cover - guarded by the factory.
            raise RuntimeError(
                "prometheus_client is not installed. Install the 'monitoring' extra."
            )
        self.registry = registry or CollectorRegistry()
        self.requests = Counter(
            "fraud_score_requests_total",
            "Total number of scoring requests served.",
            registry=self.registry,
        )
        self.decisions = Counter(
            "fraud_decisions_total",
            "Scoring decisions, partitioned by outcome.",
            ["decision"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "fraud_score_latency_seconds",
            "Per-request scoring latency in seconds.",
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.score = Histogram(
            "fraud_score_probability",
            "Distribution of predicted fraud probabilities.",
            buckets=_SCORE_BUCKETS,
            registry=self.registry,
        )

    def observe(self, *, probability: float, decision: str, latency_seconds: float) -> None:
        """Record one scoring event across all collectors."""
        self.requests.inc()
        self.decisions.labels(decision=decision).inc()
        self.latency.observe(latency_seconds)
        self.score.observe(probability)

    def render(self) -> tuple[bytes, str]:
        """Return the exposition payload + its content type for ``/metrics``."""
        return generate_latest(self.registry), CONTENT_TYPE_LATEST


def build_serving_metrics() -> ServingMetrics | None:
    """Construct a :class:`ServingMetrics`, or ``None`` if Prometheus is absent."""
    if not _HAVE_PROMETHEUS:
        return None
    return ServingMetrics()
