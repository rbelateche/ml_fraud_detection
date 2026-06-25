"""CLIs for the serving layer.

- ``fraud-serve``  — run the FastAPI app with uvicorn.
- ``fraud-bench``  — benchmark single-transaction scoring latency (p50/p95/p99).
"""

from __future__ import annotations

import argparse

from fraud_detection.logging_utils import get_logger
from fraud_detection.serving.benchmark import LATENCY_SLA_MS, benchmark
from fraud_detection.serving.bundle import ModelBundle, default_model_path

log = get_logger(__name__)


def serve(argv: list[str] | None = None) -> None:
    """Run the inference API server."""
    parser = argparse.ArgumentParser(description="Run the fraud-detection serving API.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument("--reload", action="store_true", help="Auto-reload (dev only).")
    args = parser.parse_args(argv)

    import uvicorn

    log.info("Serving model from %s on %s:%d", default_model_path(), args.host, args.port)
    uvicorn.run(
        "fraud_detection.serving.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def bench(argv: list[str] | None = None) -> None:
    """Benchmark the scoring latency and print the percentiles + SLA verdict."""
    parser = argparse.ArgumentParser(description="Benchmark fraud scoring latency.")
    parser.add_argument("--n", type=int, default=2000, help="Number of scored transactions.")
    parser.add_argument("--model-path", default=None, help="Override the model bundle path.")
    args = parser.parse_args(argv)

    bundle = ModelBundle.load(args.model_path)
    result = benchmark(bundle, n_samples=args.n)
    d = result.as_dict()

    print("\n=============== SERVING LATENCY ===============")
    print(f"  model        : {bundle.model_name}")
    print(f"  samples      : {d['n_samples']}")
    print(f"  p50 / p95    : {d['p50_ms']} ms / {d['p95_ms']} ms")
    print(f"  p99          : {d['p99_ms']} ms   (SLA < {LATENCY_SLA_MS:.0f} ms)")
    print(f"  mean / max   : {d['mean_ms']} ms / {d['max_ms']} ms")
    print(f"  meets SLA    : {'✅ yes' if d['meets_sla'] else '❌ NO'}")
    print("===============================================")


if __name__ == "__main__":
    serve()
