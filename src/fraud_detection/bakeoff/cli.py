"""CLI: run the model bake-off (``fraud-bakeoff``).

Examples:
    fraud-bakeoff                  # full tournament, default params
    fraud-bakeoff --tune           # add Optuna tuning (equal budget per model)
    fraud-bakeoff --quick          # fast path for CI / smoke tests
    fraud-bakeoff --models lightgbm xgboost logistic_regression
    fraud-bakeoff --no-mlflow      # skip experiment tracking
"""

from __future__ import annotations

import argparse

from fraud_detection.bakeoff.runner import BakeoffConfig, run, save_summary_json
from fraud_detection.logging_utils import get_logger

log = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the fraud model bake-off.")
    parser.add_argument("--source", default=None, help="synthetic | kaggle")
    parser.add_argument("--models", nargs="*", default=None, help="subset of candidates")
    parser.add_argument("--tune", action="store_true", help="enable Optuna tuning")
    parser.add_argument("--n-trials", type=int, default=20, help="Optuna trials per model")
    parser.add_argument("--quick", action="store_true", help="fast path (CI/smoke)")
    parser.add_argument("--no-mlflow", action="store_true", help="disable MLflow logging")
    args = parser.parse_args(argv)

    cfg = BakeoffConfig(
        source=args.source,
        include=args.models,
        tune=args.tune,
        n_trials=args.n_trials,
        quick=args.quick,
        log_mlflow=not args.no_mlflow,
    )

    output = run(cfg)
    save_summary_json(output)

    print("\n=================== LEADERBOARD ===================")
    print(output.leaderboard.to_string(index=False))
    print("===================================================")
    print(f"\nWinner: {output.best_model}")
    print("\nArtifacts:")
    for k, v in output.artifacts.items():
        print(f"  {k:16s}: {v}")


if __name__ == "__main__":
    main()
