"""Phase 0 baselines: the floor every later model is judged against.

- A naive/dummy classifier — contextualises every metric against "do nothing".
- Logistic regression with class weights — an interpretable, honest baseline.

Run with: ``fraud-baseline``.
"""

from __future__ import annotations

import argparse

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fraud_detection.config import settings
from fraud_detection.data.loader import load_dataset
from fraud_detection.data.split import time_based_split
from fraud_detection.logging_utils import get_logger
from fraud_detection.metrics import compute_metrics, optimal_threshold
from fraud_detection.models.preprocessing import build_preprocessor, split_xy

log = get_logger(__name__)


def build_dummy() -> DummyClassifier:
    """Stratified dummy: predicts the prior fraud probability."""
    return DummyClassifier(strategy="stratified", random_state=settings.random_seed)


def build_logistic() -> Pipeline:
    """Class-weighted logistic regression on the shared preprocessor."""
    return Pipeline(
        [
            ("prep", build_preprocessor(scale_numeric=True)),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    C=1.0,
                    random_state=settings.random_seed,
                ),
            ),
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Phase 0 baselines.")
    parser.add_argument("--source", default=None)
    parser.add_argument("--save", action="store_true", help="persist the logistic baseline")
    args = parser.parse_args(argv)

    df = load_dataset(source=args.source)
    split = time_based_split(df)
    x_train, y_train = split_xy(split.train)
    x_test, y_test = split_xy(split.test)
    amount_test = split.test["amount"].to_numpy()

    print("\nSplit summary:")
    print(split.summary().to_string(index=False))

    results = {}
    for name, model, needs_df in [
        ("dummy", build_dummy(), False),
        ("logistic_regression", build_logistic(), True),
    ]:
        log.info("Training baseline: %s", name)
        if needs_df:
            model.fit(x_train, y_train)
            scores = model.predict_proba(x_test)[:, 1]
        else:
            # Dummy needs no features; fit on labels only.
            model.fit(x_train[[]].assign(_=0)[["_"]], y_train)
            scores = model.predict_proba(x_test[[]].assign(_=0)[["_"]])[:, 1]

        m = compute_metrics(y_test, scores)
        cost = optimal_threshold(y_test, scores, amount_test)
        results[name] = (m, cost)

        print(f"\n=== {name} ===")
        for k, v in m.as_dict().items():
            print(f"  {k:24s}: {v:.4f}")
        print(f"  optimal_threshold        : {cost.threshold:.4f}")
        print(f"  expected_cost_per_txn    : {cost.cost_per_txn:.4f}")

        if args.save and name == "logistic_regression":
            out = settings.paths.artifacts / "baseline_logistic.joblib"
            joblib.dump(model, out)
            log.info("Saved baseline model to %s", out)

    print("\nTakeaway: PR-AUC above the dummy floor confirms learnable signal;")
    print("logistic regression is the bar Phase 0.5 candidates must clear.")


if __name__ == "__main__":
    main()
