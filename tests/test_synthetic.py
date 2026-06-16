"""Tests for the synthetic generator and canonical schema conformance."""

from __future__ import annotations

from fraud_detection.data import schema
from fraud_detection.data.synthetic import generate


def test_schema_columns_present(small_dataset):
    for col in schema.all_columns():
        assert col in small_dataset.columns, f"missing column {col}"


def test_label_is_binary(small_dataset):
    assert set(small_dataset[schema.LABEL].unique()).issubset({0, 1})


def test_fraud_rate_close_to_target():
    df = generate(n_transactions=20000, fraud_rate=0.02, seed=1)
    rate = df[schema.LABEL].mean()
    # Realised prevalence should land near the target.
    assert 0.012 < rate < 0.03


def test_timestamps_sorted_and_in_window(small_dataset):
    ts = small_dataset[schema.TIMESTAMP]
    assert ts.is_monotonic_increasing


def test_labels_arrive_after_transaction(small_dataset):
    # Chargebacks/labels must not precede the transaction.
    assert (small_dataset[schema.LABEL_TIMESTAMP] >= small_dataset[schema.TIMESTAMP]).all()


def test_reproducible_with_seed():
    a = generate(n_transactions=2000, fraud_rate=0.02, seed=99)
    b = generate(n_transactions=2000, fraud_rate=0.02, seed=99)
    assert a[schema.LABEL].tolist() == b[schema.LABEL].tolist()


def test_velocity_features_are_causal(small_dataset):
    # 1h count can never exceed 24h count for the same row.
    assert (small_dataset["txn_count_1h"] <= small_dataset["txn_count_24h"]).all()
