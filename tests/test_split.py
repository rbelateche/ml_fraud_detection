"""Tests for time-based splitting — the no-future-leakage guarantee."""

from __future__ import annotations

from fraud_detection.data import schema
from fraud_detection.data.split import time_based_split


def test_split_is_chronological(small_dataset):
    split = time_based_split(small_dataset, train_frac=0.6, valid_frac=0.2)
    # Each split's max timestamp must be <= the next split's min timestamp.
    assert split.train[schema.TIMESTAMP].max() <= split.valid[schema.TIMESTAMP].min()
    assert split.valid[schema.TIMESTAMP].max() <= split.test[schema.TIMESTAMP].min()


def test_split_sizes_sum_to_total(small_dataset):
    split = time_based_split(small_dataset, train_frac=0.6, valid_frac=0.2)
    assert len(split.train) + len(split.valid) + len(split.test) == len(small_dataset)


def test_split_preserves_some_fraud_in_each_part(small_dataset):
    split = time_based_split(small_dataset, train_frac=0.6, valid_frac=0.2)
    for part in (split.train, split.valid, split.test):
        assert part[schema.LABEL].sum() > 0


def test_summary_has_three_rows(small_dataset):
    split = time_based_split(small_dataset)
    assert len(split.summary()) == 3
