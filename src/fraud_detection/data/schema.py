"""Canonical transaction schema.

A single schema definition shared by the synthetic generator, the Kaggle
adapter, EDA, training and (later) the serving layer. Keeping feature lists in
one place is what prevents training/serving skew down the line.
"""

from __future__ import annotations

# Identity / bookkeeping columns (not used as model features).
TRANSACTION_ID = "transaction_id"
CARD_ID = "card_id"
MERCHANT_ID = "merchant_id"
TIMESTAMP = "timestamp"

# Target.
LABEL = "is_fraud"

# Model features.
NUMERIC_FEATURES: list[str] = [
    "amount",
    "amount_log",
    "hour",
    "day_of_week",
    "is_night",
    "is_weekend",
    "card_age_days",
    "txn_count_1h",
    "txn_count_24h",
    "amount_mean_24h",
    "amount_to_mean_ratio",
    "distance_from_home",
    "merchant_risk",
]

CATEGORICAL_FEATURES: list[str] = [
    "category",
    "channel",
    "device_type",
]

# Label arrival metadata — Phase 5 uses this to simulate delayed chargebacks.
LABEL_TIMESTAMP = "label_timestamp"


def feature_columns() -> list[str]:
    """Full ordered list of model input columns."""
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES


def all_columns() -> list[str]:
    """Every column the canonical dataset is expected to expose."""
    return [
        TRANSACTION_ID,
        CARD_ID,
        MERCHANT_ID,
        TIMESTAMP,
        *feature_columns(),
        LABEL,
        LABEL_TIMESTAMP,
    ]
