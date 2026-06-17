"""Time-based (out-of-time) dataset splitting.

Fraud drifts over time, so a random split leaks the future into the past and
inflates metrics. We split chronologically: earliest transactions train,
middle validate, latest test. This single choice is a strong maturity signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fraud_detection.config import settings
from fraud_detection.data import schema


@dataclass
class Split:
    """A chronological train/validation/test split."""

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, part in [("train", self.train), ("valid", self.valid), ("test", self.test)]:
            rows.append(
                {
                    "split": name,
                    "rows": len(part),
                    "fraud": int(part[schema.LABEL].sum()),
                    "fraud_rate_%": round(part[schema.LABEL].mean() * 100, 3),
                    "start": part[schema.TIMESTAMP].min(),
                    "end": part[schema.TIMESTAMP].max(),
                }
            )
        return pd.DataFrame(rows)


def time_based_split(
    df: pd.DataFrame,
    *,
    train_frac: float | None = None,
    valid_frac: float | None = None,
) -> Split:
    """Split ``df`` chronologically into train/validation/test.

    The frame is sorted by timestamp; the first ``train_frac`` rows become train,
    the next ``valid_frac`` validation, and the remainder test.
    """
    train_frac = train_frac if train_frac is not None else settings.train_frac
    valid_frac = valid_frac if valid_frac is not None else settings.valid_frac
    if not 0 < train_frac < 1 or not 0 < valid_frac < 1 or train_frac + valid_frac >= 1:
        raise ValueError("train_frac and valid_frac must be in (0,1) and sum to < 1.")

    ordered = df.sort_values(schema.TIMESTAMP).reset_index(drop=True)
    n = len(ordered)
    n_train = int(n * train_frac)
    n_valid = int(n * valid_frac)

    train = ordered.iloc[:n_train].copy()
    valid = ordered.iloc[n_train : n_train + n_valid].copy()
    test = ordered.iloc[n_train + n_valid :].copy()
    return Split(train=train, valid=valid, test=test)
