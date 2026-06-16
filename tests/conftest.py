"""Shared pytest fixtures — a small, fast synthetic dataset for unit tests."""

from __future__ import annotations

import pandas as pd
import pytest

from fraud_detection.data.synthetic import generate


@pytest.fixture(scope="session")
def small_dataset() -> pd.DataFrame:
    """A tiny dataset so the whole suite runs in a couple of seconds."""
    return generate(n_transactions=4000, fraud_rate=0.03, seed=7)
