"""Unified dataset loader with on-disk caching.

Selects the backend from ``settings.data_source`` (``synthetic`` | ``kaggle``)
and caches the canonical frame as Parquet so repeated runs are fast and
deterministic.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraud_detection.config import settings
from fraud_detection.data import schema
from fraud_detection.logging_utils import get_logger

log = get_logger(__name__)


def _cache_path(source: str) -> Path:
    return settings.paths.processed / f"transactions_{source}.parquet"


def load_dataset(
    source: str | None = None,
    *,
    force: bool = False,
    cache: bool = True,
) -> pd.DataFrame:
    """Load the canonical transaction dataset.

    Parameters
    ----------
    source:
        Backend override. Defaults to ``settings.data_source``.
    force:
        Ignore any cached Parquet and regenerate/redownload.
    cache:
        Persist the result to Parquet for reuse.
    """
    source = (source or settings.data_source).lower()
    cache_file = _cache_path(source)

    if cache and not force and cache_file.exists():
        log.info("Loading cached dataset from %s", cache_file)
        return pd.read_parquet(cache_file)

    if source == "synthetic":
        from fraud_detection.data.synthetic import generate

        df = generate()
    elif source == "kaggle":
        from fraud_detection.data.kaggle_source import load as kaggle_load

        df = kaggle_load()
    else:
        raise ValueError(f"Unknown data source '{source}'. Use 'synthetic' or 'kaggle'.")

    df = df.sort_values(schema.TIMESTAMP).reset_index(drop=True)

    if cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file, index=False)
        log.info("Cached dataset to %s", cache_file)

    return df
