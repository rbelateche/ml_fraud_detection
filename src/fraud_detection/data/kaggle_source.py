"""Kaggle data backend (optional).

Downloads and adapts a public Kaggle dataset to the canonical schema. This is
an *optional* backend — the synthetic generator is the zero-credential default.

Setup (one-time):
    1. ``pip install -e '.[kaggle]'``
    2. Create an API token at https://www.kaggle.com/settings -> "Create New API
       Token". Save the downloaded ``kaggle.json`` to ``~/.kaggle/kaggle.json``
       and ``chmod 600`` it.
    3. For the IEEE-CIS *competition* data you must also accept the rules on the
       competition page once.

Then: ``FRAUD_DATA_SOURCE=kaggle fraud-data``.

The default dataset is Sparkov (``kartik2112/fraud-detection``) because it ships
timestamps and geolocation, which the streaming phase replays as a live feed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fraud_detection.config import settings
from fraud_detection.data import schema
from fraud_detection.logging_utils import get_logger

log = get_logger(__name__)


def download(dest: Path | None = None) -> Path:
    """Download the configured Kaggle dataset/competition into ``data/raw``.

    Returns the directory containing the extracted CSVs.
    """
    dest = dest or (settings.paths.raw / "kaggle")
    dest.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except (ImportError, OSError) as exc:  # OSError: missing credentials
        raise RuntimeError(
            "Kaggle backend unavailable. Install with `pip install -e '.[kaggle]'` "
            "and place kaggle.json at ~/.kaggle/kaggle.json (chmod 600). "
            "See module docstring for details."
        ) from exc

    api = KaggleApi()
    api.authenticate()

    if settings.kaggle_competition:
        log.info("Downloading Kaggle competition '%s'", settings.kaggle_competition)
        api.competition_download_files(
            settings.kaggle_competition, path=str(dest), quiet=False
        )
        _unzip_all(dest)
    else:
        log.info("Downloading Kaggle dataset '%s'", settings.kaggle_dataset)
        api.dataset_download_files(
            settings.kaggle_dataset, path=str(dest), unzip=True, quiet=False
        )
    return dest


def _unzip_all(folder: Path) -> None:
    import zipfile

    for zf in folder.glob("*.zip"):
        with zipfile.ZipFile(zf) as z:
            z.extractall(folder)


def load(dest: Path | None = None) -> pd.DataFrame:
    """Download (if needed) and adapt the Sparkov dataset to the canonical schema."""
    dest = dest or (settings.paths.raw / "kaggle")
    csvs = list(dest.glob("*.csv")) if dest.exists() else []
    if not csvs:
        download(dest)
        csvs = list(dest.glob("*.csv"))
    if not csvs:
        raise RuntimeError(f"No CSV files found in {dest} after download.")

    # Sparkov ships fraudTrain.csv / fraudTest.csv — concatenate both.
    frames = [pd.read_csv(p) for p in sorted(csvs)]
    raw = pd.concat(frames, ignore_index=True)
    log.info("Loaded %d raw Kaggle rows from %d file(s)", len(raw), len(csvs))
    return _adapt_sparkov(raw)


def _adapt_sparkov(raw: pd.DataFrame) -> pd.DataFrame:
    """Map Sparkov columns onto the canonical schema with engineered features."""
    df = raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    ts = pd.to_datetime(df["trans_date_trans_time"])
    amount = df["amt"].astype(float)

    hour = ts.dt.hour
    dow = ts.dt.dayofweek

    # Distance from home using haversine-ish Euclidean on lat/long.
    dist = np.sqrt((df["lat"] - df["merch_lat"]) ** 2 + (df["long"] - df["merch_long"]) ** 2)

    out = pd.DataFrame(
        {
            schema.TRANSACTION_ID: df.get("trans_num", pd.RangeIndex(len(df)).astype(str)),
            schema.CARD_ID: df["cc_num"].astype(str),
            schema.MERCHANT_ID: df["merchant"].astype(str),
            schema.TIMESTAMP: ts,
            "amount": amount,
            "amount_log": np.log1p(amount),
            "hour": hour,
            "day_of_week": dow,
            "is_night": ((hour < 6) | (hour >= 22)).astype(int),
            "is_weekend": (dow >= 5).astype(int),
            "card_age_days": 0.0,  # not available in Sparkov.
            "txn_count_1h": 0,
            "txn_count_24h": 0,
            "amount_mean_24h": amount,
            "amount_to_mean_ratio": 1.0,
            "distance_from_home": dist,
            "merchant_risk": 0.5,
            "category": df["category"].astype(str),
            "channel": "online",
            "device_type": "web",
            schema.LABEL: df["is_fraud"].astype(int),
            schema.LABEL_TIMESTAMP: ts,
        }
    )
    return out.sort_values(schema.TIMESTAMP).reset_index(drop=True)
