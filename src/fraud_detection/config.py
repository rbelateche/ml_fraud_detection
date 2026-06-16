"""Centralised, typed configuration.

All tunables live here so notebooks, scripts and services share one source of
truth. Values can be overridden via environment variables (prefix ``FRAUD_``)
or a ``.env`` file, which keeps secrets out of the codebase.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root = three parents up from this file (src/fraud_detection/config.py).
ROOT_DIR = Path(__file__).resolve().parents[2]


class Paths(BaseSettings):
    """Filesystem layout. Directories are created on demand by callers."""

    root: Path = ROOT_DIR
    data: Path = ROOT_DIR / "data"
    raw: Path = ROOT_DIR / "data" / "raw"
    processed: Path = ROOT_DIR / "data" / "processed"
    artifacts: Path = ROOT_DIR / "artifacts"
    figures: Path = ROOT_DIR / "reports" / "figures"
    mlruns: Path = ROOT_DIR / "mlruns"

    def ensure(self) -> Paths:
        for p in (self.data, self.raw, self.processed, self.artifacts, self.figures):
            p.mkdir(parents=True, exist_ok=True)
        return self


class CostModel(BaseSettings):
    """Business cost matrix used for cost-based threshold tuning.

    The asymmetry is the whole point of fraud detection:
    - A false negative (missed fraud) costs the transaction amount.
    - A false positive (blocking a good customer) has a fixed friction cost.
    """

    model_config = SettingsConfigDict(env_prefix="FRAUD_COST_")

    # Cost of blocking a legitimate transaction (customer friction, support, churn risk).
    false_positive_cost: float = 5.0
    # Fixed cost component of a missed fraud, on top of the lost amount.
    false_negative_fixed_cost: float = 0.0
    # Fraction of the transaction amount lost when fraud is missed.
    false_negative_amount_fraction: float = 1.0
    # Operational cost of reviewing/clearing a transaction flagged for manual review.
    review_cost: float = 1.0


class Settings(BaseSettings):
    """Top-level settings object."""

    model_config = SettingsConfigDict(
        env_prefix="FRAUD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Reproducibility.
    random_seed: int = 42

    # Data source backend: "synthetic" (default, no creds) or "kaggle".
    data_source: str = "synthetic"

    # Synthetic generator knobs.
    synth_n_transactions: int = 200_000
    synth_fraud_rate: float = 0.012
    synth_n_cards: int = 12_000
    synth_n_merchants: int = 800
    synth_start_date: str = "2023-01-01"
    synth_days: int = 90

    # Kaggle backend.
    kaggle_dataset: str = "kartik2112/fraud-detection"  # Sparkov fraudTrain/fraudTest
    kaggle_competition: str | None = None  # e.g. "ieee-fraud-detection"

    # Time-based split fractions (chronological): train / validation / test.
    train_frac: float = 0.6
    valid_frac: float = 0.2  # test_frac is the remainder.

    paths: Paths = Field(default_factory=Paths)
    cost: CostModel = Field(default_factory=CostModel)


# Module-level singleton — import this everywhere.
settings = Settings()
settings.paths.ensure()
