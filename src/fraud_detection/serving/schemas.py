"""Request/response models for the serving API.

The :class:`Transaction` request mirrors ``schema.feature_columns()`` exactly —
a test asserts the field set matches, so the API contract can never silently
drift from the model's expected inputs (training/serving skew guard).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

_TXN_EXAMPLE = {
    "amount": 249.99,
    "amount_log": 5.525,
    "hour": 23,
    "day_of_week": 5,
    "is_night": 1,
    "is_weekend": 1,
    "card_age_days": 412.0,
    "txn_count_1h": 3,
    "txn_count_24h": 8,
    "amount_mean_24h": 86.40,
    "amount_to_mean_ratio": 2.894,
    "distance_from_home": 57.21,
    "merchant_risk": 0.82,
    "category": "electronics",
    "channel": "online",
    "device_type": "web",
}


class Transaction(BaseModel):
    """A single transaction's model features (mirrors the canonical schema)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": _TXN_EXAMPLE})

    # --- Numeric features ---
    amount: float = Field(..., ge=0, description="Transaction amount.")
    amount_log: float = Field(..., description="log1p(amount).")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23).")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon).")
    is_night: int = Field(..., ge=0, le=1, description="1 if 22:00-06:00.")
    is_weekend: int = Field(..., ge=0, le=1, description="1 if Sat/Sun.")
    card_age_days: float = Field(..., ge=0, description="Days since the card was issued.")
    txn_count_1h: int = Field(..., ge=0, description="Card's transactions in the last hour.")
    txn_count_24h: int = Field(..., ge=0, description="Card's transactions in the last 24h.")
    amount_mean_24h: float = Field(..., ge=0, description="Card's mean amount over 24h.")
    amount_to_mean_ratio: float = Field(..., ge=0, description="amount / amount_mean_24h.")
    distance_from_home: float = Field(..., ge=0, description="Distance from the card's home.")
    merchant_risk: float = Field(..., ge=0, le=1, description="Latent merchant risk (0-1).")

    # --- Categorical features ---
    category: str = Field(..., description="Merchant category.")
    channel: str = Field(..., description="Payment channel (chip/swipe/online/contactless).")
    device_type: str = Field(..., description="Device type (pos/mobile/web/atm).")


class ScoreResponse(BaseModel):
    """The model's decision for one transaction."""

    probability: float = Field(..., description="Calibrated fraud probability (0-1).")
    is_fraud: bool = Field(..., description="True if probability >= threshold.")
    decision: str = Field(..., description="'block' or 'allow'.")
    threshold: float = Field(..., description="Cost-based operating threshold.")
    model_name: str = Field(..., description="Name of the serving model.")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None


class ModelInfo(BaseModel):
    model_name: str
    threshold: float
    n_features: int
    feature_columns: list[str]
