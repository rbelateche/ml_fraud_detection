"""Self-contained synthetic transaction generator.

Produces a realistic, *timestamped* stream of card transactions with a
configurable fraud rate and genuine learnable signal (so the model bake-off in
Phase 0.5 is meaningful). No external data or credentials required, which keeps
the repo runnable in one command and reproducible in CI.

The generated frame conforms to :mod:`fraud_detection.data.schema`.

Design notes
------------
- Fraud is injected with feature-dependent probability (amount, night-time,
  distance from home, merchant risk, velocity) so trees/LR can learn it — but
  with enough noise that the problem is non-trivial and metrics are realistic.
- Per-card rolling aggregates (velocity, mean amount) are computed causally,
  i.e. only from each card's past, mirroring how an online feature store works.
- A ``label_timestamp`` column simulates chargebacks arriving days later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection.config import settings
from fraud_detection.data import schema
from fraud_detection.logging_utils import get_logger

log = get_logger(__name__)

_CATEGORIES = [
    "grocery",
    "restaurant",
    "fuel",
    "electronics",
    "travel",
    "entertainment",
    "health",
    "online_retail",
    "cash_advance",
    "misc",
]
_CHANNELS = ["chip", "swipe", "online", "contactless"]
_DEVICE_TYPES = ["pos", "mobile", "web", "atm"]

# Some categories carry more fraud risk; used to build a latent merchant risk.
_CATEGORY_BASE_RISK = {
    "grocery": 0.2,
    "restaurant": 0.25,
    "fuel": 0.3,
    "electronics": 0.7,
    "travel": 0.6,
    "entertainment": 0.4,
    "health": 0.2,
    "online_retail": 0.65,
    "cash_advance": 0.85,
    "misc": 0.5,
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(
    n_transactions: int | None = None,
    fraud_rate: float | None = None,
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate a synthetic transaction dataset.

    Parameters
    ----------
    n_transactions:
        Number of rows. Defaults to ``settings.synth_n_transactions``.
    fraud_rate:
        Target overall fraud prevalence. Defaults to ``settings.synth_fraud_rate``.
    seed:
        RNG seed. Defaults to ``settings.random_seed``.
    """
    n = int(n_transactions or settings.synth_n_transactions)
    target_rate = float(fraud_rate if fraud_rate is not None else settings.synth_fraud_rate)
    rng = np.random.default_rng(seed if seed is not None else settings.random_seed)

    n_cards = settings.synth_n_cards
    n_merchants = settings.synth_n_merchants

    log.info(
        "Generating %d synthetic transactions (target fraud rate %.3f%%, %d cards, %d merchants)",
        n,
        target_rate * 100,
        n_cards,
        n_merchants,
    )

    # --- Timestamps: spread across the configured window, sorted ascending. ---
    start = pd.Timestamp(settings.synth_start_date)
    span_seconds = settings.synth_days * 24 * 3600
    offsets = np.sort(rng.uniform(0, span_seconds, size=n))
    timestamps = start + pd.to_timedelta(offsets, unit="s")

    # --- Entities. ---
    card_idx = rng.integers(0, n_cards, size=n)
    merchant_idx = rng.integers(0, n_merchants, size=n)
    card_id = np.array([f"card_{i:06d}" for i in card_idx])
    merchant_id = np.array([f"merch_{i:05d}" for i in merchant_idx])

    # Per-card "home location" and signup date for distance / age features.
    card_home = rng.uniform(0, 100, size=(n_cards, 2))
    card_signup_offset = rng.uniform(-720, 0, size=n_cards)  # days before window start

    # Per-merchant latent risk derived from category mix + noise.
    category = rng.choice(_CATEGORIES, size=n, p=_category_probs())
    cat_risk = np.array([_CATEGORY_BASE_RISK[c] for c in category])
    merchant_noise = rng.normal(0, 0.1, size=n_merchants)[merchant_idx]
    merchant_risk = np.clip(cat_risk + merchant_noise, 0.01, 0.99)

    channel = rng.choice(_CHANNELS, size=n, p=[0.35, 0.2, 0.35, 0.10])
    device_type = rng.choice(_DEVICE_TYPES, size=n, p=[0.4, 0.3, 0.25, 0.05])

    # --- Amounts: log-normal, heavier for some categories. ---
    base_mu = np.select(
        [category == "electronics", category == "travel", category == "cash_advance"],
        [4.6, 4.8, 4.4],
        default=3.2,
    )
    amount = np.round(rng.lognormal(mean=base_mu, sigma=0.9), 2)
    amount = np.clip(amount, 1.0, 8000.0)
    amount_log = np.log1p(amount)

    # --- Time-derived features. ---
    hour = timestamps.hour.to_numpy()
    day_of_week = timestamps.dayofweek.to_numpy()
    is_night = ((hour < 6) | (hour >= 22)).astype(int)
    is_weekend = (day_of_week >= 5).astype(int)

    # --- Card age. ---
    txn_day = (timestamps - start) / pd.Timedelta(days=1)
    card_age_days = (txn_day.to_numpy() - card_signup_offset[card_idx]).clip(min=0)

    # --- Distance from home (Euclidean in synthetic 2D geo space). ---
    txn_loc = rng.uniform(0, 100, size=(n, 2))
    distance_from_home = np.linalg.norm(txn_loc - card_home[card_idx], axis=1)

    # --- Causal per-card velocity & amount aggregates. ---
    txn_count_1h, txn_count_24h, amount_mean_24h = _rolling_card_features(
        card_idx, offsets, amount
    )
    amount_to_mean_ratio = amount / np.maximum(amount_mean_24h, 1.0)

    # --- Fraud propensity: latent score -> probability, calibrated to target rate. ---
    z = (
        0.9 * (amount_log - amount_log.mean()) / amount_log.std()
        + 1.1 * (merchant_risk - 0.5)
        + 0.8 * is_night
        + 0.5 * (distance_from_home - distance_from_home.mean()) / distance_from_home.std()
        + 0.6 * np.clip(amount_to_mean_ratio - 1.0, 0, 5) / 5.0
        + 0.4 * np.clip(txn_count_1h - 1, 0, 10) / 10.0
        + rng.normal(0, 1.0, size=n)  # irreducible noise
    )
    # Shift intercept so realised prevalence ~ target_rate.
    intercept = _solve_intercept(z, target_rate)
    fraud_prob = _sigmoid(intercept + z)
    is_fraud = (rng.uniform(0, 1, size=n) < fraud_prob).astype(int)

    # --- Delayed labels: chargebacks land 3-45 days after the transaction. ---
    label_delay_days = np.where(
        is_fraud == 1,
        rng.uniform(3, 45, size=n),
        rng.uniform(0, 1, size=n),  # legit confirmed quickly
    )
    label_timestamp = timestamps + pd.to_timedelta(label_delay_days, unit="D")

    df = pd.DataFrame(
        {
            schema.TRANSACTION_ID: [f"txn_{i:08d}" for i in range(n)],
            schema.CARD_ID: card_id,
            schema.MERCHANT_ID: merchant_id,
            schema.TIMESTAMP: timestamps,
            "amount": amount,
            "amount_log": amount_log,
            "hour": hour,
            "day_of_week": day_of_week,
            "is_night": is_night,
            "is_weekend": is_weekend,
            "card_age_days": np.round(card_age_days, 2),
            "txn_count_1h": txn_count_1h,
            "txn_count_24h": txn_count_24h,
            "amount_mean_24h": np.round(amount_mean_24h, 2),
            "amount_to_mean_ratio": np.round(amount_to_mean_ratio, 3),
            "distance_from_home": np.round(distance_from_home, 3),
            "merchant_risk": np.round(merchant_risk, 3),
            "category": category,
            "channel": channel,
            "device_type": device_type,
            schema.LABEL: is_fraud,
            schema.LABEL_TIMESTAMP: label_timestamp,
        }
    )

    realised = df[schema.LABEL].mean()
    log.info("Generated %d rows; realised fraud rate %.3f%%", len(df), realised * 100)
    return df


def _category_probs() -> list[float]:
    # Skewed category mix (grocery/restaurant most common).
    weights = np.array([0.22, 0.18, 0.14, 0.05, 0.04, 0.07, 0.08, 0.10, 0.02, 0.10])
    return list(weights / weights.sum())


def _rolling_card_features(
    card_idx: np.ndarray, offsets: np.ndarray, amount: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal per-card velocity and 24h mean amount.

    For each transaction, counts the same card's transactions in the preceding
    1h and 24h windows, and the mean amount over the past 24h. Rows are already
    globally time-sorted, so per-card order is preserved.
    """
    n = len(card_idx)
    txn_count_1h = np.zeros(n, dtype=int)
    txn_count_24h = np.zeros(n, dtype=int)
    amount_mean_24h = np.zeros(n, dtype=float)

    h1 = 3600.0
    h24 = 24 * 3600.0
    history: dict[int, list[tuple[float, float]]] = {}

    for i in range(n):
        c = int(card_idx[i])
        t = offsets[i]
        hist = history.get(c)
        if hist is None:
            hist = []
            history[c] = hist

        cnt1 = cnt24 = 0
        sum24 = 0.0
        # Walk backwards over recent history (lists are time-ordered per card).
        for past_t, past_amt in reversed(hist):
            dt = t - past_t
            if dt <= h24:
                cnt24 += 1
                sum24 += past_amt
                if dt <= h1:
                    cnt1 += 1
            else:
                break  # older than 24h; everything before is older too.

        txn_count_1h[i] = cnt1
        txn_count_24h[i] = cnt24
        amount_mean_24h[i] = (sum24 / cnt24) if cnt24 > 0 else amount[i]
        hist.append((t, float(amount[i])))

    return txn_count_1h, txn_count_24h, amount_mean_24h


def _solve_intercept(z: np.ndarray, target_rate: float, iters: int = 60) -> float:
    """Bisection search for the intercept that yields the target mean fraud rate."""
    lo, hi = -20.0, 20.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        rate = _sigmoid(mid + z).mean()
        if rate > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2
