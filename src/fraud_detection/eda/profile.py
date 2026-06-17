"""EDA profiler (``fraud-eda``).

Generates a compact, headless set of figures and a text summary that surface the
defining traits of fraud data: extreme class imbalance, amount distributions by
class, and temporal patterns. Figures are written to ``reports/figures``.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")  # headless — works in CI / containers.
import matplotlib.pyplot as plt  # noqa: E402

from fraud_detection.config import settings  # noqa: E402
from fraud_detection.data import schema  # noqa: E402
from fraud_detection.data.loader import load_dataset  # noqa: E402
from fraud_detection.logging_utils import get_logger  # noqa: E402

log = get_logger(__name__)


def profile(source: str | None = None) -> dict:
    """Compute summary stats and write EDA figures. Returns the summary dict."""
    df = load_dataset(source=source)
    fig_dir = settings.paths.figures
    fig_dir.mkdir(parents=True, exist_ok=True)

    fraud_rate = float(df[schema.LABEL].mean())
    summary = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "fraud_rate_%": round(fraud_rate * 100, 4),
        "fraud_count": int(df[schema.LABEL].sum()),
        "date_min": str(df[schema.TIMESTAMP].min()),
        "date_max": str(df[schema.TIMESTAMP].max()),
        "amount_median_legit": round(
            float(df.loc[df[schema.LABEL] == 0, "amount"].median()), 2
        ),
        "amount_median_fraud": round(
            float(df.loc[df[schema.LABEL] == 1, "amount"].median()), 2
        ),
    }

    # 1) Class balance.
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[schema.LABEL].value_counts().sort_index()
    ax.bar(["legit", "fraud"], counts.values, color=["#4c72b0", "#c44e52"])
    ax.set_yscale("log")
    ax.set_title(f"Class balance (fraud = {fraud_rate * 100:.2f}%)")
    ax.set_ylabel("count (log scale)")
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_class_balance.png", dpi=120)
    plt.close(fig)

    # 2) Amount distribution by class.
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, color in [(0, "#4c72b0"), (1, "#c44e52")]:
        vals = df.loc[df[schema.LABEL] == label, "amount_log"]
        ax.hist(vals, bins=60, alpha=0.6, density=True, color=color,
                label="fraud" if label else "legit")
    ax.set_title("log(amount) distribution by class")
    ax.set_xlabel("log1p(amount)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_amount_by_class.png", dpi=120)
    plt.close(fig)

    # 3) Fraud rate by hour of day.
    fig, ax = plt.subplots(figsize=(7, 4))
    by_hour = df.groupby("hour")[schema.LABEL].mean() * 100
    ax.plot(by_hour.index, by_hour.values, marker="o", color="#c44e52")
    ax.set_title("Fraud rate by hour of day")
    ax.set_xlabel("hour")
    ax.set_ylabel("fraud rate (%)")
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_fraud_by_hour.png", dpi=120)
    plt.close(fig)

    log.info("EDA figures written to %s", fig_dir)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Profile the fraud dataset.")
    parser.add_argument("--source", default=None)
    args = parser.parse_args(argv)

    summary = profile(source=args.source)
    print("\nEDA summary:")
    for k, v in summary.items():
        print(f"  {k:22s}: {v}")
    print(f"\nFigures saved under: {settings.paths.figures}")


if __name__ == "__main__":
    main()
