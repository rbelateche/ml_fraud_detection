"""``fraud-drift`` — compute drift between a reference and a current window.

Splits the canonical dataset chronologically into an older **reference** half and
a newer **current** half, computes PSI over every feature (and, when a trained
model bundle is available, over the model's score distribution), prints a report
and writes it as JSON. With ``--inject-drift`` it perturbs the current window so
you can *see* the detector fire — a self-contained demo of the alert path.

Exit code is ``1`` when a *major* drift alert fires, so it can gate a pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_detection.config import settings
from fraud_detection.data import schema
from fraud_detection.data.loader import load_dataset
from fraud_detection.logging_utils import get_logger
from fraud_detection.monitoring.report import compute_drift

log = get_logger(__name__)


def _split_reference_current(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological 50/50 split: older half = reference, newer half = current."""
    if schema.TIMESTAMP in df.columns:
        df = df.sort_values(schema.TIMESTAMP)
    mid = len(df) // 2
    reference = df.iloc[:mid].reset_index(drop=True)
    current = df.iloc[mid:].reset_index(drop=True)
    return reference, current


def inject_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Perturb a window so PSI clearly fires — a deterministic demo of drift."""
    out = df.copy()
    if "amount" in out:
        out["amount"] = out["amount"] * 1.8 + 25.0
        if "amount_log" in out:
            out["amount_log"] = np.log1p(out["amount"])
    if "hour" in out:
        out["hour"] = (out["hour"] + 7) % 24
        if "is_night" in out:
            out["is_night"] = ((out["hour"] < 6) | (out["hour"] >= 22)).astype(int)
    if "merchant_risk" in out:
        out["merchant_risk"] = np.clip(out["merchant_risk"] + 0.2, 0.0, 1.0)
    if "channel" in out:
        # Shove most traffic onto a single channel to move the categorical mix.
        out["channel"] = "online"
    return out


def _maybe_scores(reference: pd.DataFrame, current: pd.DataFrame):
    """Score both windows with the trained bundle, if one exists on disk."""
    from fraud_detection.serving.bundle import ModelBundle, default_model_path

    path = default_model_path()
    if not Path(path).exists():
        log.info("No model bundle at %s — skipping score drift.", path)
        return None, None
    bundle = ModelBundle.load(path)
    return bundle.predict_proba(reference), bundle.predict_proba(current)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute feature + score drift (PSI).")
    parser.add_argument(
        "--inject-drift",
        action="store_true",
        help="Perturb the current window to demonstrate the alert path.",
    )
    parser.add_argument("--bins", type=int, default=10, help="Numeric PSI bin count.")
    parser.add_argument(
        "--no-scores",
        action="store_true",
        help="Skip model score drift even if a bundle is available.",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Where to write the JSON report (default: artifacts/drift_report.json).",
    )
    parser.add_argument(
        "--exit-zero",
        action="store_true",
        help="Always exit 0 on a clean run, even when an alert fires (CI/demo).",
    )
    args = parser.parse_args(argv)

    df = load_dataset()
    reference, current = _split_reference_current(df)
    if args.inject_drift:
        current = inject_drift(current)
        log.info("Injected synthetic drift into the current window.")

    scores_ref = scores_cur = None
    if not args.no_scores:
        try:
            scores_ref, scores_cur = _maybe_scores(reference, current)
        except Exception as exc:  # noqa: BLE001 - score drift is best-effort.
            log.warning("Could not compute score drift: %s", exc)

    report = compute_drift(
        reference,
        current,
        bins=args.bins,
        scores_reference=scores_ref,
        scores_current=scores_cur,
    )

    print(report.summary())

    out_path = Path(args.json) if args.json else settings.paths.artifacts / "drift_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.as_dict(), indent=2))
    log.info("Wrote drift report to %s", out_path)

    if args.exit_zero:
        return 0
    return 1 if report.alert else 0


if __name__ == "__main__":
    sys.exit(main())
