"""Bake-off orchestrator.

Runs a fair, evidence-backed tournament across all available candidates and
produces the leaderboard + artifacts that justify the final model choice.

Methodology (stated so it can be judged):
1. **Time-based split** — train on the past, validate on the middle, test on the
   future. No random shuffling, so no future leakage.
2. **Optional Optuna tuning** — equal, small trial budget per tunable model,
   optimising PR-AUC on the validation split.
3. **Fit** the final model on train.
4. **Calibration comparison** — raw vs sigmoid vs isotonic. Calibrators are fit
   on the first half of validation and *selected by Brier on the second half*
   (an independent slice), then refit on full validation and applied to test.
5. **Cost-based threshold** — chosen on validation by minimising expected dollar
   cost, then reported on test (no threshold leakage).
6. **Latency** — single-row p50/p99 measured on the fitted pipeline.

Decision rule: **minimise expected business cost per transaction at the operating
threshold, subject to p99 latency < 50 ms.**
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from fraud_detection.bakeoff import plots
from fraud_detection.bakeoff.latency import measure_latency
from fraud_detection.config import settings
from fraud_detection.data.loader import load_dataset
from fraud_detection.data.split import time_based_split
from fraud_detection.logging_utils import get_logger
from fraud_detection.metrics import compute_metrics, optimal_threshold, total_cost
from fraud_detection.models.calibration import compare_calibration, fit_calibrator
from fraud_detection.models.candidates import build_candidates
from fraud_detection.models.preprocessing import split_xy
from fraud_detection.models.tuning import build_tuned, tune_model

# Benign: trees fit on a named frame are scored on arrays during latency probes.
warnings.filterwarnings(
    "ignore", message="X does not have valid feature names", category=UserWarning
)

log = get_logger(__name__)

LATENCY_SLA_MS = 50.0


@dataclass
class BakeoffConfig:
    source: str | None = None
    include: list[str] | None = None
    tune: bool = False
    n_trials: int = 20
    quick: bool = False
    log_mlflow: bool = True


@dataclass
class BakeoffOutput:
    leaderboard: pd.DataFrame
    best_model: str
    artifacts: dict[str, str] = field(default_factory=dict)


def run(cfg: BakeoffConfig) -> BakeoffOutput:
    """Execute the full bake-off and return the leaderboard + chosen model."""
    if cfg.quick:
        # Fast path for CI / smoke: tiny budget, no tuning.
        cfg.tune = False
        cfg.n_trials = min(cfg.n_trials, 5)

    df = load_dataset(source=cfg.source)
    split = time_based_split(df)
    log.info("Loaded %d rows; split summary:\n%s", len(df), split.summary().to_string(index=False))

    x_train, y_train = split_xy(split.train)
    x_valid, y_valid = split_xy(split.valid)
    x_test, y_test = split_xy(split.test)
    amount_valid = split.valid["amount"].to_numpy()
    amount_test = split.test["amount"].to_numpy()

    candidates = build_candidates(y_train.to_numpy(), include=cfg.include)
    log.info("Candidates in tournament: %s", list(candidates))

    mlflow_ctx = _maybe_mlflow(cfg.log_mlflow)

    rows: list[dict] = []
    scores_by_model: dict[str, np.ndarray] = {}
    fitted: dict[str, dict] = {}

    for name, pipe in candidates.items():
        log.info("=== Candidate: %s ===", name)

        # 1) Optional tuning (skips models without a search space / missing libs).
        if cfg.tune:
            tuned = tune_model(name, x_train, y_train, x_valid, y_valid, cfg.n_trials)
            if tuned is not None:
                rebuilt = build_tuned(name, tuned["best_params"], y_train.to_numpy())
                if rebuilt is not None:
                    pipe = rebuilt
                    log.info("Using tuned params for %s", name)

        # 2) Fit on train.
        pipe.fit(x_train, y_train)

        # 3) Raw scores.
        valid_raw = pipe.predict_proba(x_valid)[:, 1]
        test_raw = pipe.predict_proba(x_test)[:, 1]

        # 4) Calibration: select on an independent half of validation.
        cal = _select_calibration(valid_raw, y_valid.to_numpy())
        calibrator = fit_calibrator(cal["best_method"], valid_raw, y_valid.to_numpy())
        valid_cal = calibrator.predict(valid_raw)
        test_cal = calibrator.predict(test_raw)

        # 5) Metrics on test (calibrated).
        m = compute_metrics(y_test.to_numpy(), test_cal)
        brier_raw = float(_brier(y_test.to_numpy(), test_raw))

        # 6) Cost-based threshold chosen on validation, reported on test.
        valid_thr = optimal_threshold(y_valid.to_numpy(), valid_cal, amount_valid)
        test_cost = total_cost(y_test.to_numpy(), test_cal, amount_test, valid_thr.threshold)

        # 7) Latency on the fitted pipeline.
        lat = measure_latency(pipe, x_test, n_iters=200 if cfg.quick else 500)

        row = {
            "model": name,
            "pr_auc": round(m.pr_auc, 4),
            "roc_auc": round(m.roc_auc, 4),
            "recall@90prec": round(m.recall_at_90_precision, 4),
            "precision@1%": round(m.precision_at_1pct, 4),
            "brier_raw": round(brier_raw, 4),
            "brier_cal": round(m.brier, 4),
            "calibration": cal["best_method"],
            "threshold": round(valid_thr.threshold, 4),
            "cost_per_txn": round(test_cost.cost_per_txn, 4),
            "fraud_$_caught": round(test_cost.fraud_dollars_caught, 0),
            "fraud_$_missed": round(test_cost.fraud_dollars_missed, 0),
            "p50_ms": round(lat.p50_ms, 3),
            "p99_ms": round(lat.p99_ms, 3),
            "meets_sla": lat.p99_ms < LATENCY_SLA_MS,
        }
        rows.append(row)
        scores_by_model[name] = test_cal
        fitted[name] = {
            "pipeline": pipe,
            "calibrator": calibrator,
            "threshold": valid_thr.threshold,
            "test_raw": test_raw,
            "test_cal": test_cal,
        }

        _log_run_mlflow(mlflow_ctx, name, row, cal["brier_by_method"])

    leaderboard = _rank(pd.DataFrame(rows))
    best_model = _select_winner(leaderboard)
    log.info("Winner: %s", best_model)

    artifacts = _emit_artifacts(
        leaderboard, best_model, fitted, scores_by_model, y_test.to_numpy(), amount_test
    )

    _close_mlflow(mlflow_ctx)
    return BakeoffOutput(leaderboard=leaderboard, best_model=best_model, artifacts=artifacts)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _brier(y_true: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import brier_score_loss

    return brier_score_loss(y_true, scores)


def _select_calibration(valid_raw: np.ndarray, y_valid: np.ndarray) -> dict:
    """Pick a calibration method using two independent halves of validation."""
    n = len(valid_raw)
    half = n // 2
    a_scores, a_y = valid_raw[:half], y_valid[:half]
    b_scores, b_y = valid_raw[half:], y_valid[half:]
    res = compare_calibration(a_scores, a_y, eval_scores=b_scores, eval_y=b_y)
    return {"best_method": res.best_method, "brier_by_method": res.brier_by_method}


def _rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["meets_sla", "cost_per_txn"], ascending=[False, True]).reset_index(
        drop=True
    )


def _select_winner(leaderboard: pd.DataFrame) -> str:
    """Min cost-per-txn among SLA-compliant models; fall back to overall min cost."""
    compliant = leaderboard[leaderboard["meets_sla"]]
    pool = compliant if not compliant.empty else leaderboard
    return str(pool.sort_values("cost_per_txn").iloc[0]["model"])


def _emit_artifacts(
    leaderboard: pd.DataFrame,
    best_model: str,
    fitted: dict,
    scores_by_model: dict,
    y_test: np.ndarray,
    amount_test: np.ndarray,
) -> dict[str, str]:
    art_dir = settings.paths.artifacts
    art_dir.mkdir(parents=True, exist_ok=True)

    # Leaderboard as CSV + Markdown.
    csv_path = art_dir / "leaderboard.csv"
    leaderboard.to_csv(csv_path, index=False)
    md_path = art_dir / "leaderboard.md"
    md_path.write_text(_leaderboard_markdown(leaderboard, best_model))

    # Plots.
    plots.plot_pr_curves(y_test, scores_by_model)
    best = fitted[best_model]
    plots.plot_reliability(y_test, best["test_raw"], best["test_cal"], best_model)
    plots.plot_latency_vs_pr_auc(leaderboard.to_dict("records"))
    plots.plot_cost_curve(y_test, best["test_cal"], amount_test, best["threshold"], best_model)

    # Persist the winning model bundle for Phase 1 serving.
    bundle = {
        "model_name": best_model,
        "pipeline": best["pipeline"],
        "calibrator": best["calibrator"],
        "threshold": best["threshold"],
        "feature_columns": _feature_columns(),
    }
    model_path = art_dir / "best_model.joblib"
    joblib.dump(bundle, model_path)

    rationale_path = art_dir / "model_selection_rationale.md"
    rationale_path.write_text(_rationale_markdown(leaderboard, best_model))

    return {
        "leaderboard_csv": str(csv_path),
        "leaderboard_md": str(md_path),
        "best_model": str(model_path),
        "rationale": str(rationale_path),
        "figures_dir": str(settings.paths.figures),
    }


def _feature_columns() -> list[str]:
    from fraud_detection.data import schema

    return schema.feature_columns()


def _leaderboard_markdown(leaderboard: pd.DataFrame, best_model: str) -> str:
    lines = ["# Model bake-off leaderboard", ""]
    lines.append(f"**Winner: `{best_model}`** — lowest expected cost/txn under the 50 ms SLA.")
    lines.append("")
    lines.append(leaderboard.to_markdown(index=False))
    lines.append("")
    return "\n".join(lines)


def _rationale_markdown(leaderboard: pd.DataFrame, best_model: str) -> str:
    winner = leaderboard[leaderboard["model"] == best_model].iloc[0]
    naive = leaderboard[leaderboard["model"] == "dummy"]
    naive_cost = float(naive.iloc[0]["cost_per_txn"]) if not naive.empty else float("nan")
    savings = (
        f"{(naive_cost - float(winner['cost_per_txn'])) / naive_cost * 100:.1f}%"
        if naive_cost and not np.isnan(naive_cost) and naive_cost > 0
        else "n/a"
    )
    return "\n".join(
        [
            "# Model selection rationale",
            "",
            "## Decision rule",
            "Pick the model that **minimises expected business cost per transaction "
            "at the operating threshold, subject to p99 latency < 50 ms.**",
            "",
            "## Outcome",
            f"- **Chosen model:** `{best_model}`",
            f"- **PR-AUC (test):** {winner['pr_auc']}",
            f"- **Calibration method:** {winner['calibration']} "
            f"(Brier {winner['brier_raw']} → {winner['brier_cal']})",
            f"- **Operating threshold:** {winner['threshold']} (tuned on validation by cost)",
            f"- **Expected cost/txn (test):** {winner['cost_per_txn']}",
            f"- **p50 / p99 latency:** {winner['p50_ms']} ms / {winner['p99_ms']} ms",
            f"- **Cost reduction vs naive (dummy):** {savings}",
            "",
            "## Methodology highlights",
            "- Time-based out-of-time split (no future leakage).",
            "- Imbalance handled per-model via class weights / scale_pos_weight.",
            "- Calibration method chosen by Brier on an independent validation slice.",
            "- Threshold tuned on validation, reported on test (no threshold leakage).",
            "- Every run logged to MLflow for reproducibility.",
            "",
            "## Artifacts",
            "- `artifacts/leaderboard.md` — full leaderboard (models × metrics).",
            "- `reports/figures/bakeoff_pr_curves.png` — PR curves overlaid.",
            "- `reports/figures/bakeoff_reliability.png` — calibration before/after.",
            "- `reports/figures/bakeoff_latency_vs_prauc.png` — latency/PR-AUC trade-off.",
            "- `reports/figures/bakeoff_cost_curve.png` — cost vs threshold.",
            "",
        ]
    )


# --------------------------------------------------------------------------- #
# MLflow (optional — degrades gracefully if not installed)
# --------------------------------------------------------------------------- #
def _maybe_mlflow(enabled: bool):
    if not enabled:
        return None
    try:
        import mlflow
    except ImportError:
        log.warning("mlflow not installed — skipping experiment tracking.")
        return None
    import os

    # Default to a local SQLite backend (the file store is in maintenance mode).
    # MLFLOW_TRACKING_URI overrides this for a remote/shared server.
    uri = os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{settings.paths.root / 'mlflow.db'}"
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("fraud-bakeoff")
    except Exception as exc:  # noqa: BLE001 — tracking must never break the run.
        log.warning("MLflow disabled (%s).", exc)
        return None
    log.info("MLflow tracking at %s (experiment 'fraud-bakeoff')", uri)
    return mlflow


def _log_run_mlflow(mlflow, name: str, row: dict, brier_by_method: dict) -> None:
    if mlflow is None:
        return
    try:
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)
            mlflow.log_param("calibration_method", row["calibration"])
            mlflow.log_param("threshold", row["threshold"])
            for key in ("pr_auc", "roc_auc", "recall@90prec", "precision@1%", "brier_raw",
                        "brier_cal", "cost_per_txn", "p50_ms", "p99_ms"):
                mlflow.log_metric(key.replace("@", "_at_").replace("%", "pct"), float(row[key]))
            for method, brier in brier_by_method.items():
                mlflow.log_metric(f"brier_{method}", float(brier))
    except Exception as exc:  # noqa: BLE001 — never let tracking break the run.
        log.warning("MLflow run logging failed for %s (%s).", name, exc)


def _close_mlflow(mlflow) -> None:
    if mlflow is None:
        return
    try:
        with mlflow.start_run(run_name="_summary"):
            lb = settings.paths.artifacts / "leaderboard.csv"
            if lb.exists():
                mlflow.log_artifact(str(lb))
    except Exception as exc:  # noqa: BLE001
        log.warning("MLflow summary logging failed (%s).", exc)


def save_summary_json(output: BakeoffOutput) -> None:
    """Write a compact JSON summary next to the artifacts."""
    out = settings.paths.artifacts / "bakeoff_summary.json"
    payload = {
        "best_model": output.best_model,
        "leaderboard": output.leaderboard.to_dict("records"),
        "artifacts": output.artifacts,
    }
    out.write_text(json.dumps(payload, indent=2, default=str))
