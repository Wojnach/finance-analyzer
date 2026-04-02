"""Train signal weights using historical signal log data.

Reads signal_log.jsonl, extracts per-signal votes and forward returns,
trains a LinearFactorModel via ridge regression, and runs walk-forward
validation to assess weight stability and out-of-sample performance.

Usage:
    from portfolio.train_signal_weights import train_weights
    result = train_weights()  # trains on all available data

    # Or from CLI:
    # .venv/Scripts/python.exe -m portfolio.train_signal_weights
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from portfolio.file_utils import load_jsonl
from portfolio.linear_factor import LinearFactorModel
from portfolio.signal_weight_optimizer import (
    save_results,
    walk_forward_optimize,
)

logger = logging.getLogger("portfolio.train_signal_weights")

_SIGNAL_LOG = Path("data/signal_log.jsonl")
_VOTE_MAP = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}

# Signals to include as features (skip disabled/broken ones)
_SKIP_SIGNALS = {"ml", "funding", "lora"}


def _load_signal_history(
    log_path: Path | None = None,
    horizon: str = "1d",
    min_entries: int = 50,
) -> tuple[pd.DataFrame, pd.Series] | None:
    """Load signal log and extract signal votes + forward returns.

    Args:
        log_path: Path to signal_log.jsonl.
        horizon: Outcome horizon to use for returns ("3h", "1d", "3d").
        min_entries: Minimum entries with outcomes required.

    Returns:
        (signals_df, returns_series) or None if insufficient data.
    """
    log_path = log_path or _SIGNAL_LOG
    entries = load_jsonl(log_path)
    if not entries:
        logger.warning("No signal log entries found at %s", log_path)
        return None

    rows = []
    for entry in entries:
        ts = entry.get("ts")
        tickers = entry.get("tickers", {})
        outcomes = entry.get("outcomes", {})

        for ticker, tdata in tickers.items():
            signals = tdata.get("signals", {})
            # Get outcome for this horizon
            ticker_outcomes = outcomes.get(ticker, {})
            outcome = ticker_outcomes.get(horizon)
            if outcome is None:
                continue

            change_pct = outcome if isinstance(outcome, (int, float)) else outcome.get("change_pct")
            if change_pct is None:
                continue

            # Convert votes to numeric
            row = {"ts": ts, "ticker": ticker, "return": float(change_pct) / 100.0}
            for sig_name, vote in signals.items():
                if sig_name in _SKIP_SIGNALS:
                    continue
                row[sig_name] = _VOTE_MAP.get(vote, 0.0)
            rows.append(row)

    if len(rows) < min_entries:
        logger.warning("Insufficient entries with outcomes: %d < %d", len(rows), min_entries)
        return None

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts").sort_index()

    # Separate returns from signals
    returns = df["return"]
    signal_cols = [c for c in df.columns if c not in ("return", "ticker")]
    signals_df = df[signal_cols].fillna(0.0)

    return signals_df, returns


def train_weights(
    horizon: str = "1d",
    alpha: float = 1.0,
    log_path: Path | None = None,
) -> dict:
    """Train linear factor model and run walk-forward validation.

    Args:
        horizon: Forward return horizon ("3h", "1d", "3d").
        alpha: Ridge regularization strength.
        log_path: Path to signal_log.jsonl.

    Returns:
        Dict with model stats, walk-forward results, and feature rankings.
        Empty dict on failure.
    """
    data = _load_signal_history(log_path, horizon=horizon)
    if data is None:
        return {}

    signals_df, returns = data
    logger.info("Training on %d samples, %d signals, horizon=%s",
               len(signals_df), len(signals_df.columns), horizon)

    # Train full model
    model = LinearFactorModel(alpha=alpha)
    if not model.fit(signals_df, returns):
        logger.warning("Model training failed")
        return {}

    model.save()
    logger.info("Model saved: R²=%.4f, %d features", model.r_squared, len(model.weights))

    # Run walk-forward validation
    wf_result = walk_forward_optimize(
        signals_df, returns,
        train_window=min(720, len(signals_df) // 3),
        test_window=min(168, len(signals_df) // 6),
        step_size=min(168, len(signals_df) // 6),
        alpha=alpha,
    )

    if wf_result.n_windows > 0:
        save_results(wf_result)
        logger.info("Walk-forward: %d windows, OOS corr=%.4f",
                    wf_result.n_windows, wf_result.avg_oos_corr)

    return {
        "model": {
            "r_squared": model.r_squared,
            "n_samples": model.n_samples,
            "n_features": len(model.weights),
            "top_features": model.feature_importance()[:10],
        },
        "walk_forward": wf_result.to_dict() if wf_result.n_windows > 0 else {},
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = train_weights()
    if result:
        print(f"\nModel: R²={result['model']['r_squared']:.4f}, "
              f"{result['model']['n_features']} features, "
              f"{result['model']['n_samples']} samples")
        print("\nTop features:")
        for name, weight in result["model"]["top_features"]:
            print(f"  {name:25s} β={weight:+.6f}")
        if result.get("walk_forward"):
            wf = result["walk_forward"]
            print(f"\nWalk-forward: {wf['n_windows']} windows, "
                  f"R²={wf['avg_r_squared']:.4f}, "
                  f"OOS corr={wf['avg_oos_corr']:.4f}")
    else:
        print("Training failed — insufficient data")
