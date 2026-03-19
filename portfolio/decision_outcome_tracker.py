"""Backfill outcomes for Layer 2 autonomous decisions.

Reads data/layer2_decisions.jsonl, checks if enough time has elapsed for
each prediction horizon, fetches historical prices, and writes outcome
records to data/layer2_decision_outcomes.jsonl.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

from portfolio.file_utils import atomic_append_jsonl, load_jsonl

logger = logging.getLogger("portfolio.decision_outcome_tracker")

BASE_DIR = Path(__file__).resolve().parent.parent
DECISIONS_FILE = BASE_DIR / "data" / "layer2_decisions.jsonl"
OUTCOMES_FILE = BASE_DIR / "data" / "layer2_decision_outcomes.jsonl"

HORIZONS = {"1d": 86400, "3d": 259200}


def backfill_decision_outcomes(max_entries: int = 500) -> int:
    """Backfill outcomes for layer2 decisions. Returns count of new outcomes."""
    from portfolio.outcome_tracker import _fetch_historical_price

    decisions = load_jsonl(DECISIONS_FILE)
    if not decisions:
        return 0

    # Load existing outcomes to avoid duplicates
    existing: set[tuple[str | None, str | None, str | None]] = set()
    existing_outcomes = load_jsonl(OUTCOMES_FILE)
    for o in existing_outcomes or []:
        existing.add((o.get("decision_ts"), o.get("ticker"), o.get("horizon")))

    now = dt.datetime.now(dt.UTC)
    new_count = 0

    # Process most recent entries first (limited to max_entries)
    for decision in decisions[-max_entries:]:
        ts_str = decision.get("ts")
        if not ts_str:
            continue
        try:
            decision_ts = dt.datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            continue

        predictions = decision.get("predictions") or {}
        prices = decision.get("prices") or {}

        for ticker, pred in predictions.items():
            outlook = pred.get("outlook")
            if outlook == "neutral":
                continue  # nothing to score

            base_price = prices.get(ticker)
            if not base_price:
                continue

            for horizon_name, horizon_secs in HORIZONS.items():
                if (ts_str, ticker, horizon_name) in existing:
                    continue

                target_dt = decision_ts + dt.timedelta(seconds=horizon_secs)
                if now < target_dt:
                    continue  # not enough time elapsed

                # _fetch_historical_price expects a Unix timestamp (float)
                target_ts = target_dt.timestamp()
                try:
                    hist_price = _fetch_historical_price(ticker, target_ts)
                except Exception:
                    continue
                if not hist_price:
                    continue

                change_pct = ((hist_price - base_price) / base_price) * 100
                correct = (outlook == "bullish" and change_pct > 0) or \
                          (outlook == "bearish" and change_pct < 0)

                outcome = {
                    "decision_ts": ts_str,
                    "ticker": ticker,
                    "horizon": horizon_name,
                    "outlook": outlook,
                    "conviction": pred.get("conviction", 0),
                    "recommendation": pred.get("recommendation"),
                    "base_price_usd": base_price,
                    "outcome_price_usd": round(hist_price, 4),
                    "change_pct": round(change_pct, 4),
                    "correct": correct,
                    "resolved_at": now.isoformat(),
                    "regime": decision.get("regime"),
                }
                atomic_append_jsonl(OUTCOMES_FILE, outcome)
                existing.add((ts_str, ticker, horizon_name))
                new_count += 1

    if new_count:
        logger.info("Backfilled %d Layer 2 decision outcomes", new_count)
    return new_count
