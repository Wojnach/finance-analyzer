"""Outcome backfill for `data/sentiment_ab_log.jsonl` — shadow sentiment models.

Why this is separate from `llm_outcome_backfill.py`
---------------------------------------------------
`llm_probability_log.jsonl` is the unified probability log we added this
session; every LLM signal's vote lands there with a clean schema. It's
going to get outcome-paired by `portfolio.llm_outcome_backfill`.

The A/B sentiment log `data/sentiment_ab_log.jsonl` has a DIFFERENT shape
and existed before the probability log. Each row captures ONE primary
sentiment call plus its shadow model outputs (FinGPT, FinBERT, FinGPT
cumulative) in a `shadow` array. Three years' history is already there.
Mixing it into the probability-log schema would require backfilling or
lossy transformation — cheaper to pair outcomes with the existing schema
in place.

What this writes
----------------
`data/sentiment_shadow_outcomes.jsonl` — one row per (original ts, model)
pair, linking the shadow model's `sentiment` verdict to the market's
realized move at a fixed horizon (default 1d).

Schema:
```json
{"ts": "...", "ticker": "BTC-USD", "model": "fingpt:finance-llama-8b",
 "kind": "shadow",  // or "primary" for the non-shadow sentiment
 "sentiment": "positive", "confidence": 0.69,
 "predicted_class": "BUY",  // mapped from sentiment
 "outcome": "BUY",          // derived from realized pct_change
 "pct_change": 0.012, "entry_price": 100.0, "target_price": 101.2,
 "horizon": "1d",
 "correct": true,           // predicted_class == outcome
 "agreement_with_primary": true,
 "backfilled_at": "..."}
```

Primary rows are included too with `kind="primary"` so we can compute the
primary's own accuracy through the same pipeline — currently measured
separately by `portfolio.accuracy_stats`, but for cross-checking with
shadows it's convenient to have all four model verdicts per row in one
place.

Idempotent: keyed on `(ts, ticker, model, horizon)`.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from portfolio.file_utils import atomic_append_jsonl
from portfolio.forecast_accuracy import _lookup_price_at_time
from portfolio.llm_calibration import outcome_from_return

logger = logging.getLogger("portfolio.sentiment_shadow_backfill")

_BASE_DIR = Path(__file__).resolve().parent.parent
_AB_LOG = _BASE_DIR / "data" / "sentiment_ab_log.jsonl"
_OUTCOMES = _BASE_DIR / "data" / "sentiment_shadow_outcomes.jsonl"

_HORIZON_HOURS = {
    "3h": 3,
    "4h": 4,
    "12h": 12,
    "1d": 24,
    "3d": 72,
    "5d": 120,
    "10d": 240,
}

# Sentiment labels → BUY/HOLD/SELL class.
_SENTIMENT_TO_CLASS = {
    "positive": "BUY",
    "negative": "SELL",
    "neutral": "HOLD",
}

# The A/B log stores sentiment for tickers like "BTC" (short form), but the
# price snapshots key by "BTC-USD" etc. Map here so lookups hit.
_TICKER_EXPAND = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "XAG": "XAG-USD",
    "XAU": "XAU-USD",
    # Stocks pass through unchanged.
}


def _expand_ticker(t: str) -> str:
    """Map short-form A/B log ticker to the price-snapshot key."""
    return _TICKER_EXPAND.get(t, t)


def _load_existing_keys(outcomes_path: Path) -> set[tuple[str, str, str, str]]:
    keys: set[tuple[str, str, str, str]] = set()
    if not outcomes_path.exists():
        return keys
    for line in outcomes_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            keys.add((
                row.get("ts", ""),
                row.get("ticker", ""),
                row.get("model", ""),
                row.get("horizon", ""),
            ))
        except json.JSONDecodeError:
            continue
    return keys


def backfill(
    *,
    ab_log_path: Path | str | None = None,
    outcomes_path: Path | str | None = None,
    snapshot_path: Path | str | None = None,
    horizon: str = "1d",
    min_age_hours: int = 0,
    max_rows: int | None = None,
) -> dict:
    """Read the sentiment A/B log and backfill outcomes.

    Args:
      ab_log_path: override for `sentiment_ab_log.jsonl`.
      outcomes_path: override for `sentiment_shadow_outcomes.jsonl`.
      snapshot_path: override for the hourly price snapshot file.
      horizon: which horizon to evaluate at. Default "1d" matches the
        existing argmax sentiment accuracy horizon.
      min_age_hours: buffer beyond the horizon before we trust the
        snapshot. Default 0.
      max_rows: stop after N rows written (diagnostic).

    Returns stats dict with counts for each skip/write category.
    """
    ab_log_path = Path(ab_log_path) if ab_log_path else _AB_LOG
    outcomes_path = Path(outcomes_path) if outcomes_path else _OUTCOMES

    stats = {
        "rows_read": 0,
        "outcomes_written": 0,
        "skipped_already_present": 0,
        "skipped_too_recent": 0,
        "skipped_missing_price": 0,
        "skipped_bad_row": 0,
    }

    if not ab_log_path.exists():
        logger.info("No A/B log at %s; nothing to backfill", ab_log_path)
        return stats

    hours = _HORIZON_HOURS.get(horizon)
    if hours is None:
        logger.warning("Unknown horizon %s", horizon)
        return stats

    existing = _load_existing_keys(outcomes_path)
    now = datetime.now(UTC)

    for line in ab_log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            stats["skipped_bad_row"] += 1
            continue

        stats["rows_read"] += 1

        ts = row.get("ts", "")
        raw_ticker = row.get("ticker", "")
        if not ts or not raw_ticker:
            stats["skipped_bad_row"] += 1
            continue
        ticker = _expand_ticker(raw_ticker)

        try:
            entry_time = datetime.fromisoformat(ts)
            if entry_time.tzinfo is None:
                entry_time = entry_time.replace(tzinfo=UTC)
        except (TypeError, ValueError):
            stats["skipped_bad_row"] += 1
            continue

        target_time = entry_time + timedelta(hours=hours)
        if now < target_time + timedelta(hours=min_age_hours):
            # Too recent — every model verdict from this row is skipped.
            # Count once per row, not once per model, otherwise the
            # "too recent" bucket gets N× inflated.
            stats["skipped_too_recent"] += 1
            continue

        entry_price = _lookup_price_at_time(ticker, entry_time, snapshot_file=snapshot_path)
        target_price = _lookup_price_at_time(ticker, target_time, snapshot_file=snapshot_path)
        if entry_price is None or target_price is None or entry_price == 0:
            stats["skipped_missing_price"] += 1
            continue

        pct = (target_price - entry_price) / entry_price
        realized_class = outcome_from_return(pct)

        primary = row.get("primary") or {}
        primary_model = primary.get("model", "unknown")
        primary_sentiment = primary.get("sentiment", "neutral")
        primary_class = _SENTIMENT_TO_CLASS.get(primary_sentiment, "HOLD")

        models_to_write: list[tuple[str, dict, str]] = [
            (primary_model, primary, "primary"),
        ]
        for shadow in row.get("shadow") or []:
            sm = shadow.get("model")
            if not sm:
                continue
            models_to_write.append((sm, shadow, "shadow"))

        for model_name, verdict, kind in models_to_write:
            key = (ts, ticker, model_name, horizon)
            if key in existing:
                stats["skipped_already_present"] += 1
                continue
            sentiment = verdict.get("sentiment", "neutral")
            predicted_class = _SENTIMENT_TO_CLASS.get(sentiment, "HOLD")
            outcome_row = {
                "ts": ts,
                "ticker": ticker,
                "model": model_name,
                "kind": kind,
                "sentiment": sentiment,
                "confidence": verdict.get("confidence"),
                "predicted_class": predicted_class,
                "outcome": realized_class,
                "pct_change": pct,
                "entry_price": entry_price,
                "target_price": target_price,
                "horizon": horizon,
                "correct": predicted_class == realized_class,
                "agreement_with_primary": (
                    predicted_class == primary_class if kind == "shadow" else None
                ),
                "backfilled_at": now.isoformat(),
            }
            try:
                atomic_append_jsonl(outcomes_path, outcome_row)
            except Exception as e:
                logger.warning("sentiment outcome append failed: %s", e)
                continue
            existing.add(key)
            stats["outcomes_written"] += 1
            if max_rows is not None and stats["outcomes_written"] >= max_rows:
                return stats

    return stats


def compute_model_accuracy(
    *,
    outcomes_path: Path | str | None = None,
    days: int | None = 30,
    horizon: str = "1d",
) -> dict:
    """Aggregate per-model accuracy from the backfilled outcomes.

    Returns `{model: {"samples", "correct", "accuracy", "agreement_with_primary"}}`.
    `agreement_with_primary` is None for the primary model itself.
    """
    path = Path(outcomes_path) if outcomes_path else _OUTCOMES
    if not path.exists():
        return {}
    cutoff = None
    if days is not None:
        cutoff = datetime.now(UTC) - timedelta(days=days)

    samples: dict[str, int] = {}
    correct: dict[str, int] = {}
    agree: dict[str, int] = {}
    agree_total: dict[str, int] = {}
    kinds: dict[str, str] = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("horizon") != horizon:
            continue
        try:
            ts = datetime.fromisoformat(row.get("ts", ""))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
        except (TypeError, ValueError):
            continue
        if cutoff is not None and ts < cutoff:
            continue
        model = row.get("model")
        if not model:
            continue
        samples[model] = samples.get(model, 0) + 1
        if row.get("correct"):
            correct[model] = correct.get(model, 0) + 1
        agr = row.get("agreement_with_primary")
        if agr is not None:
            agree_total[model] = agree_total.get(model, 0) + 1
            if agr:
                agree[model] = agree.get(model, 0) + 1
        kinds[model] = row.get("kind", kinds.get(model, "?"))

    result = {}
    for model, n in samples.items():
        c = correct.get(model, 0)
        at = agree_total.get(model, 0)
        result[model] = {
            "samples": n,
            "correct": c,
            "accuracy": c / n if n else None,
            "agreement_with_primary": (
                agree.get(model, 0) / at if at else None
            ),
            "kind": kinds.get(model, "?"),
        }
    return result
