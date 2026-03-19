"""System-wide self-improvement engine for all prediction sources.

Scores verdicts from:
1. /fin-silver and /fin-gold commands (data/fin_command_log.jsonl)
2. Layer 2 journal outlooks (data/layer2_journal.jsonl -> data/journal_outcomes.jsonl)

Computes accuracy, identifies patterns, and writes unified lessons learned.

Run every 4h via main loop integration, or standalone:
    .venv/Scripts/python.exe portfolio/fin_evolve.py

Output: data/system_lessons.json (replaces data/fin_command_lessons.json)
"""

import logging
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import mean

from portfolio.file_utils import (
    atomic_append_jsonl as _atomic_append_jsonl_single,
)
from portfolio.file_utils import (
    atomic_write_json as _atomic_write_json,
)
from portfolio.file_utils import (
    atomic_write_jsonl as _atomic_write_jsonl,
)
from portfolio.file_utils import (
    load_json as _load_json,
)
from portfolio.file_utils import (
    load_jsonl as _load_jsonl,
)

logger = logging.getLogger("portfolio.fin_evolve")

# Resolve project root so imports work when run standalone
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
_DATA_DIR = _PROJECT_ROOT / "data"

_LOG_FILE = _DATA_DIR / "fin_command_log.jsonl"
_JOURNAL_FILE = _DATA_DIR / "layer2_journal.jsonl"
_JOURNAL_OUTCOMES_FILE = _DATA_DIR / "journal_outcomes.jsonl"
_PRICE_FILE = _DATA_DIR / "price_snapshots_hourly.jsonl"
_LESSONS_FILE = _DATA_DIR / "system_lessons.json"
_LEGACY_LESSONS_FILE = _DATA_DIR / "fin_command_lessons.json"
_EVOLVE_STATE_FILE = _DATA_DIR / "fin_evolve_state.json"
_EVOLVE_INTERVAL_SEC = 2 * 3600  # Every 2 hours — aligned with L2 invocation cycle

# Minimum samples required before generating a lesson for a group
_MIN_SAMPLES_LESSON = 3
# Minimum total scored verdicts before generating lessons at all
_MIN_TOTAL_SCORED = 5
# Maximum time window (in hours) to accept a price snapshot as valid
_MAX_PRICE_WINDOW_HOURS = 6


def _atomic_append_jsonl(path, entries):
    """Append multiple entries to a JSONL file."""
    for entry in entries:
        _atomic_append_jsonl_single(path, entry)


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

def _parse_iso(ts_str):
    """Parse an ISO-8601 timestamp string into a timezone-aware datetime."""
    if not ts_str:
        return None
    try:
        # Python 3.11+ handles most ISO formats natively
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Price history lookup
# ---------------------------------------------------------------------------

def _load_price_history():
    """Load hourly price snapshots from disk.

    Returns a list of dicts: [{"ts": "...", "prices": {"XAG-USD": 80.5, ...}}, ...]
    Sorted by timestamp ascending.
    """
    entries = _load_jsonl(_PRICE_FILE)
    # Pre-parse timestamps for faster lookups
    for entry in entries:
        entry["_parsed_ts"] = _parse_iso(entry.get("ts"))
    return entries


def _find_price_at(price_history, ticker, target_ts):
    """Find the price closest to *target_ts* for the given ticker.

    Only accepts a match within _MAX_PRICE_WINDOW_HOURS hours of target.
    Returns the price (float) or None.
    """
    if not price_history or not target_ts:
        return None

    best = None
    best_diff = float("inf")

    for entry in price_history:
        entry_ts = entry.get("_parsed_ts")
        if entry_ts is None:
            continue
        # Ensure both are offset-aware for comparison
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.replace(tzinfo=UTC)
        tgt = target_ts
        if tgt.tzinfo is None:
            tgt = tgt.replace(tzinfo=UTC)

        diff = abs((entry_ts - tgt).total_seconds())
        if diff < best_diff and ticker in entry.get("prices", {}):
            best_diff = diff
            best = entry["prices"][ticker]

    if best_diff < _MAX_PRICE_WINDOW_HOURS * 3600:
        return best
    return None


# ---------------------------------------------------------------------------
# Verdict correctness checker
# ---------------------------------------------------------------------------

def _check_verdict(verdict, outcome_pct):
    """Was the verdict correct?

    bullish + price went up = correct
    bearish + price went down = correct
    neutral = always None (no claim made)
    """
    if not verdict or verdict == "neutral":
        return None  # Can't evaluate neutral
    if verdict == "bullish":
        return outcome_pct > 0
    if verdict == "bearish":
        return outcome_pct < 0
    return None


# ---------------------------------------------------------------------------
# Outcome backfill — fin_command_log.jsonl (existing)
# ---------------------------------------------------------------------------

def backfill_outcomes():
    """For each logged verdict, check if enough time has passed to evaluate it.

    Reads fin_command_log.jsonl, looks up prices from price_snapshots_hourly.jsonl,
    and writes outcome fields back into the log entries. Atomic rewrite.

    Returns the number of entries updated.
    """
    entries = _load_jsonl(_LOG_FILE)
    if not entries:
        return 0

    prices = _load_price_history()
    if not prices:
        logger.debug("No price history available for backfill")
        return 0

    now = datetime.now(UTC)
    updated_count = 0

    for entry in entries:
        ts = _parse_iso(entry.get("ts"))
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)

        age_hours = (now - ts).total_seconds() / 3600
        ticker = entry.get("ticker")
        price_then = entry.get("price_usd")

        if not ticker or not price_then:
            continue

        # Backfill 1d outcome after 24h
        if age_hours >= 24 and "outcome_1d_pct" not in entry:
            price_1d = _find_price_at(prices, ticker, ts + timedelta(days=1))
            if price_1d:
                entry["outcome_1d_pct"] = round((price_1d / price_then - 1) * 100, 3)
                entry["verdict_correct_1d"] = _check_verdict(
                    entry.get("verdict_1_3d"), entry["outcome_1d_pct"]
                )
                updated_count += 1

        # Backfill 3d outcome after 72h
        if age_hours >= 72 and "outcome_3d_pct" not in entry:
            price_3d = _find_price_at(prices, ticker, ts + timedelta(days=3))
            if price_3d:
                entry["outcome_3d_pct"] = round((price_3d / price_then - 1) * 100, 3)
                entry["verdict_correct_3d"] = _check_verdict(
                    entry.get("verdict_1_3d"), entry["outcome_3d_pct"]
                )
                updated_count += 1

        # Backfill 7d outcome after 168h
        if age_hours >= 168 and "outcome_7d_pct" not in entry:
            price_7d = _find_price_at(prices, ticker, ts + timedelta(days=7))
            if price_7d:
                entry["outcome_7d_pct"] = round((price_7d / price_then - 1) * 100, 3)
                entry["verdict_correct_7d"] = _check_verdict(
                    entry.get("verdict_1_4w"), entry["outcome_7d_pct"]
                )
                updated_count += 1

    if updated_count > 0:
        _atomic_write_jsonl(_LOG_FILE, entries)
        logger.info("Backfilled %d outcome fields in fin_command_log.jsonl", updated_count)

    return updated_count


# ---------------------------------------------------------------------------
# Outcome backfill — layer2_journal.jsonl (NEW)
# ---------------------------------------------------------------------------

def backfill_journal_outcomes():
    """Score Layer 2 journal outlooks against actual price moves.

    Reads data/layer2_journal.jsonl, extracts non-neutral outlooks with prices,
    and writes scored results to data/journal_outcomes.jsonl (append-only).

    Returns the number of new outcomes scored.
    """
    journal = _load_jsonl(_JOURNAL_FILE)
    if not journal:
        return 0

    prices = _load_price_history()
    if not prices:
        logger.debug("No price history available for journal backfill")
        return 0

    # Load existing outcomes to avoid re-scoring
    existing = _load_jsonl(_JOURNAL_OUTCOMES_FILE)
    scored_keys = {(e["journal_ts"], e["ticker"]) for e in existing
                   if "journal_ts" in e and "ticker" in e}

    new_outcomes = []
    now = datetime.now(UTC)

    for entry in journal:
        ts = _parse_iso(entry.get("ts"))
        if not ts:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)

        age_hours = (now - ts).total_seconds() / 3600
        if age_hours < 24:
            continue  # Too recent to score

        regime = entry.get("regime", "unknown")
        entry_prices = entry.get("prices", {})
        tickers = entry.get("tickers", {})

        for ticker, ticker_data in tickers.items():
            if not isinstance(ticker_data, dict):
                continue
            outlook = ticker_data.get("outlook")
            if not outlook or outlook == "neutral":
                continue

            price_then = entry_prices.get(ticker)
            if not price_then:
                continue

            key = (entry["ts"], ticker)
            if key in scored_keys:
                continue

            record = {
                "journal_ts": entry["ts"],
                "ticker": ticker,
                "source": "layer2",
                "outlook": outlook,
                "conviction": ticker_data.get("conviction", 0),
                "price_at_verdict": price_then,
                "regime": regime,
                "trigger": entry.get("trigger", ""),
            }

            # Score 1d
            if age_hours >= 24:
                p1d = _find_price_at(prices, ticker, ts + timedelta(days=1))
                if p1d:
                    pct = round((p1d / price_then - 1) * 100, 3)
                    record["outcome_1d_pct"] = pct
                    record["correct_1d"] = _check_verdict(outlook, pct)

            # Score 3d
            if age_hours >= 72:
                p3d = _find_price_at(prices, ticker, ts + timedelta(days=3))
                if p3d:
                    pct = round((p3d / price_then - 1) * 100, 3)
                    record["outcome_3d_pct"] = pct
                    record["correct_3d"] = _check_verdict(outlook, pct)

            record["scored_at"] = now.isoformat()
            new_outcomes.append(record)
            scored_keys.add(key)

    if new_outcomes:
        _atomic_append_jsonl(_JOURNAL_OUTCOMES_FILE, new_outcomes)
        logger.info(
            "Scored %d journal outcomes -> journal_outcomes.jsonl",
            len(new_outcomes),
        )

    return len(new_outcomes)


# ---------------------------------------------------------------------------
# Pattern analysis helpers
# ---------------------------------------------------------------------------

def _analyze_by_field(scored, field):
    """Group accuracy by a categorical field (regime, command, etc.)."""
    groups = {}
    for e in scored:
        val = e.get(field, "unknown")
        if val not in groups:
            groups[val] = {"correct": 0, "wrong": 0, "neutral": 0}
        vc = e.get("verdict_correct_3d")
        if vc is True:
            groups[val]["correct"] += 1
        elif vc is False:
            groups[val]["wrong"] += 1
        else:
            groups[val]["neutral"] += 1

    result = {}
    for val, counts in groups.items():
        total = counts["correct"] + counts["wrong"]
        if total >= _MIN_SAMPLES_LESSON:
            accuracy = counts["correct"] / total
            if accuracy >= 0.65:
                lesson = (
                    f"Verdicts in {val} regime are reliable ({accuracy:.0%})"
                )
            elif accuracy < 0.5:
                lesson = (
                    f"Verdicts in {val} regime are weak ({accuracy:.0%})"
                    " -- reduce confidence by 0.15"
                )
            else:
                lesson = (
                    f"Verdicts in {val} regime are marginal ({accuracy:.0%})"
                )
            result[val] = {
                "accuracy": round(accuracy, 3),
                "n": total,
                "lesson": lesson,
            }
    return result


def _bucket_midpoint(bucket_name):
    """Return the expected midpoint for a confidence bucket."""
    if "high" in bucket_name or ">0.7" in bucket_name:
        return 0.85
    if "low" in bucket_name or "<0.4" in bucket_name:
        return 0.25
    if "medium" in bucket_name or "0.4-0.7" in bucket_name:
        return 0.55
    return 0.5


def _analyze_by_confidence(scored):
    """Accuracy by confidence bucket."""
    buckets = {"high (>0.7)": [], "medium (0.4-0.7)": [], "low (<0.4)": []}
    for e in scored:
        conf = e.get("verdict_1_3d_conf", 0) or 0
        vc = e.get("verdict_correct_3d")
        if vc is None:
            continue
        if conf > 0.7:
            buckets["high (>0.7)"].append(vc)
        elif conf >= 0.4:
            buckets["medium (0.4-0.7)"].append(vc)
        else:
            buckets["low (<0.4)"].append(vc)

    result = {}
    for bucket, outcomes in buckets.items():
        if len(outcomes) >= _MIN_SAMPLES_LESSON:
            accuracy = sum(1 for o in outcomes if o) / len(outcomes)
            result[bucket] = {
                "accuracy": round(accuracy, 3),
                "n": len(outcomes),
                "calibrated": abs(accuracy - _bucket_midpoint(bucket)) < 0.15,
            }
    return result


def _find_anti_patterns(scored):
    """Find conditions where verdicts are consistently WRONG."""
    patterns = []

    # Check: bearish at low RSI (oversold bounce)
    bearish_low_rsi = [
        e for e in scored
        if e.get("verdict_1_3d") == "bearish" and (e.get("rsi") or 50) < 35
    ]
    if len(bearish_low_rsi) >= _MIN_SAMPLES_LESSON:
        wrong = sum(1 for e in bearish_low_rsi if e.get("verdict_correct_3d") is False)
        total = sum(1 for e in bearish_low_rsi if e.get("verdict_correct_3d") is not None)
        if total > 0:
            accuracy = 1 - wrong / total
            if accuracy < 0.4:
                patterns.append(
                    f"Bearish at RSI<35 wrong {1 - accuracy:.0%} of the time"
                    " -- oversold bounces are real"
                )

    # Check: bullish with DXY > 102
    bullish_strong_dxy = [
        e for e in scored
        if e.get("verdict_1_3d") == "bullish" and (e.get("dxy") or 0) > 102
    ]
    if len(bullish_strong_dxy) >= _MIN_SAMPLES_LESSON:
        wrong = sum(
            1 for e in bullish_strong_dxy if e.get("verdict_correct_3d") is False
        )
        total = sum(
            1 for e in bullish_strong_dxy if e.get("verdict_correct_3d") is not None
        )
        if total > 0:
            accuracy = 1 - wrong / total
            if accuracy < 0.4:
                patterns.append(
                    f"Bullish with DXY>102 wrong {1 - accuracy:.0%}"
                    " -- dollar headwind too strong"
                )

    # Check: high confidence but wrong
    high_conf_wrong = [
        e for e in scored
        if (e.get("verdict_1_3d_conf") or 0) > 0.8
        and e.get("verdict_correct_3d") is False
    ]
    high_conf_total = [
        e for e in scored
        if (e.get("verdict_1_3d_conf") or 0) > 0.8
        and e.get("verdict_correct_3d") is not None
    ]
    if len(high_conf_total) >= _MIN_SAMPLES_LESSON:
        accuracy = 1 - len(high_conf_wrong) / len(high_conf_total)
        if accuracy < 0.5:
            patterns.append(
                f"High confidence (>0.8) verdicts only {accuracy:.0%} accurate"
                " -- overconfidence detected"
            )

    # Check: bearish outlook from journal consistently wrong (Layer 2 bearish bias)
    bearish_journal = [
        e for e in scored
        if e.get("source") == "layer2"
        and e.get("verdict_1_3d") == "bearish"
    ]
    if len(bearish_journal) >= _MIN_SAMPLES_LESSON:
        wrong = sum(1 for e in bearish_journal if e.get("verdict_correct_3d") is False)
        total = sum(
            1 for e in bearish_journal if e.get("verdict_correct_3d") is not None
        )
        if total > 0:
            accuracy = 1 - wrong / total
            if accuracy < 0.4:
                patterns.append(
                    f"Layer 2 bearish outlook wrong {1 - accuracy:.0%}"
                    " -- bearish bias detected in journal"
                )

    return patterns


def _find_confirmed_patterns(scored):
    """Find conditions where verdicts are consistently RIGHT."""
    patterns = []

    # Check: bullish when G/S ratio > 65 (silver catch-up thesis)
    bullish_high_gs = [
        e for e in scored
        if e.get("verdict_1_3d") == "bullish"
        and e.get("ticker") == "XAG-USD"
        and (e.get("gs_ratio") or 0) > 65
    ]
    if len(bullish_high_gs) >= _MIN_SAMPLES_LESSON:
        correct = sum(
            1 for e in bullish_high_gs if e.get("verdict_correct_3d") is True
        )
        total = sum(
            1 for e in bullish_high_gs if e.get("verdict_correct_3d") is not None
        )
        if total > 0:
            accuracy = correct / total
            if accuracy > 0.65:
                patterns.append(
                    f"Bullish silver when G/S ratio > 65: {accuracy:.0%} correct"
                    " (catch-up thesis works)"
                )

    # Check: bullish when Chronos accuracy > 70%
    bullish_good_chronos = [
        e for e in scored
        if e.get("verdict_1_3d") == "bullish"
        and (e.get("chronos_accuracy") or 0) > 70
    ]
    if len(bullish_good_chronos) >= _MIN_SAMPLES_LESSON:
        correct = sum(
            1 for e in bullish_good_chronos if e.get("verdict_correct_3d") is True
        )
        total = sum(
            1 for e in bullish_good_chronos
            if e.get("verdict_correct_3d") is not None
        )
        if total > 0:
            accuracy = correct / total
            if accuracy > 0.65:
                patterns.append(
                    f"Bullish + Chronos accuracy>70%: {accuracy:.0%} correct"
                    " -- trust Chronos for directional calls"
                )

    # Check: any verdict aligned with high probability (prob_1d > 65%)
    aligned_high_prob = [
        e for e in scored
        if (
            (e.get("verdict_1_3d") == "bullish" and (e.get("prob_1d") or 50) > 65)
            or (
                e.get("verdict_1_3d") == "bearish"
                and (e.get("prob_1d") or 50) < 35
            )
        )
    ]
    if len(aligned_high_prob) >= _MIN_SAMPLES_LESSON:
        correct = sum(
            1 for e in aligned_high_prob if e.get("verdict_correct_3d") is True
        )
        total = sum(
            1 for e in aligned_high_prob if e.get("verdict_correct_3d") is not None
        )
        if total > 0:
            accuracy = correct / total
            if accuracy > 0.65:
                patterns.append(
                    f"Verdict aligned with 1d probability (>65% or <35%):"
                    f" {accuracy:.0%} correct"
                )

    # Check: Layer 2 bullish outlook consistently correct
    bullish_journal = [
        e for e in scored
        if e.get("source") == "layer2"
        and e.get("verdict_1_3d") == "bullish"
    ]
    if len(bullish_journal) >= _MIN_SAMPLES_LESSON:
        correct = sum(1 for e in bullish_journal if e.get("verdict_correct_3d") is True)
        total = sum(
            1 for e in bullish_journal if e.get("verdict_correct_3d") is not None
        )
        if total > 0:
            accuracy = correct / total
            if accuracy > 0.65:
                patterns.append(
                    f"Layer 2 bullish outlook {accuracy:.0%} correct"
                    " -- bullish conviction is reliable"
                )

    return patterns


def _compute_signal_trust(scored):
    """Compute per-signal trust based on logged signal consensus vs outcome."""
    # Group by signal consensus direction vs actual outcome
    consensus_outcomes = {"BUY": [], "SELL": [], "HOLD": []}
    for e in scored:
        consensus = e.get("signal_consensus", "HOLD")
        vc = e.get("verdict_correct_3d")
        if vc is not None and consensus in consensus_outcomes:
            consensus_outcomes[consensus].append(vc)

    result = {}
    for direction, outcomes in consensus_outcomes.items():
        if len(outcomes) >= _MIN_SAMPLES_LESSON:
            accuracy = sum(1 for o in outcomes if o) / len(outcomes)
            result[direction] = {
                "accuracy": round(accuracy, 3),
                "n": len(outcomes),
            }
    return result


def _compute_calibration_advice(scored):
    """Compute calibration advice: is the system over/underconfident?"""
    evaluable = [
        (e.get("verdict_1_3d_conf") or 0, e.get("verdict_correct_3d"))
        for e in scored
        if e.get("verdict_correct_3d") is not None
    ]
    if len(evaluable) < _MIN_TOTAL_SCORED:
        return "Not enough data yet"

    avg_conf = mean([c for c, _ in evaluable])
    accuracy = sum(1 for _, correct in evaluable if correct) / len(evaluable)

    diff = avg_conf - accuracy
    if diff > 0.15:
        return (
            f"OVERCONFIDENT: avg confidence {avg_conf:.2f} but accuracy"
            f" {accuracy:.2f}. Reduce confidence by {diff:.2f}."
        )
    elif diff < -0.15:
        return (
            f"UNDERCONFIDENT: avg confidence {avg_conf:.2f} but accuracy"
            f" {accuracy:.2f}. Your calls are better than you think."
        )
    else:
        return (
            f"WELL CALIBRATED: avg confidence {avg_conf:.2f},"
            f" accuracy {accuracy:.2f}."
        )


# ---------------------------------------------------------------------------
# Cross-asset analysis (NEW)
# ---------------------------------------------------------------------------

def _compute_cross_asset(scored):
    """Compute cross-asset correlation patterns.

    E.g., "when gold outlook is bullish, silver follows X% of the time".
    """
    # Group entries by journal_ts to find same-invocation outlooks
    by_ts = defaultdict(dict)
    for e in scored:
        ts = e.get("journal_ts") or e.get("ts")
        if ts and e.get("ticker"):
            by_ts[ts][e["ticker"]] = e

    results = {}

    # Gold bullish -> silver follow
    gold_bull_silver = []
    for ts, tickers in by_ts.items():
        gold = tickers.get("XAU-USD")
        silver = tickers.get("XAG-USD")
        if (gold and silver
                and gold.get("verdict_1_3d") == "bullish"
                and silver.get("verdict_correct_3d") is not None):
            gold_bull_silver.append(silver.get("verdict_correct_3d"))

    if len(gold_bull_silver) >= _MIN_SAMPLES_LESSON:
        pct = sum(1 for v in gold_bull_silver if v) / len(gold_bull_silver)
        results["gold_bullish_silver_follows"] = {
            "pct": round(pct, 3),
            "n": len(gold_bull_silver),
            "note": f"When gold outlook is bullish, silver is correct {pct:.0%} of the time",
        }

    # BTC bullish -> ETH follow
    btc_bull_eth = []
    for ts, tickers in by_ts.items():
        btc = tickers.get("BTC-USD")
        eth = tickers.get("ETH-USD")
        if (btc and eth
                and btc.get("verdict_1_3d") == "bullish"
                and eth.get("verdict_correct_3d") is not None):
            btc_bull_eth.append(eth.get("verdict_correct_3d"))

    if len(btc_bull_eth) >= _MIN_SAMPLES_LESSON:
        pct = sum(1 for v in btc_bull_eth if v) / len(btc_bull_eth)
        results["btc_bullish_eth_follows"] = {
            "pct": round(pct, 3),
            "n": len(btc_bull_eth),
            "note": f"When BTC outlook is bullish, ETH is correct {pct:.0%} of the time",
        }

    return results


# ---------------------------------------------------------------------------
# Unified scored entries — normalize both sources
# ---------------------------------------------------------------------------

def _normalize_scored(fin_scored, journal_scored):
    """Normalize both prediction sources into a common format for analysis.

    Returns a list of dicts with unified field names.
    """
    all_scored = []

    for e in fin_scored:
        all_scored.append({
            "source": e.get("command", "fin-command"),
            "ticker": e.get("ticker"),
            "verdict_1_3d": e.get("verdict_1_3d"),
            "verdict_1_3d_conf": e.get("verdict_1_3d_conf", 0),
            "verdict_correct_3d": e.get("verdict_correct_3d"),
            "regime": e.get("regime"),
            "outcome_3d_pct": e.get("outcome_3d_pct"),
            "rsi": e.get("rsi"),
            "dxy": e.get("dxy"),
            "gs_ratio": e.get("gs_ratio"),
            "signal_consensus": e.get("signal_consensus"),
            "chronos_accuracy": e.get("chronos_accuracy"),
            "prob_1d": e.get("prob_1d"),
            "ts": e.get("ts"),
        })

    for e in journal_scored:
        all_scored.append({
            "source": "layer2",
            "ticker": e.get("ticker"),
            "verdict_1_3d": e.get("outlook"),
            "verdict_1_3d_conf": e.get("conviction", 0),
            "verdict_correct_3d": e.get("correct_3d"),
            "regime": e.get("regime"),
            "outcome_3d_pct": e.get("outcome_3d_pct"),
            "journal_ts": e.get("journal_ts"),
            "ts": e.get("journal_ts"),
        })

    return all_scored


# ---------------------------------------------------------------------------
# Evolution / lesson generation
# ---------------------------------------------------------------------------

def evolve():
    """Analyze past verdicts with outcomes from ALL sources. Write lessons learned.

    Sources:
    1. fin_command_log.jsonl — /fin-silver, /fin-gold verdicts
    2. journal_outcomes.jsonl — Layer 2 journal outlooks

    Returns the lessons dict, or None if not enough data.
    """
    # --- Source 1: fin_command entries ---
    fin_entries = _load_jsonl(_LOG_FILE)
    fin_scored = [e for e in fin_entries if "outcome_3d_pct" in e]

    # --- Source 2: journal outcomes ---
    journal_outcomes = _load_jsonl(_JOURNAL_OUTCOMES_FILE)
    journal_scored = [e for e in journal_outcomes if "outcome_3d_pct" in e]

    # --- Merge into unified format ---
    all_scored = _normalize_scored(fin_scored, journal_scored)

    if len(all_scored) < _MIN_TOTAL_SCORED:
        logger.info(
            "Fin evolve: only %d scored verdicts (need %d), skipping",
            len(all_scored),
            _MIN_TOTAL_SCORED,
        )
        return None

    now_iso = datetime.now(UTC).isoformat()
    lessons = {
        "generated_at": now_iso,
        "total_verdicts": len(all_scored),
        "total_fin_command": len(fin_scored),
        "total_journal": len(journal_scored),
        "total_logged": len(fin_entries) + len(journal_outcomes),
        "min_sample_for_lesson": _MIN_SAMPLES_LESSON,
    }

    # 1. Accuracy by source
    lessons["by_source"] = {}
    source_groups = defaultdict(list)
    for e in all_scored:
        source_groups[e.get("source", "unknown")].append(e)

    for source, entries in source_groups.items():
        correct = sum(1 for e in entries if e.get("verdict_correct_3d") is True)
        evaluable = [e for e in entries if e.get("verdict_correct_3d") is not None]
        if evaluable:
            accuracy = correct / len(evaluable)
            lessons["by_source"][source] = {
                "accuracy_3d": round(accuracy, 3),
                "n_evaluable": len(evaluable),
                "n_total": len(entries),
            }

    # 2. Backwards-compatible: by_command (fin-silver, fin-gold only)
    lessons["by_command"] = {}
    for cmd in ("fin-silver", "fin-gold"):
        cmd_entries = [e for e in fin_scored if e.get("command") == cmd]
        if cmd_entries:
            correct_3d = [
                e for e in cmd_entries if e.get("verdict_correct_3d") is True
            ]
            total_evaluable = [
                e for e in cmd_entries if e.get("verdict_correct_3d") is not None
            ]
            accuracy = (
                len(correct_3d) / len(total_evaluable) if total_evaluable else None
            )
            confs = [e.get("verdict_1_3d_conf") or 0 for e in cmd_entries]
            avg_conf = mean(confs) if confs else 0
            lessons["by_command"][cmd] = {
                "accuracy_3d": round(accuracy, 3) if accuracy is not None else None,
                "n_evaluable": len(total_evaluable),
                "n_total": len(cmd_entries),
                "avg_confidence": round(avg_conf, 3),
                "calibration_error": (
                    round(avg_conf - accuracy, 3) if accuracy is not None else None
                ),
            }

    # 3. Accuracy by regime (ALL sources)
    lessons["by_regime"] = _analyze_by_field(all_scored, "regime")

    # 4. Accuracy by confidence bucket (ALL sources)
    lessons["by_confidence"] = _analyze_by_confidence(all_scored)

    # 5. Anti-patterns (ALL sources)
    lessons["anti_patterns"] = _find_anti_patterns(all_scored)

    # 6. Confirmed patterns (ALL sources)
    lessons["confirmed_patterns"] = _find_confirmed_patterns(all_scored)

    # 7. Signal trust adjustments (fin_command only — journal doesn't have signal_consensus)
    lessons["signal_trust"] = _compute_signal_trust(all_scored)

    # 8. Confidence calibration advice (ALL sources)
    lessons["calibration_advice"] = _compute_calibration_advice(all_scored)

    # 9. Execution time stats (fin_command only)
    times = [
        e.get("execution_time_sec")
        for e in fin_entries
        if e.get("execution_time_sec") is not None
    ]
    if times:
        lessons["execution_stats"] = {
            "avg_sec": round(mean(times), 1),
            "min_sec": round(min(times), 1),
            "max_sec": round(max(times), 1),
            "n": len(times),
        }

    # 10. Per-ticker accuracy (ALL sources, ALL tickers)
    lessons["by_ticker"] = {}
    ticker_groups = defaultdict(list)
    for e in all_scored:
        t = e.get("ticker")
        if t:
            ticker_groups[t].append(e)

    for ticker, entries in sorted(ticker_groups.items()):
        correct_3d = [e for e in entries if e.get("verdict_correct_3d") is True]
        evaluable = [e for e in entries if e.get("verdict_correct_3d") is not None]
        if evaluable:
            accuracy = len(correct_3d) / len(evaluable)
            lessons["by_ticker"][ticker] = {
                "accuracy_3d": round(accuracy, 3),
                "n_evaluable": len(evaluable),
                "n_total": len(entries),
            }

    # 11. Cross-asset patterns (NEW)
    lessons["cross_asset"] = _compute_cross_asset(all_scored)

    # 12. Layer 2 specific patterns
    layer2_patterns = _compute_layer2_patterns(journal_scored)
    if layer2_patterns:
        lessons["layer2_patterns"] = layer2_patterns

    # Write to system_lessons.json (primary output)
    _atomic_write_json(_LESSONS_FILE, lessons)

    # Write backwards-compatible copy for /fin-silver and /fin-gold
    _atomic_write_json(_LEGACY_LESSONS_FILE, lessons)

    logger.info(
        "System evolve wrote lessons: %d verdicts scored "
        "(%d fin_command + %d journal)",
        len(all_scored),
        len(fin_scored),
        len(journal_scored),
    )
    return lessons


def _compute_layer2_patterns(journal_scored):
    """Compute patterns specific to Layer 2 journal predictions.

    Uses the larger journal dataset to identify:
    - Accuracy by trigger type
    - Conviction vs accuracy correlation
    - Regime accuracy for Layer 2 specifically
    """
    if len(journal_scored) < _MIN_SAMPLES_LESSON:
        return None

    result = {}

    # Accuracy by trigger type
    trigger_groups = defaultdict(list)
    for e in journal_scored:
        trigger = e.get("trigger", "unknown")
        # Normalize trigger strings (they can be verbose)
        if "consensus" in str(trigger).lower():
            trigger_key = "consensus"
        elif "price" in str(trigger).lower():
            trigger_key = "price_move"
        elif "fear" in str(trigger).lower() or "greed" in str(trigger).lower():
            trigger_key = "fear_greed"
        elif "sentiment" in str(trigger).lower():
            trigger_key = "sentiment"
        elif "post-trade" in str(trigger).lower():
            trigger_key = "post_trade"
        else:
            trigger_key = "other"

        if e.get("correct_3d") is not None:
            trigger_groups[trigger_key].append(e.get("correct_3d"))

    by_trigger = {}
    for trigger, outcomes in trigger_groups.items():
        if len(outcomes) >= _MIN_SAMPLES_LESSON:
            accuracy = sum(1 for o in outcomes if o) / len(outcomes)
            by_trigger[trigger] = {
                "accuracy": round(accuracy, 3),
                "n": len(outcomes),
            }
    if by_trigger:
        result["by_trigger"] = by_trigger

    # Accuracy by conviction bucket (journal uses conviction, not verdict_1_3d_conf)
    conviction_buckets = {"high (>0.7)": [], "medium (0.4-0.7)": [], "low (<0.4)": []}
    for e in journal_scored:
        conv = e.get("conviction", 0) or 0
        correct = e.get("correct_3d")
        if correct is None:
            continue
        if conv > 0.7:
            conviction_buckets["high (>0.7)"].append(correct)
        elif conv >= 0.4:
            conviction_buckets["medium (0.4-0.7)"].append(correct)
        else:
            conviction_buckets["low (<0.4)"].append(correct)

    by_conviction = {}
    for bucket, outcomes in conviction_buckets.items():
        if len(outcomes) >= _MIN_SAMPLES_LESSON:
            accuracy = sum(1 for o in outcomes if o) / len(outcomes)
            by_conviction[bucket] = {
                "accuracy": round(accuracy, 3),
                "n": len(outcomes),
            }
    if by_conviction:
        result["by_conviction"] = by_conviction

    return result if result else None


# ---------------------------------------------------------------------------
# Main loop integration
# ---------------------------------------------------------------------------

def maybe_evolve(config=None):
    """Run evolution if enough time has elapsed. Called from main loop.

    Returns the lessons dict if evolution ran, else None.
    """
    state = _load_json(_EVOLVE_STATE_FILE, default={})
    last_run = state.get("last_run_epoch", 0)
    now = time.time()

    if (now - last_run) < _EVOLVE_INTERVAL_SEC:
        return None

    try:
        n_backfilled = backfill_outcomes()
        n_journal = backfill_journal_outcomes()
        result = evolve()
        _atomic_write_json(
            _EVOLVE_STATE_FILE,
            {
                "last_run_epoch": now,
                "last_run_iso": datetime.now(UTC).isoformat(),
                "status": "ok",
                "verdicts_scored": (
                    result.get("total_verdicts") if result else 0
                ),
                "outcomes_backfilled": n_backfilled,
                "journal_outcomes_scored": n_journal,
            },
        )
        logger.info(
            "System evolve completed: %d verdicts scored, "
            "%d fin outcomes backfilled, %d journal outcomes scored",
            result.get("total_verdicts", 0) if result else 0,
            n_backfilled,
            n_journal,
        )
        return result
    except Exception as e:
        logger.warning("System evolve failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main():
    """Run backfill + evolution directly (standalone mode)."""
    # Ensure project root is on sys.path for imports
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    print("=== System Evolve: Self-Improvement Engine ===")
    print()

    # Step 1: backfill fin_command outcomes
    print("Step 1: Backfilling fin_command outcomes...")
    n_backfilled = backfill_outcomes()
    print(f"  Backfilled {n_backfilled} outcome fields")

    # Step 2: backfill journal outcomes
    print()
    print("Step 2: Scoring journal outlooks...")
    n_journal = backfill_journal_outcomes()
    print(f"  Scored {n_journal} new journal outcomes")

    # Step 3: evolve / generate lessons
    print()
    print("Step 3: Generating unified lessons...")
    result = evolve()
    if result is None:
        print("  Not enough scored verdicts yet (need >= 5)")
    else:
        print(f"  Total verdicts scored: {result.get('total_verdicts', 0)}")
        print(f"    - fin_command: {result.get('total_fin_command', 0)}")
        print(f"    - journal: {result.get('total_journal', 0)}")
        print(f"  Calibration: {result.get('calibration_advice', 'N/A')}")

        if result.get("by_source"):
            print("  By source:")
            for source, data in result["by_source"].items():
                print(f"    {source}: {data.get('accuracy_3d', 'N/A'):.1%} "
                      f"({data.get('n_evaluable', 0)} evaluable)")

        if result.get("by_ticker"):
            print("  By ticker:")
            for ticker, data in sorted(result["by_ticker"].items()):
                print(f"    {ticker}: {data.get('accuracy_3d', 'N/A'):.1%} "
                      f"({data.get('n_evaluable', 0)} evaluable)")

        if result.get("anti_patterns"):
            print(f"  Anti-patterns found: {len(result['anti_patterns'])}")
            for p in result["anti_patterns"]:
                print(f"    - {p}")
        if result.get("confirmed_patterns"):
            print(f"  Confirmed patterns: {len(result['confirmed_patterns'])}")
            for p in result["confirmed_patterns"]:
                print(f"    - {p}")

        if result.get("cross_asset"):
            print("  Cross-asset patterns:")
            for key, data in result["cross_asset"].items():
                print(f"    - {data.get('note', key)}")

        print(f"\n  Lessons written to: {_LESSONS_FILE}")
        print(f"  Legacy copy: {_LEGACY_LESSONS_FILE}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
