"""Self-improvement engine for /fin-silver and /fin-gold commands.

Backfills outcomes for past verdicts, computes accuracy, identifies patterns,
and writes lessons learned for future invocations to read.

Run daily: .venv/Scripts/python.exe portfolio/fin_evolve.py
Auto-runs via main loop integration.

Output: data/fin_command_lessons.json
"""

import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean

logger = logging.getLogger("portfolio.fin_evolve")

# Resolve project root so imports work when run standalone
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
_DATA_DIR = _PROJECT_ROOT / "data"

_LOG_FILE = _DATA_DIR / "fin_command_log.jsonl"
_PRICE_FILE = _DATA_DIR / "price_snapshots_hourly.jsonl"
_LESSONS_FILE = _DATA_DIR / "fin_command_lessons.json"
_EVOLVE_STATE_FILE = _DATA_DIR / "fin_evolve_state.json"
_EVOLVE_INTERVAL_SEC = 24 * 3600  # Daily

# Minimum samples required before generating a lesson for a group
_MIN_SAMPLES_LESSON = 3
# Minimum total scored verdicts before generating lessons at all
_MIN_TOTAL_SCORED = 5
# Maximum time window (in hours) to accept a price snapshot as valid
_MAX_PRICE_WINDOW_HOURS = 6


# ---------------------------------------------------------------------------
# File I/O helpers — delegate to file_utils when available
# ---------------------------------------------------------------------------

def _load_json(path, default=None):
    """Load JSON, falling back to file_utils if available."""
    try:
        from portfolio.file_utils import load_json
        return load_json(path, default=default)
    except ImportError:
        import json
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            return default


def _load_jsonl(path, limit=None):
    """Load JSONL, falling back to file_utils if available."""
    try:
        from portfolio.file_utils import load_jsonl
        return load_jsonl(path, limit=limit)
    except ImportError:
        import json
        entries = []
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            pass
        return entries


def _atomic_write_json(path, data):
    """Atomic JSON write, falling back to file_utils if available."""
    try:
        from portfolio.file_utils import atomic_write_json
        atomic_write_json(path, data)
    except ImportError:
        import json
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(data, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )


def _atomic_write_jsonl(path, entries):
    """Atomic JSONL rewrite, falling back to file_utils if available."""
    try:
        from portfolio.file_utils import atomic_write_jsonl
        atomic_write_jsonl(path, entries)
    except ImportError:
        import json
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


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
            entry_ts = entry_ts.replace(tzinfo=timezone.utc)
        tgt = target_ts
        if tgt.tzinfo is None:
            tgt = tgt.replace(tzinfo=timezone.utc)

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
# Outcome backfill
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

    now = datetime.now(timezone.utc)
    updated_count = 0

    for entry in entries:
        ts = _parse_iso(entry.get("ts"))
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

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
# Evolution / lesson generation
# ---------------------------------------------------------------------------

def evolve():
    """Analyze past verdicts with outcomes. Write lessons learned.

    Returns the lessons dict, or None if not enough data.
    """
    entries = _load_jsonl(_LOG_FILE)

    # Only use entries that have outcomes
    scored = [e for e in entries if "outcome_3d_pct" in e]
    if len(scored) < _MIN_TOTAL_SCORED:
        logger.info(
            "Fin evolve: only %d scored verdicts (need %d), skipping",
            len(scored),
            _MIN_TOTAL_SCORED,
        )
        return None

    now_iso = datetime.now(timezone.utc).isoformat()
    lessons = {
        "generated_at": now_iso,
        "total_verdicts": len(scored),
        "total_logged": len(entries),
        "min_sample_for_lesson": _MIN_SAMPLES_LESSON,
    }

    # 1. Overall accuracy per command
    lessons["by_command"] = {}
    for cmd in ("fin-silver", "fin-gold"):
        cmd_entries = [e for e in scored if e.get("command") == cmd]
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
                # Positive = overconfident, negative = underconfident
            }

    # 2. Accuracy by regime
    lessons["by_regime"] = _analyze_by_field(scored, "regime")

    # 3. Accuracy by confidence bucket
    lessons["by_confidence"] = _analyze_by_confidence(scored)

    # 4. Anti-patterns (conditions where verdicts are consistently wrong)
    lessons["anti_patterns"] = _find_anti_patterns(scored)

    # 5. Confirmed patterns (conditions where verdicts are consistently right)
    lessons["confirmed_patterns"] = _find_confirmed_patterns(scored)

    # 6. Signal trust adjustments
    lessons["signal_trust"] = _compute_signal_trust(scored)

    # 7. Confidence calibration advice
    lessons["calibration_advice"] = _compute_calibration_advice(scored)

    # 8. Execution time stats
    times = [
        e.get("execution_time_sec")
        for e in entries
        if e.get("execution_time_sec") is not None
    ]
    if times:
        lessons["execution_stats"] = {
            "avg_sec": round(mean(times), 1),
            "min_sec": round(min(times), 1),
            "max_sec": round(max(times), 1),
            "n": len(times),
        }

    # 9. Per-ticker accuracy
    lessons["by_ticker"] = {}
    for ticker in ("XAG-USD", "XAU-USD"):
        ticker_entries = [e for e in scored if e.get("ticker") == ticker]
        if ticker_entries:
            correct_3d = [
                e for e in ticker_entries if e.get("verdict_correct_3d") is True
            ]
            total_evaluable = [
                e for e in ticker_entries
                if e.get("verdict_correct_3d") is not None
            ]
            accuracy = (
                len(correct_3d) / len(total_evaluable) if total_evaluable else None
            )
            lessons["by_ticker"][ticker] = {
                "accuracy_3d": round(accuracy, 3) if accuracy is not None else None,
                "n_evaluable": len(total_evaluable),
                "n_total": len(ticker_entries),
            }

    _atomic_write_json(_LESSONS_FILE, lessons)
    logger.info(
        "Fin evolve wrote lessons: %d verdicts scored, %d total",
        len(scored),
        len(entries),
    )
    return lessons


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
        result = evolve()
        _atomic_write_json(
            _EVOLVE_STATE_FILE,
            {
                "last_run_epoch": now,
                "last_run_iso": datetime.now(timezone.utc).isoformat(),
                "status": "ok",
                "verdicts_scored": (
                    result.get("total_verdicts") if result else 0
                ),
                "outcomes_backfilled": n_backfilled,
            },
        )
        logger.info(
            "Fin evolve completed: %d verdicts scored, %d outcomes backfilled",
            result.get("total_verdicts", 0) if result else 0,
            n_backfilled,
        )
        return result
    except Exception as e:
        logger.warning("Fin evolve failed: %s", e)
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

    print("=== Fin Evolve: Self-Improvement Engine ===")
    print()

    # Step 1: backfill outcomes
    print("Step 1: Backfilling outcomes...")
    n_backfilled = backfill_outcomes()
    print(f"  Backfilled {n_backfilled} outcome fields")

    # Step 2: evolve / generate lessons
    print()
    print("Step 2: Generating lessons...")
    result = evolve()
    if result is None:
        print("  Not enough scored verdicts yet (need >= 5)")
    else:
        print(f"  Total verdicts scored: {result.get('total_verdicts', 0)}")
        print(f"  Calibration: {result.get('calibration_advice', 'N/A')}")
        if result.get("anti_patterns"):
            print(f"  Anti-patterns found: {len(result['anti_patterns'])}")
            for p in result["anti_patterns"]:
                print(f"    - {p}")
        if result.get("confirmed_patterns"):
            print(f"  Confirmed patterns: {len(result['confirmed_patterns'])}")
            for p in result["confirmed_patterns"]:
                print(f"    - {p}")
        print(f"\n  Lessons written to: {_LESSONS_FILE}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
