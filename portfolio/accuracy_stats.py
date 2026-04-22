import json
import logging
import threading
import time
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("portfolio.accuracy_stats")

from datetime import UTC

from portfolio.file_utils import atomic_write_json as _atomic_write_json
from portfolio.file_utils import load_json, load_jsonl_tail
from portfolio.tickers import DISABLED_SIGNALS, SIGNAL_NAMES

# C2: Protect all read-modify-write cache operations from concurrent ticker threads
_accuracy_write_lock = threading.Lock()

# BUG-178 (2026-04-16): thundering-herd protection. The disk-backed accuracy
# caches expire on a 1h TTL; on the first cycle after expiry, all 5 ticker
# threads race through load_cached_accuracy() → None → signal_accuracy() and
# each pays the 7s+ cost of loading 50,000 signal-log entries from SQLite.
# Wall time was measured at 215s for a 5-thread race vs 7s single-threaded —
# 30x amplification driven by GIL + DB + file-I/O serialization. The
# get_or_compute_*() helpers below use double-checked locking: cache hits
# take the fast path with no lock acquisition; only the first miss-thread
# computes, and the others wait on _accuracy_compute_lock and then read the
# freshly-populated cache. The lock is held THROUGH the compute (unlike the
# signal_utility cache below) because cache-miss is rare (~once per hour
# per horizon) and serializing 4 threads through a 7s wait is far cheaper
# than 4 redundant 50000-entry SQL scans.
_accuracy_compute_lock = threading.Lock()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SIGNAL_LOG = DATA_DIR / "signal_log.jsonl"
ACCURACY_CACHE_FILE = DATA_DIR / "accuracy_cache.json"
BEST_HORIZON_CACHE_FILE = DATA_DIR / "best_horizon_cache.json"
ACCURACY_CACHE_TTL = 3600
HORIZONS = ["3h", "4h", "12h", "1d", "3d", "5d", "10d"]

# In-memory cache for signal_utility (added 2026-04-15, BUG-178 mitigation).
# signal_utility() walks every entry in the signal log (~6320 snapshots / ~92K
# ticker rows as of this writing) and costs ~3.6s cold and <50ms hot. It's
# called from generate_signal() on every ticker, every cycle, with NO
# disk-backed cache — so when the OS file cache is cold (memory pressure,
# fresh process, antivirus scan) 5 concurrent ticker threads each pay the
# 3-4s cold read, which can compound under file-cache page-in contention.
#
# Invalidation is pure TTL (300s) — there is NO mtime check against
# signal_log.db, so a backfill that writes new outcomes mid-cycle is only
# visible to signal_utility after the TTL expires. This is an explicit
# trade: outcome backfill runs daily at 18:00 local via the PF-OutcomeCheck
# scheduled task (see docs/operational-runbook.md), so a 5-minute staleness
# window is dominated by the 24-hour write cadence. Code paths that need
# immediately-fresh utility (tests, outcome_tracker, post-backfill reports)
# must either pass entries= explicitly (which bypasses the cache) or call
# invalidate_signal_utility_cache() after the write.
#
# 300s TTL matches the shortest LLM rotation period and is well below the
# 3600s ACCURACY_CACHE_TTL used for the disk-backed caches. The lock
# guards the (timestamp, value) tuple so two threads racing to refresh
# can't corrupt the dict. Dogpile behavior: the lock is held ONLY for the
# swap, NOT for the compute — the slow signal_utility() call happens
# outside the lock, so other threads waiting on the lock see the fresh
# value the moment the first thread returns. Two threads that both miss
# on a TTL-boundary race will each recompute once (one wasted walk), but
# neither blocks the other. This is cheaper than holding a global lock
# through a 3.6s disk scan and funneling every ticker thread through it.
_SIGNAL_UTILITY_CACHE_TTL = 300.0
_signal_utility_cache: dict[str, tuple[float, dict]] = {}
_signal_utility_cache_lock = threading.Lock()


def load_entries():
    """Load signal log entries. Prefers SQLite if available, falls back to JSONL."""
    try:
        from portfolio.signal_db import SignalDB
        db = SignalDB()
        try:
            count = db.snapshot_count()
            if count > 0:
                entries = db.load_entries()
                return entries
        finally:
            # BUG-137: Always close DB, even if load_entries() throws.
            db.close()
    except Exception as e:
        logger.debug("SQLite signal_db unavailable, falling back to JSONL: %s", e)
    # H2: Fallback to JSONL using atomic load_jsonl_tail (avoids raw open()).
    # 50000 entries covers full accuracy computation; reading all 68MB risks OOM.
    if not SIGNAL_LOG.exists():
        return []
    entries = load_jsonl_tail(SIGNAL_LOG, max_entries=50000)
    return entries if entries else []


_MIN_CHANGE_PCT = 0.05  # outcomes within ±0.05% are treated as neutral (skip)


def _vote_correct(vote, change_pct, min_change_pct=None):
    """Check if a signal vote matches the price outcome.

    Returns True (correct), False (incorrect), or None (neutral — skip this outcome).
    Outcomes within ±min_change_pct are considered noise and should not count
    for or against the signal's accuracy.
    """
    threshold = min_change_pct if min_change_pct is not None else _MIN_CHANGE_PCT
    # 2026-04-22: some outcome entries have change_pct=None (missing backfill
    # data for 4h+ horizons). Treat as neutral instead of TypeErroring — was
    # killing --accuracy report mid-horizon. Matches the None-guard pattern
    # at accuracy_stats.py:1617 and in ic_computation / train_signal_weights.
    if change_pct is None or abs(change_pct) < threshold:
        return None  # neutral — price didn't move enough to judge (or unknown)
    if vote == "BUY" and change_pct > 0:
        return True
    return bool(vote == "SELL" and change_pct < 0)


def signal_accuracy(horizon="1d", since=None, entries=None):
    """Compute per-signal accuracy, optionally filtered to entries after `since`.

    Args:
        horizon: Outcome horizon to evaluate ("1d", "3d", "5d", "10d").
        since: Optional ISO-8601 string cutoff. Only entries with ts >= since
               are included. None means all entries (no time filter).
        entries: Pre-loaded entries list. If None, loads from disk.

    Returns:
        dict: {signal_name: {correct, total, accuracy, pct}} for each signal.
    """
    if entries is None:
        entries = load_entries()
    stats = {s: {"correct": 0, "total": 0,
                 "correct_buy": 0, "total_buy": 0,
                 "correct_sell": 0, "total_sell": 0} for s in SIGNAL_NAMES}
    # 2026-04-22 follow-up: count outcomes we skip because change_pct is None.
    # Previously these crashed the report; now they're silently dropped, which
    # would let a data-quality regression (e.g. outcome_tracker writing nulls)
    # go unnoticed. Surface the count so operators see drift.
    null_change_pct_skipped = 0
    total_outcomes_seen = 0

    for entry in entries:
        if since and entry.get("ts", "") < since:
            continue
        outcomes = entry.get("outcomes", {})
        tickers = entry.get("tickers", {})

        for ticker, tdata in tickers.items():
            outcome = outcomes.get(ticker, {}).get(horizon)
            if not outcome:
                continue
            total_outcomes_seen += 1

            change_pct = outcome.get("change_pct", 0)
            if change_pct is None:
                null_change_pct_skipped += 1
            signals = tdata.get("signals", {})

            for sig_name in SIGNAL_NAMES:
                vote = signals.get(sig_name, "HOLD")
                if vote == "HOLD":
                    continue
                result_val = _vote_correct(vote, change_pct)
                if result_val is None:
                    continue  # neutral outcome — don't count
                stats[sig_name]["total"] += 1
                if vote == "BUY":
                    stats[sig_name]["total_buy"] += 1
                    if result_val:
                        stats[sig_name]["correct_buy"] += 1
                else:
                    stats[sig_name]["total_sell"] += 1
                    if result_val:
                        stats[sig_name]["correct_sell"] += 1
                if result_val:
                    stats[sig_name]["correct"] += 1

    if null_change_pct_skipped > 0:
        pct = 100.0 * null_change_pct_skipped / total_outcomes_seen
        logger.info(
            "signal_accuracy[%s]: skipped %d/%d outcomes (%.2f%%) with "
            "change_pct=None — check outcome_tracker / signal_db backfill",
            horizon, null_change_pct_skipped, total_outcomes_seen, pct,
        )

    result = {}
    for sig_name in SIGNAL_NAMES:
        s = stats[sig_name]
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        buy_acc = s["correct_buy"] / s["total_buy"] if s["total_buy"] > 0 else 0.0
        sell_acc = s["correct_sell"] / s["total_sell"] if s["total_sell"] > 0 else 0.0
        result[sig_name] = {
            "correct": s["correct"],
            "total": s["total"],
            "accuracy": acc,
            "pct": round(acc * 100, 1),
            "correct_buy": s["correct_buy"],
            "total_buy": s["total_buy"],
            "buy_accuracy": round(buy_acc, 4),
            "correct_sell": s["correct_sell"],
            "total_sell": s["total_sell"],
            "sell_accuracy": round(sell_acc, 4),
        }
    return result


def signal_accuracy_recent(horizon="1d", days=7):
    """Compute per-signal accuracy using only the last N days of data.

    Thin wrapper around signal_accuracy() with a time cutoff.
    """
    from datetime import datetime, timedelta

    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    return signal_accuracy(horizon, since=cutoff)


def signal_accuracy_ewma(horizon="1d", halflife_days=5, entries=None):
    """Compute per-signal accuracy with exponential decay weighting.

    Recent observations are weighted higher than older ones. The weight for
    an observation of age ``age_days`` is ``exp(-ln(2) / halflife_days * age_days)``,
    meaning entries that are ``halflife_days`` old receive half the weight of
    today's entries.

    Args:
        horizon: Outcome horizon to evaluate ("1d", "3d", "5d", "10d").
        halflife_days: Half-life of the decay in days (default 5). Smaller
            values weight recent data more aggressively.
        entries: Pre-loaded entries list. If None, loads from disk.

    Returns:
        dict: {signal_name: {accuracy, total_weight, effective_samples, total, correct, pct}}
        where ``total`` and ``correct`` are int(round(...)) of the weighted sums
        for compatibility with the existing accuracy pipeline.
    """
    import math
    from datetime import datetime

    if entries is None:
        entries = load_entries()
    now = datetime.now(UTC)
    decay_rate = math.log(2) / halflife_days  # λ = ln(2) / t½

    # Accumulate per-signal: weighted_total, weighted_correct, sum_of_sq_weights
    stats = {
        s: {"w_total": 0.0, "w_correct": 0.0, "sum_w2": 0.0}
        for s in SIGNAL_NAMES
    }

    for entry in entries:
        # Compute age in days from entry timestamp
        ts_str = entry.get("ts", "")
        try:
            entry_dt = datetime.fromisoformat(ts_str)
            age_days = (now - entry_dt).total_seconds() / 86400.0
            age_days = max(age_days, 0.0)  # clamp: never negative for future entries
        except (ValueError, TypeError):
            continue  # skip malformed timestamps

        weight = math.exp(-decay_rate * age_days)

        outcomes = entry.get("outcomes", {})
        tickers = entry.get("tickers", {})

        for ticker, tdata in tickers.items():
            outcome = outcomes.get(ticker, {}).get(horizon)
            if not outcome:
                continue

            change_pct = outcome.get("change_pct", 0)
            signals = tdata.get("signals", {})

            for sig_name in SIGNAL_NAMES:
                vote = signals.get(sig_name, "HOLD")
                if vote == "HOLD":
                    continue
                result_val = _vote_correct(vote, change_pct)
                if result_val is None:
                    continue  # neutral outcome — skip

                stats[sig_name]["w_total"] += weight
                stats[sig_name]["sum_w2"] += weight * weight
                if result_val:
                    stats[sig_name]["w_correct"] += weight

    result = {}
    for sig_name in SIGNAL_NAMES:
        s = stats[sig_name]
        w_total = s["w_total"]
        w_correct = s["w_correct"]
        sum_w2 = s["sum_w2"]

        if w_total > 0:
            accuracy = w_correct / w_total
            # Kish (1965) effective sample size: n_eff = (Σwᵢ)² / Σwᵢ²
            effective_samples = (w_total * w_total) / sum_w2
        else:
            accuracy = 0.0
            effective_samples = 0.0

        result[sig_name] = {
            "accuracy": accuracy,
            "total_weight": w_total,
            "effective_samples": effective_samples,
            "total": int(round(w_total)),
            "correct": int(round(w_correct)),
            "pct": round(accuracy * 100, 1),
        }
    return result


def signal_accuracy_cost_adjusted(horizon="1d", cost_bps=10.0, entries=None):
    """Compute per-signal accuracy adjusted for transaction costs.

    A signal vote is only counted as correct if the price move exceeds
    the estimated round-trip cost (spread + slippage).  This reveals
    signals that are "technically correct" but unprofitable after execution.

    Args:
        horizon: Outcome horizon to evaluate.
        cost_bps: Estimated round-trip cost in basis points (default 10 bps).
                  Must exceed _MIN_CHANGE_PCT (5 bps) to filter beyond
                  the neutral outcome threshold. Metals warrants ~10 bps,
                  crypto ~5 bps.
        entries: Pre-loaded entries list. If None, loads from disk.

    Returns:
        dict: {signal_name: {correct, total, accuracy, pct, cost_bps}}
    """
    if entries is None:
        entries = load_entries()

    # Cost threshold: moves below this are unprofitable even if directionally correct
    cost_pct = cost_bps / 100.0  # convert bps to percentage

    stats = {s: {"correct": 0, "total": 0} for s in SIGNAL_NAMES}

    for entry in entries:
        outcomes = entry.get("outcomes", {})
        tickers = entry.get("tickers", {})

        for ticker, tdata in tickers.items():
            outcome = outcomes.get(ticker, {}).get(horizon)
            if not outcome:
                continue

            change_pct = outcome.get("change_pct", 0)
            signals = tdata.get("signals", {})

            for sig_name in SIGNAL_NAMES:
                vote = signals.get(sig_name, "HOLD")
                if vote == "HOLD":
                    continue

                # Skip neutral outcomes (below minimum move)
                if abs(change_pct) < _MIN_CHANGE_PCT:
                    continue

                stats[sig_name]["total"] += 1

                # Cost-adjusted: correct only if move exceeds cost
                if (vote == "BUY" and change_pct > cost_pct) or (vote == "SELL" and change_pct < -cost_pct):
                    stats[sig_name]["correct"] += 1

    result = {}
    for sig_name in SIGNAL_NAMES:
        s = stats[sig_name]
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        result[sig_name] = {
            "correct": s["correct"],
            "total": s["total"],
            "accuracy": acc,
            "pct": round(acc * 100, 1),
            "cost_bps": cost_bps,
        }
    return result


def consensus_accuracy(horizon="1d", entries=None, days=None):
    """Aggregate consensus decision accuracy across all tickers.

    For each signal-log entry that has an outcome at `horizon`, check if
    the per-ticker `tdata["consensus"]` BUY/SELL call matched actual
    direction. HOLD calls are skipped (no direction to score).

    BUG-178/W15-W16 follow-up (2026-04-16): added optional `days` kwarg
    for the recent-window variant the degradation tracker needs. The
    existing `entries` kwarg is preserved for callers that already pass
    a pre-loaded list. When both are passed, `entries` wins (caller has
    already filtered).

    Args:
        horizon: Outcome horizon ("3h", "4h", "12h", "1d", "3d", "5d", "10d").
        entries: Pre-loaded entries list (skips both load_entries() and
            the days filter — caller is assumed to have filtered already).
        days: Optional lookback window in days. Ignored if entries is
            provided. None = lifetime aggregate.
    """
    if entries is None:
        entries = load_entries()
        if days is not None:
            from datetime import datetime, timedelta
            cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
            entries = [e for e in entries if e.get("ts", "") >= cutoff]
    correct = 0
    total = 0

    for entry in entries:
        outcomes = entry.get("outcomes", {})
        tickers = entry.get("tickers", {})

        for ticker, tdata in tickers.items():
            consensus = tdata.get("consensus", "HOLD")
            if consensus == "HOLD":
                continue

            outcome = outcomes.get(ticker, {}).get(horizon)
            if not outcome:
                continue

            change_pct = outcome.get("change_pct", 0)
            result_val = _vote_correct(consensus, change_pct)
            if result_val is None:
                continue
            total += 1
            if result_val:
                correct += 1

    acc = correct / total if total > 0 else 0.0
    return {
        "correct": correct,
        "total": total,
        "accuracy": acc,
        "pct": round(acc * 100, 1),
    }


def per_ticker_accuracy(horizon="1d", entries=None):
    if entries is None:
        entries = load_entries()
    stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for entry in entries:
        outcomes = entry.get("outcomes", {})
        tickers = entry.get("tickers", {})

        for ticker, tdata in tickers.items():
            consensus = tdata.get("consensus", "HOLD")
            if consensus == "HOLD":
                continue

            outcome = outcomes.get(ticker, {}).get(horizon)
            if not outcome:
                continue

            change_pct = outcome.get("change_pct", 0)
            result_val = _vote_correct(consensus, change_pct)
            if result_val is None:
                continue
            stats[ticker]["total"] += 1
            if result_val:
                stats[ticker]["correct"] += 1

    result = {}
    for ticker, s in stats.items():
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        result[ticker] = {
            "correct": s["correct"],
            "total": s["total"],
            "accuracy": acc,
            "pct": round(acc * 100, 1),
        }
    return result


def accuracy_by_signal_ticker(signal_name, horizon="1d", days=None, entries=None):
    """Compute per-ticker accuracy for one signal.

    Args:
        signal_name: Signal name present in SIGNAL_NAMES.
        horizon: Outcome horizon to evaluate.
        days: Optional lookback window in days.
        entries: Pre-loaded entries list. BUG-178/W15-W16 follow-up
            (2026-04-16 review): callers that iterate over many signal
            names (e.g. accuracy_degradation._per_ticker_recent) must
            pass a single pre-loaded list instead of letting each call
            re-scan the 50,000-entry SQLite file. Skipping that knob
            blew cycle time by ~290s in the original implementation.

    Returns:
        dict: {ticker: {"accuracy": float, "samples": int, "correct": int}}
    """
    if signal_name not in SIGNAL_NAMES:
        return {}

    if entries is None:
        entries = load_entries()
    cutoff = None
    if days is not None:
        from datetime import datetime, timedelta

        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()

    stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for entry in entries:
        if cutoff and entry.get("ts", "") < cutoff:
            continue

        outcomes = entry.get("outcomes", {})
        tickers = entry.get("tickers", {})
        for ticker, tdata in tickers.items():
            outcome = outcomes.get(ticker, {}).get(horizon)
            if not outcome:
                continue

            vote = (tdata.get("signals") or {}).get(signal_name, "HOLD")
            if vote == "HOLD":
                continue

            change_pct = outcome.get("change_pct", 0)
            result_val = _vote_correct(vote, change_pct)
            if result_val is None:
                continue
            stats[ticker]["total"] += 1
            if result_val:
                stats[ticker]["correct"] += 1

    result = {}
    for ticker, data in stats.items():
        total = data["total"]
        if total == 0:
            continue
        result[ticker] = {
            "accuracy": data["correct"] / total,
            "samples": total,
            "correct": data["correct"],
        }
    return result


def signal_utility(horizon="1d", entries=None):
    """Compute per-signal return magnitude utility.

    For each non-HOLD signal vote with a non-neutral outcome, compute the
    directional return:
      - BUY  → +change_pct   (positive = correct direction)
      - SELL → -change_pct   (negative change = correct direction → positive return)

    Neutral outcomes (|change_pct| < _MIN_CHANGE_PCT) are skipped.

    Args:
        horizon: Outcome horizon to evaluate.
        entries: Pre-loaded entries list. If None, loads from disk.

    Returns:
        dict: {signal_name: {avg_return, total_return, samples, utility_score}}
        where utility_score = avg_return * sqrt(samples).
        Signals with no data get zeros.

    2026-04-15 (BUG-178 mitigation): when `entries` is None, the result is
    cached for _SIGNAL_UTILITY_CACHE_TTL seconds keyed by horizon. The cold
    walk costs ~3.6s on a 6K-snapshot log; with 5 ticker threads per cycle
    and the OS file cache occasionally cold, this was a legitimate
    per-cycle cost. Passing an explicit `entries` list bypasses the cache
    (preserves the old behavior for test fixtures that want a specific
    entries snapshot).
    """
    if entries is None:
        now = time.time()
        with _signal_utility_cache_lock:
            cached = _signal_utility_cache.get(horizon)
            if cached and now - cached[0] < _SIGNAL_UTILITY_CACHE_TTL:
                return cached[1]
        # Cache miss or expired — compute outside the lock to avoid
        # serializing all threads behind the slow path.
        result = _compute_signal_utility(horizon, None)
        with _signal_utility_cache_lock:
            _signal_utility_cache[horizon] = (time.time(), result)
        return result
    # Explicit entries — bypass cache (caller controls the dataset).
    return _compute_signal_utility(horizon, entries)


def _compute_signal_utility(horizon, entries):
    """Actual utility computation. Extracted from signal_utility so the
    cache wrapper can call it without re-entering the cached function
    (and so test fixtures passing explicit entries can hit the raw path).
    """
    import math

    if entries is None:
        entries = load_entries()
    # {sig_name: {"total_return": float, "samples": int}}
    stats = {s: {"total_return": 0.0, "samples": 0} for s in SIGNAL_NAMES}

    for entry in entries:
        outcomes = entry.get("outcomes", {})
        tickers = entry.get("tickers", {})

        for ticker, tdata in tickers.items():
            outcome = outcomes.get(ticker, {}).get(horizon)
            if not outcome:
                continue

            change_pct = outcome.get("change_pct", 0)
            if abs(change_pct) < _MIN_CHANGE_PCT:
                continue  # neutral outcome — skip

            signals = tdata.get("signals", {})
            for sig_name in SIGNAL_NAMES:
                vote = signals.get(sig_name, "HOLD")
                if vote == "HOLD":
                    continue
                # Directional return: positive when signal was correct
                if vote == "BUY":
                    dir_return = change_pct
                else:  # SELL
                    dir_return = -change_pct

                stats[sig_name]["total_return"] += dir_return
                stats[sig_name]["samples"] += 1

    result = {}
    for sig_name in SIGNAL_NAMES:
        s = stats[sig_name]
        n = s["samples"]
        total_ret = s["total_return"]
        avg_ret = total_ret / n if n > 0 else 0.0
        utility = avg_ret * math.sqrt(n) if n > 0 else 0.0
        result[sig_name] = {
            "avg_return": avg_ret,
            "total_return": total_ret,
            "samples": n,
            "utility_score": utility,
        }
    return result


def invalidate_signal_utility_cache():
    """Clear the signal_utility in-memory cache.

    Primarily exists so tests and outcome-backfill code can force a refresh
    after writing new outcomes. Production code does NOT need to call this —
    the 300s TTL is the source of truth.
    """
    with _signal_utility_cache_lock:
        _signal_utility_cache.clear()


def best_worst_signals(horizon="1d", acc=None):
    if acc is None:
        acc = signal_accuracy(horizon)
    qualified = {k: v for k, v in acc.items() if v["total"] >= 5}
    if not qualified:
        return {"best": None, "worst": None}

    best_name = max(qualified, key=lambda k: qualified[k]["accuracy"])
    worst_name = min(qualified, key=lambda k: qualified[k]["accuracy"])
    return {
        "best": (best_name, qualified[best_name]["accuracy"]),
        "worst": (worst_name, qualified[worst_name]["accuracy"]),
    }


def signal_activation_rates(entries=None):
    """Compute per-signal activation rates (how often each signal votes non-HOLD).

    Args:
        entries: Pre-loaded entries list. If None, loads from disk.

    Returns dict: {signal_name: {activation_rate, buy_rate, sell_rate, bias, samples}}
    - activation_rate: fraction of votes that are BUY or SELL (0.0 to 1.0)
    - bias: directional bias = abs(buy_rate - sell_rate) / activation_rate (0=balanced, 1=all one side)
    - rarity_weight: log(1 + 1/activation_rate) — rare signals get higher weight
    - bias_penalty: 1 - bias (minimum 0.1 floor) — directional signals get penalized
    - normalized_weight: rarity_weight * bias_penalty — the final multiplier
    """
    import math

    if entries is None:
        entries = load_entries()
    stats = {s: {"buy": 0, "sell": 0, "total": 0} for s in SIGNAL_NAMES}

    for entry in entries:
        tickers = entry.get("tickers", {})
        for _ticker, tdata in tickers.items():
            signals = tdata.get("signals", {})
            for sig_name in SIGNAL_NAMES:
                vote = signals.get(sig_name)
                if vote is None:
                    continue
                stats[sig_name]["total"] += 1
                if vote == "BUY":
                    stats[sig_name]["buy"] += 1
                elif vote == "SELL":
                    stats[sig_name]["sell"] += 1

    result = {}
    for sig_name in SIGNAL_NAMES:
        s = stats[sig_name]
        total = s["total"]
        if total == 0:
            result[sig_name] = {
                "activation_rate": 0.0, "buy_rate": 0.0, "sell_rate": 0.0,
                "bias": 1.0, "rarity_weight": 1.0, "bias_penalty": 0.1,
                "normalized_weight": 0.1, "samples": 0,
            }
            continue

        buy_rate = s["buy"] / total
        sell_rate = s["sell"] / total
        activation_rate = buy_rate + sell_rate

        # Rarity: IDF-style weight — rare signals get more weight when they vote
        if activation_rate > 0.01:
            rarity_weight = math.log(1 + 1 / activation_rate)
        else:
            rarity_weight = math.log(1 + 100)  # cap for near-zero activation

        # Bias: penalize signals that always vote one direction
        if activation_rate > 0:
            bias = abs(buy_rate - sell_rate) / activation_rate
        else:
            bias = 1.0
        bias_penalty = max(1.0 - bias, 0.1)  # floor at 0.1

        result[sig_name] = {
            "activation_rate": round(activation_rate, 4),
            "buy_rate": round(buy_rate, 4),
            "sell_rate": round(sell_rate, 4),
            "bias": round(bias, 4),
            "rarity_weight": round(rarity_weight, 4),
            "bias_penalty": round(bias_penalty, 4),
            "normalized_weight": round(rarity_weight * bias_penalty, 4),
            "samples": total,
        }

    return result


# P2-B (2026-04-17 adversarial review): default min_recent_samples was 50
# while production (signal_engine) passes 30. Default lowered so non-prod
# callers (backtester, replay script) match live behavior rather than
# silently dropping the blended value for signals with 30-49 recent samples.
_BLEND_DEFAULT_MIN_RECENT_SAMPLES = 30


def blend_accuracy_data(alltime, recent, divergence_threshold=0.15,
                        normal_weight=0.70, fast_weight=0.90,
                        min_recent_samples=_BLEND_DEFAULT_MIN_RECENT_SAMPLES):
    """Blend all-time and recent accuracy using adaptive recency weighting.

    When recent accuracy diverges sharply from all-time (> divergence_threshold),
    fast-track to higher recent weight for faster regime adaptation.

    Args:
        alltime: Dict of {signal_name: {accuracy, total, correct, pct}}.
        recent: Dict of {signal_name: {accuracy, total, correct, pct}}.
        divergence_threshold: Absolute accuracy difference that triggers fast blend.
        normal_weight: Recent weight when divergence is below threshold.
        fast_weight: Recent weight when divergence exceeds threshold.
        min_recent_samples: Minimum recent samples before blending (else use alltime).

    Returns:
        dict: Blended {signal_name: {accuracy, total, correct, pct}}.
    """
    if not alltime and not recent:
        return {}
    # Codex round-10 P2 (2026-04-17 follow-up): don't early-return recent
    # without going through the blend loop - the min_recent_samples floor
    # must apply to recent-only signals too (a 20-sample signal with recent
    # accuracy=0.80 should default to neutral 0.5, not vote at 0.80).
    # Treating empty alltime/recent as {} keeps the loop's per-signal
    # sample-threshold logic authoritative.
    if not alltime:
        alltime = {}
    if not recent:
        recent = {}

    # P1-D (2026-04-17 adversarial review): iterate over the UNION of signal
    # names, not just alltime. Previously a signal present only in `recent`
    # was silently dropped, and directional keys (buy_accuracy, sell_accuracy,
    # total_buy, total_sell) were copied only from `at` - so a signal with
    # sell_accuracy=0.28 over 400 samples in recent but no alltime entry
    # silently passed the directional gate because total_sell defaulted to 0.
    accuracy_data = {}
    all_signal_names = set(alltime) | set(recent)
    for sig_name in all_signal_names:
        at = alltime.get(sig_name) or {}
        rc = recent.get(sig_name) or {}
        at_acc = at.get("accuracy", 0.5) if at else 0.5
        rc_acc = rc.get("accuracy", 0.5) if rc else 0.5
        rc_samples = rc.get("total", 0) if rc else 0
        at_samples = at.get("total", 0) if at else 0

        # Blend only when recent has enough samples AND alltime exists;
        # otherwise fall back to whichever source has data.
        # Codex round-10 P2 (2026-04-17 follow-up): previously a recent-only
        # signal with <min_recent_samples samples fell through to rc_acc,
        # letting an immature signal's raw recent accuracy drive consensus.
        # Now we require min_recent_samples even for recent-only signals,
        # falling back to a neutral 0.5 otherwise (matches pre-patch
        # semantics for signals below the recent-sample floor).
        if rc_samples >= min_recent_samples and at_samples > 0:
            divergence = abs(rc_acc - at_acc)
            w = fast_weight if divergence > divergence_threshold else normal_weight
            blended = w * rc_acc + (1 - w) * at_acc
        elif at_samples > 0:
            blended = at_acc
        elif rc_samples >= min_recent_samples:
            blended = rc_acc  # recent-only signal with enough samples
        else:
            blended = 0.5  # immature signal: neutral default

        total = max(at_samples, rc_samples)
        result = {
            "accuracy": blended,
            "total": total,
            "correct": int(round(blended * total)),  # BUG-186
            "pct": round(blended * 100, 1),
        }
        # Codex round 11 P2 (2026-04-17 follow-up): directional stats must
        # follow the same sample-floor rule as `accuracy`. Without this,
        # a recent-only signal with 20-29 one-sided votes still influenced
        # _weighted_consensus's directional gate/weighting at its raw
        # recent directional accuracy, even though the overall `accuracy`
        # field had already been set back to neutral 0.5. Omit directional
        # keys entirely for immature signals so downstream callers see the
        # `.get('buy_accuracy', acc)` fallback.
        _directionals_trustworthy = (
            at_samples > 0 or rc_samples >= min_recent_samples
        )
        if _directionals_trustworthy:
            # Merge directional keys from the larger-sample source per key.
            # Prevents silent gate-bypass when a key exists only in `recent`.
            for key in ("buy_accuracy", "sell_accuracy"):
                if key in at and key in rc:
                    side_total = "total_buy" if key == "buy_accuracy" else "total_sell"
                    at_side = at.get(side_total, 0) or 0
                    rc_side = rc.get(side_total, 0) or 0
                    result[key] = at[key] if at_side >= rc_side else rc[key]
                elif key in at:
                    result[key] = at[key]
                elif key in rc:
                    result[key] = rc[key]
            for key in ("total_buy", "total_sell"):
                at_v = at.get(key, 0) or 0
                rc_v = rc.get(key, 0) or 0
                if at_v or rc_v:
                    result[key] = max(at_v, rc_v)
        accuracy_data[sig_name] = result
    return accuracy_data


ACTIVATION_CACHE_TTL = 3600  # recompute hourly


def load_cached_activation_rates():
    """Load cached activation rates, recomputing if stale."""
    cache_file = DATA_DIR / "activation_cache.json"
    cache = load_json(cache_file)
    if cache is not None:
        try:
            if time.time() - cache.get("time", 0) < ACTIVATION_CACHE_TTL:
                return cache.get("rates", {})
        except (KeyError, AttributeError):
            logger.debug("Activation rates cache corrupted, regenerating")
    rates = signal_activation_rates()
    try:
        with _accuracy_write_lock:
            _atomic_write_json(cache_file, {"rates": rates, "time": time.time()})
    except Exception:
        logger.warning("Failed to write activation rates cache", exc_info=True)
    return rates


def load_cached_accuracy(horizon="1d"):
    cache = load_json(ACCURACY_CACHE_FILE)
    if cache is not None:
        try:
            # BUG-133: Use per-horizon timestamps to avoid cross-horizon staleness.
            # Fall back to legacy shared "time" key for backwards compatibility.
            ts = cache.get(f"time_{horizon}", cache.get("time", 0))
            if time.time() - ts < ACCURACY_CACHE_TTL:
                cached = cache.get(horizon)
                if cached:
                    return cached
        except (KeyError, AttributeError):
            logger.debug("Accuracy cache corrupted or missing horizon %s", horizon)
    return None


def write_accuracy_cache(horizon, data):
    with _accuracy_write_lock:
        cache = load_json(ACCURACY_CACHE_FILE, default={})
        if not isinstance(cache, dict):
            cache = {}
        cache[horizon] = data
        # BUG-133: Write per-horizon timestamp so other horizons don't appear fresh.
        cache[f"time_{horizon}"] = time.time()
        # Keep legacy "time" key for backwards compat with older code paths.
        cache["time"] = time.time()
        _atomic_write_json(ACCURACY_CACHE_FILE, cache)


# BUG-178 (2026-04-16) cache-miss wrappers. See _accuracy_compute_lock comment
# at the top of this module for the rationale. Callers that previously did
# `cached = load_cached_accuracy(h); if not cached: cached = signal_accuracy(h);
# write_accuracy_cache(h, cached)` should call these instead so the compute
# is done at most once across all racing ticker threads.

def get_or_compute_accuracy(horizon: str):
    """Return cached all-time accuracy, computing it once if cache is cold.

    Thread-safe via double-checked locking — first miss-thread computes,
    others wait on _accuracy_compute_lock and then read the populated cache.
    """
    cached = load_cached_accuracy(horizon)
    if cached:
        return cached
    with _accuracy_compute_lock:
        cached = load_cached_accuracy(horizon)
        if cached:
            return cached
        result = signal_accuracy(horizon)
        if result:
            write_accuracy_cache(horizon, result)
        return result


def get_or_compute_recent_accuracy(horizon: str, days: int = 7):
    """Cached recent-window (default 7d) accuracy, computed at most once."""
    cache_key = f"{horizon}_recent"
    cached = load_cached_accuracy(cache_key)
    if cached:
        return cached
    with _accuracy_compute_lock:
        cached = load_cached_accuracy(cache_key)
        if cached:
            return cached
        result = signal_accuracy_recent(horizon, days=days)
        if result:
            write_accuracy_cache(cache_key, result)
        return result


def get_or_compute_per_ticker_accuracy(horizon: str):
    """Cached per-ticker consensus accuracy, computed at most once.

    Cache key matches the BUG-164 lazy-populate convention used by
    signal_engine.py:_ptc_key.
    """
    cache_key = f"per_ticker_consensus_{horizon}"
    cached = load_cached_accuracy(cache_key)
    if cached:
        return cached
    with _accuracy_compute_lock:
        cached = load_cached_accuracy(cache_key)
        if cached:
            return cached
        result = per_ticker_accuracy(horizon)
        if result:
            write_accuracy_cache(cache_key, result)
        return result


def _count_entries_with_outcomes(entries, horizon):
    count = 0
    for entry in entries:
        outcomes = entry.get("outcomes", {})
        for _ticker, horizons in outcomes.items():
            if horizons.get(horizon):
                count += 1
                break
    return count


def print_accuracy_report():
    entries = load_entries()
    if not entries:
        print("No signal log data found.")
        return

    horizon_counts = {h: _count_entries_with_outcomes(entries, h) for h in HORIZONS}
    counts_str = ", ".join(f"{horizon_counts[h]} with {h} outcomes" for h in HORIZONS)

    print("=== Signal Accuracy Report ===")
    print()
    print(f"Entries: {len(entries)} total, {counts_str}")

    for h in HORIZONS:
        if horizon_counts[h] == 0:
            continue

        print()
        print(f"--- {h} Horizon ({horizon_counts[h]} entries with outcomes) ---")
        print()

        # ARCH-24: Pass pre-loaded entries to avoid re-reading 68MB file per call.
        sig_acc = signal_accuracy(h, entries=entries)
        sorted_sigs = sorted(
            SIGNAL_NAMES, key=lambda s: sig_acc[s]["accuracy"], reverse=True
        )

        print(f"{'Signal':<16}{'Correct':>7}  {'Total':>5}  {'Accuracy':>8}")
        print(f"{'------':<16}{'-------':>7}  {'-----':>5}  {'--------':>8}")

        for sig_name in sorted_sigs:
            s = sig_acc[sig_name]
            if s["total"] == 0:
                continue
            disabled_tag = " (OFF)" if sig_name in DISABLED_SIGNALS else ""
            print(
                f"{sig_name:<16}{s['correct']:>7}  {s['total']:>5}  {s['accuracy']*100:>7.1f}%{disabled_tag}"
            )

        cons = consensus_accuracy(h, entries=entries)
        print()
        if cons["total"] > 0:
            print(
                f"{'Consensus':<16}{cons['correct']:>7}  {cons['total']:>5}  {cons['accuracy']*100:>7.1f}%"
            )

        ticker_acc = per_ticker_accuracy(h, entries=entries)
        if ticker_acc:
            print()
            print("Per-Ticker:")
            sorted_tickers = sorted(
                ticker_acc.keys(), key=lambda t: ticker_acc[t]["accuracy"], reverse=True
            )
            for ticker in sorted_tickers:
                s = ticker_acc[ticker]
                print(
                    f"{ticker:<16}{s['correct']:>7}  {s['total']:>5}  {s['accuracy']*100:>7.1f}%"
                )


REGIME_ACCURACY_CACHE_FILE = DATA_DIR / "regime_accuracy_cache.json"


def signal_accuracy_by_regime(horizon="1d", since=None, entries=None):
    """Compute per-signal accuracy grouped by market regime.

    Args:
        horizon: Outcome horizon to evaluate ("1d", "3d", "5d", "10d").
        since: Optional ISO-8601 string cutoff. Only entries with ts >= since
               are included. None means all entries (no time filter).
        entries: Pre-loaded entries list. If None, loads from disk.

    Returns:
        dict: {regime: {signal_name: {correct, total, accuracy, pct}}}
              Only includes signals with total > 0.
    """
    if entries is None:
        entries = load_entries()

    # {regime: {signal_name: {correct, total}}}
    regime_stats = defaultdict(lambda: {s: {"correct": 0, "total": 0} for s in SIGNAL_NAMES})

    for entry in entries:
        if since and entry.get("ts", "") < since:
            continue
        outcomes = entry.get("outcomes", {})
        tickers = entry.get("tickers", {})

        for ticker, tdata in tickers.items():
            outcome = outcomes.get(ticker, {}).get(horizon)
            if not outcome:
                continue

            change_pct = outcome.get("change_pct", 0)
            signals = tdata.get("signals", {})
            regime = tdata.get("regime", "unknown")

            for sig_name in SIGNAL_NAMES:
                vote = signals.get(sig_name, "HOLD")
                if vote == "HOLD":
                    continue
                result_val = _vote_correct(vote, change_pct)
                if result_val is None:
                    continue  # neutral outcome — don't count
                regime_stats[regime][sig_name]["total"] += 1
                if result_val:
                    regime_stats[regime][sig_name]["correct"] += 1

    result = {}
    for regime, sig_map in regime_stats.items():
        regime_result = {}
        for sig_name, s in sig_map.items():
            if s["total"] == 0:
                continue
            acc = s["correct"] / s["total"]
            regime_result[sig_name] = {
                "correct": s["correct"],
                "total": s["total"],
                "accuracy": acc,
                "pct": round(acc * 100, 1),
            }
        if regime_result:
            result[regime] = regime_result

    return result


def load_cached_regime_accuracy(horizon="1d"):
    """Load cached regime accuracy, returning None if missing or stale.

    Uses the same TTL as the main accuracy cache (ACCURACY_CACHE_TTL).
    """
    cache = load_json(REGIME_ACCURACY_CACHE_FILE)
    if cache is not None:
        try:
            if time.time() - cache.get("time", 0) < ACCURACY_CACHE_TTL:
                cached = cache.get(horizon)
                if cached:
                    return cached
        except (KeyError, AttributeError):
            logger.debug("Regime accuracy cache corrupted or missing horizon %s", horizon)
    return None


def write_regime_accuracy_cache(horizon, data):
    """Persist regime accuracy data to the cache file.

    Merges with any existing horizons to avoid overwriting other cached data.
    """
    with _accuracy_write_lock:
        cache = load_json(REGIME_ACCURACY_CACHE_FILE, default={})
        if not isinstance(cache, dict):
            cache = {}
        cache[horizon] = data
        cache["time"] = time.time()
        _atomic_write_json(REGIME_ACCURACY_CACHE_FILE, cache)


ACCURACY_SNAPSHOTS_FILE = DATA_DIR / "accuracy_snapshots.jsonl"


def save_accuracy_snapshot(extras=None):
    """Save current per-signal accuracy as a timestamped snapshot.

    Appends one JSON line to accuracy_snapshots.jsonl with the current
    accuracy for each signal at the 1d horizon. Used by check_accuracy_changes()
    and accuracy_degradation.check_degradation() to detect significant shifts
    over time.

    Args:
        extras: Optional dict of extra blocks to merge into the snapshot.
            BUG-178/W15-W16 follow-up (2026-04-16): the degradation tracker
            needs more than the lifetime per-signal block. Callers can pass
            recent-window per-signal accuracy, per-ticker per-signal blocks,
            forecast model accuracy, and aggregate consensus accuracy by
            providing keys like "signals_recent", "per_ticker", "forecast",
            "consensus", etc. Unknown keys are accepted as-is so future
            scopes can be added without churning this function. Old single-
            block snapshots remain readable — the loader treats absent
            keys as missing.
    """
    from datetime import datetime

    acc = signal_accuracy("1d")
    snapshot = {
        "ts": datetime.now(UTC).isoformat(),
        "signals": {
            name: {"accuracy": data["accuracy"], "total": data["total"]}
            for name, data in acc.items()
        },
    }
    if extras:
        for key, value in extras.items():
            snapshot[key] = value
    from portfolio.file_utils import atomic_append_jsonl
    atomic_append_jsonl(ACCURACY_SNAPSHOTS_FILE, snapshot)
    return snapshot




def _load_accuracy_snapshots():
    """Load all accuracy snapshots from JSONL file."""
    if not ACCURACY_SNAPSHOTS_FILE.exists():
        return []
    entries = []
    for line in ACCURACY_SNAPSHOTS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _find_snapshot_near(snapshots, target_ts, max_delta_hours=36):
    """Find the snapshot closest to target_ts within max_delta_hours.

    Args:
        snapshots: List of snapshot dicts with 'ts' field.
        target_ts: datetime to search near.
        max_delta_hours: Maximum allowed time difference in hours.

    Returns:
        dict or None: The closest snapshot, or None if none within range.
    """
    from datetime import datetime

    best = None
    best_delta = None
    for snap in snapshots:
        try:
            snap_ts = datetime.fromisoformat(snap["ts"])
            delta = abs((snap_ts - target_ts).total_seconds()) / 3600
            if delta <= max_delta_hours and (best_delta is None or delta < best_delta):
                    best = snap
                    best_delta = delta
        except (ValueError, TypeError, KeyError):
            continue
    return best


def check_accuracy_changes(threshold_drop=0.1, threshold_rise=0.1):
    """Check for significant accuracy changes vs 7 days ago.

    Compares current signal accuracy against the snapshot closest to 7 days ago.
    Returns alerts for signals whose accuracy changed by more than the thresholds.

    Args:
        threshold_drop: Minimum accuracy drop (as fraction, e.g. 0.1 = 10pp) to alert.
        threshold_rise: Minimum accuracy rise (as fraction, e.g. 0.1 = 10pp) to alert.

    Returns:
        list[dict]: List of alert dicts with keys:
            signal, old_accuracy, new_accuracy, change, direction ("dropped"/"rose"),
            old_samples, new_samples.
        Empty list if no significant changes or no historical snapshot available.
    """
    from datetime import datetime, timedelta

    snapshots = _load_accuracy_snapshots()
    if not snapshots:
        return []

    now = datetime.now(UTC)
    target = now - timedelta(days=7)
    old_snapshot = _find_snapshot_near(snapshots, target)

    if old_snapshot is None:
        return []

    # Compute current accuracy
    current_acc = signal_accuracy("1d")
    old_signals = old_snapshot.get("signals", {})

    alerts = []
    for sig_name in SIGNAL_NAMES:
        old_data = old_signals.get(sig_name)
        new_data = current_acc.get(sig_name)

        if not old_data or not new_data:
            continue

        # Require minimum samples for meaningful comparison
        if old_data.get("total", 0) < 10 or new_data.get("total", 0) < 10:
            continue

        old_acc = old_data.get("accuracy", 0.0)
        new_acc = new_data.get("accuracy", 0.0)
        change = new_acc - old_acc

        if change <= -threshold_drop:
            alerts.append({
                "signal": sig_name,
                "old_accuracy": round(old_acc * 100, 1),
                "new_accuracy": round(new_acc * 100, 1),
                "change": round(change * 100, 1),
                "direction": "dropped",
                "old_samples": old_data.get("total", 0),
                "new_samples": new_data.get("total", 0),
            })
        elif change >= threshold_rise:
            alerts.append({
                "signal": sig_name,
                "old_accuracy": round(old_acc * 100, 1),
                "new_accuracy": round(new_acc * 100, 1),
                "change": round(change * 100, 1),
                "direction": "rose",
                "old_samples": old_data.get("total", 0),
                "new_samples": new_data.get("total", 0),
            })

    # Sort by absolute change magnitude, largest first
    alerts.sort(key=lambda a: abs(a["change"]), reverse=True)
    return alerts


def format_accuracy_alerts(alerts):
    """Format accuracy change alerts as human-readable strings.

    Args:
        alerts: List of alert dicts from check_accuracy_changes().

    Returns:
        list[str]: Formatted alert strings.
    """
    lines = []
    for a in alerts:
        lines.append(
            f"{a['signal']} accuracy {a['direction']} from "
            f"{a['old_accuracy']}% to {a['new_accuracy']}% "
            f"({a['change']:+.1f}pp, {a['new_samples']} samples)"
        )
    return lines


def signal_best_horizon_accuracy(min_samples=50, entries=None):
    """Compute each signal's best accuracy across all horizons.

    For each signal, evaluates accuracy at every horizon in HORIZONS and returns
    the horizon with the highest accuracy (provided it meets the minimum sample
    threshold). This allows the signal weighting system to use the most
    predictive horizon for each signal rather than a fixed 1d window.

    Results are cached in BEST_HORIZON_CACHE_FILE with the same TTL as the
    main accuracy cache (ACCURACY_CACHE_TTL).

    Args:
        min_samples: Minimum number of datapoints required for a horizon to be
            considered. Horizons below this threshold are skipped. Default 50.

    Returns:
        dict: {signal_name: {accuracy, total, correct, pct, best_horizon}}
        Signals with no qualifying horizons are omitted from the result.
    """
    # --- Cache check ---
    cached = load_json(BEST_HORIZON_CACHE_FILE)
    if cached is not None and isinstance(cached, dict):
        try:
            if time.time() - cached.get("time", 0) < ACCURACY_CACHE_TTL:
                data = cached.get("data")
                if isinstance(data, dict):
                    return data
        except (KeyError, TypeError):
            pass

    if entries is None:
        entries = load_entries()
    # {sig_name: {horizon: {correct, total}}}
    stats: dict[str, dict[str, dict[str, int]]] = {
        s: {h: {"correct": 0, "total": 0} for h in HORIZONS}
        for s in SIGNAL_NAMES
    }

    for entry in entries:
        outcomes = entry.get("outcomes", {})
        tickers = entry.get("tickers", {})

        for ticker, tdata in tickers.items():
            signals = tdata.get("signals", {})
            for horizon in HORIZONS:
                outcome = outcomes.get(ticker, {}).get(horizon)
                if not outcome:
                    continue
                change_pct = outcome.get("change_pct", 0)

                for sig_name in SIGNAL_NAMES:
                    vote = signals.get(sig_name, "HOLD")
                    if vote == "HOLD":
                        continue
                    result_val = _vote_correct(vote, change_pct)
                    if result_val is None:
                        continue
                    stats[sig_name][horizon]["total"] += 1
                    if result_val:
                        stats[sig_name][horizon]["correct"] += 1

    result = {}
    for sig_name in SIGNAL_NAMES:
        best_hz = None
        best_acc = -1.0
        best_total = 0
        best_correct = 0

        for horizon in HORIZONS:
            h_stats = stats[sig_name][horizon]
            total = h_stats["total"]
            if total < min_samples:
                continue
            acc = h_stats["correct"] / total
            if acc > best_acc:
                best_acc = acc
                best_hz = horizon
                best_total = total
                best_correct = h_stats["correct"]

        if best_hz is not None:
            result[sig_name] = {
                "accuracy": best_acc,
                "total": best_total,
                "correct": best_correct,
                "pct": round(best_acc * 100, 1),
                "best_horizon": best_hz,
            }

    # --- Write cache ---
    try:
        with _accuracy_write_lock:
            _atomic_write_json(BEST_HORIZON_CACHE_FILE, {"time": time.time(), "data": result})
    except Exception:
        logger.debug("Failed to write best_horizon cache", exc_info=True)

    return result


def accuracy_by_ticker_signal(horizon="1d", min_samples=0):
    """Compute per-ticker per-signal accuracy cross-tabulation.

    Returns nested dict: {ticker: {signal_name: {correct, total, accuracy, pct,
        correct_buy, total_buy, buy_accuracy, correct_sell, total_sell, sell_accuracy}}}
    Only includes signals that voted BUY or SELL (HOLD excluded).
    Directional fields (buy_accuracy, sell_accuracy) enable per-ticker directional
    gating in signal_engine._weighted_consensus().

    Args:
        horizon: Outcome horizon ("1d", "3d", "5d", "10d").
        min_samples: Minimum votes required to include a signal for a ticker.
    """
    entries = load_entries()
    # {ticker: {signal: {correct, total, correct_buy, total_buy, correct_sell, total_sell}}}
    def _empty():
        return {"correct": 0, "total": 0, "correct_buy": 0, "total_buy": 0,
                "correct_sell": 0, "total_sell": 0}
    stats = defaultdict(lambda: defaultdict(_empty))

    for entry in entries:
        outcomes = entry.get("outcomes", {})
        tickers = entry.get("tickers", {})

        for ticker, tdata in tickers.items():
            outcome = outcomes.get(ticker, {}).get(horizon)
            if not outcome:
                continue

            change_pct = outcome.get("change_pct", 0)
            signals = tdata.get("signals", {})

            for sig_name, vote in signals.items():
                if vote == "HOLD":
                    continue
                result_val = _vote_correct(vote, change_pct)
                if result_val is None:
                    continue
                s = stats[ticker][sig_name]
                s["total"] += 1
                if vote == "BUY":
                    s["total_buy"] += 1
                    if result_val:
                        s["correct"] += 1
                        s["correct_buy"] += 1
                else:
                    s["total_sell"] += 1
                    if result_val:
                        s["correct"] += 1
                        s["correct_sell"] += 1

    result = {}
    for ticker, sig_stats in stats.items():
        ticker_result = {}
        for sig_name, s in sig_stats.items():
            if s["total"] < min_samples:
                continue
            acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
            buy_acc = s["correct_buy"] / s["total_buy"] if s["total_buy"] > 0 else 0.0
            sell_acc = s["correct_sell"] / s["total_sell"] if s["total_sell"] > 0 else 0.0
            ticker_result[sig_name] = {
                "correct": s["correct"],
                "total": s["total"],
                "accuracy": acc,
                "pct": round(acc * 100, 1),
                "correct_buy": s["correct_buy"],
                "total_buy": s["total_buy"],
                "buy_accuracy": round(buy_acc, 4),
                "correct_sell": s["correct_sell"],
                "total_sell": s["total_sell"],
                "sell_accuracy": round(sell_acc, 4),
            }
        if ticker_result:
            result[ticker] = ticker_result

    return result


def top_signals_for_ticker(ticker, horizon="1d", min_samples=5):
    """Return ranked list of signals for a specific ticker, sorted by accuracy.

    Args:
        ticker: Ticker symbol (e.g. "BTC-USD").
        horizon: Outcome horizon.
        min_samples: Minimum votes to qualify.

    Returns:
        list[dict]: Sorted by accuracy descending. Each dict has:
            signal, correct, total, accuracy, pct.
    """
    all_data = accuracy_by_ticker_signal(horizon, min_samples=min_samples)
    ticker_data = all_data.get(ticker, {})
    ranked = [
        {"signal": sig, **data}
        for sig, data in ticker_data.items()
    ]
    ranked.sort(key=lambda x: x["accuracy"], reverse=True)
    return ranked


# ---------------------------------------------------------------------------
# Per-ticker accuracy cache
# ---------------------------------------------------------------------------

TICKER_ACCURACY_CACHE_FILE = DATA_DIR / "ticker_signal_accuracy_cache.json"


def load_cached_ticker_accuracy(horizon="1d"):
    """Load cached per-ticker per-signal accuracy, returning None if stale.

    Uses the same TTL as the main accuracy cache (ACCURACY_CACHE_TTL).
    Cache structure: {horizon: {ticker: {signal: {correct, total, accuracy, pct}}}, "time": ...}
    """
    cache = load_json(TICKER_ACCURACY_CACHE_FILE)
    if cache is not None:
        try:
            if time.time() - cache.get("time", 0) < ACCURACY_CACHE_TTL:
                cached = cache.get(horizon)
                if cached:
                    return cached
        except (KeyError, AttributeError):
            logger.debug("Ticker accuracy cache corrupted or missing horizon %s", horizon)
    return None


def write_ticker_accuracy_cache(horizon, data):
    """Persist per-ticker per-signal accuracy data to the cache file.

    Merges with existing horizons to avoid overwriting other cached data.
    """
    with _accuracy_write_lock:
        cache = load_json(TICKER_ACCURACY_CACHE_FILE, default={})
        if not isinstance(cache, dict):
            cache = {}
        cache[horizon] = data
        cache["time"] = time.time()
        _atomic_write_json(TICKER_ACCURACY_CACHE_FILE, cache)


def _filter_min_samples(data, min_samples):
    if min_samples <= 0:
        return data
    return {
        ticker: {
            sig: sdata for sig, sdata in sigs.items()
            if sdata.get("total", 0) >= min_samples
        }
        for ticker, sigs in data.items()
    }


def accuracy_by_ticker_signal_cached(horizon="1d", min_samples=0):
    """Cached version of accuracy_by_ticker_signal().

    Checks the ticker accuracy cache first; on miss, computes from the
    full signal log and writes the cache. BUG-178 (2026-04-16): the
    cache-miss compute path is now serialized via _accuracy_compute_lock
    so concurrent ticker threads can't all redundantly walk the 50,000-
    entry signal log when the 1h TTL expires.
    """
    cached = load_cached_ticker_accuracy(horizon)
    if cached:
        return _filter_min_samples(cached, min_samples)

    with _accuracy_compute_lock:
        # Re-check after acquiring the lock — another thread may have
        # populated the cache while we waited.
        cached = load_cached_ticker_accuracy(horizon)
        if cached:
            return _filter_min_samples(cached, min_samples)

        data = accuracy_by_ticker_signal(horizon, min_samples=0)
        if data:
            write_ticker_accuracy_cache(horizon, data)
        return _filter_min_samples(data, min_samples)


def probability_calibration(horizon="1d", buckets=None, since=None):
    """Compute calibration data for reliability diagrams.

    Groups consensus predictions by confidence bucket and computes actual
    accuracy per bucket. Confidence = max(buy, sell) / (buy + sell).

    Args:
        horizon: Outcome horizon to evaluate ("1d", "3d", "5d", "10d").
        buckets: List of bucket boundaries. Defaults to [0.5, 0.6, 0.7, 0.8, 0.9, 1.01].
        since: Optional ISO-8601 string cutoff. Only entries with ts >= since
               are included. None means all entries.

    Returns:
        list[dict]: One dict per bucket with keys: bucket_low, bucket_high,
        predicted_confidence, actual_accuracy, sample_count, correct_count.
    """
    if buckets is None:
        buckets = [0.5, 0.6, 0.7, 0.8, 0.9, 1.01]  # 1.01 to include 1.0

    entries = load_entries()
    # Build list of (confidence, correct_bool) tuples
    samples = []
    for entry in entries:
        if since and entry.get("ts", "") < since:
            continue
        outcomes = entry.get("outcomes") or {}
        tickers = entry.get("tickers") or {}
        for ticker, tdata in tickers.items():
            consensus = tdata.get("consensus")
            if consensus not in ("BUY", "SELL"):
                continue
            buy_c = int(tdata.get("buy_count") or 0)
            sell_c = int(tdata.get("sell_count") or 0)
            total = buy_c + sell_c
            if total < 1:
                continue
            confidence = max(buy_c, sell_c) / total

            outcome = (outcomes.get(ticker) or {}).get(horizon)
            if outcome is None:
                continue
            change_pct = outcome.get("change_pct")
            if change_pct is None or abs(change_pct) < _MIN_CHANGE_PCT:
                continue

            correct = (consensus == "BUY" and change_pct > 0) or \
                      (consensus == "SELL" and change_pct < 0)
            samples.append((confidence, correct))

    # Bucket the samples
    result = []
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i + 1]
        bucket_samples = [(c, correct) for c, correct in samples if lo <= c < hi]
        if not bucket_samples:
            result.append({
                "bucket_low": lo, "bucket_high": hi,
                "predicted_confidence": (lo + hi) / 2,
                "actual_accuracy": None, "sample_count": 0, "correct_count": 0,
            })
            continue
        correct_count = sum(1 for _, c in bucket_samples if c)
        n = len(bucket_samples)
        avg_conf = sum(c for c, _ in bucket_samples) / n
        result.append({
            "bucket_low": lo, "bucket_high": hi,
            "predicted_confidence": round(avg_conf, 4),
            "actual_accuracy": round(correct_count / n, 4),
            "sample_count": n, "correct_count": correct_count,
        })
    return result


if __name__ == "__main__":
    print_accuracy_report()

    # Also show accuracy changes if snapshots exist
    alerts = check_accuracy_changes()
    if alerts:
        print()
        print("=== Accuracy Changes (vs 7 days ago) ===")
        print()
        for line in format_accuracy_alerts(alerts):
            print(f"  {line}")
    else:
        print()
        print("No significant accuracy changes detected (or no 7-day snapshot available).")
