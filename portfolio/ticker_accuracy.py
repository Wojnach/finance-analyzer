"""Per-ticker per-signal accuracy and directional probability engine.

Computes P(up) at multiple horizons for focus instruments using accuracy-weighted
signal votes. This is the core engine for Mode B probability-based notifications.
"""

import logging
import math
from collections import defaultdict

from portfolio.tickers import SIGNAL_NAMES, DISABLED_SIGNALS

logger = logging.getLogger("portfolio.ticker_accuracy")


def accuracy_by_ticker_signal(ticker, horizon="1d", days=None):
    """Per-signal accuracy for a specific ticker.

    Queries signal_log data (SQLite preferred, JSONL fallback) and computes
    accuracy grouped by signal name for a single ticker.

    Args:
        ticker: Instrument ticker (e.g., "XAG-USD").
        horizon: Outcome horizon ("3h", "1d", "3d", "5d", "10d").
        days: If set, only include entries from the last N days.

    Returns:
        dict: {signal_name: {"accuracy": float, "samples": int, "correct": int}}
    """
    from portfolio.accuracy_stats import load_entries

    entries = load_entries()

    cutoff = None
    if days is not None:
        from datetime import datetime, timedelta, timezone
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    stats = {s: {"correct": 0, "total": 0} for s in SIGNAL_NAMES}

    for entry in entries:
        if cutoff and entry.get("ts", "") < cutoff:
            continue

        tdata = entry.get("tickers", {}).get(ticker)
        if not tdata:
            continue

        outcome = entry.get("outcomes", {}).get(ticker, {}).get(horizon)
        if not outcome:
            continue

        change_pct = outcome.get("change_pct", 0)
        signals = tdata.get("signals", {})

        for sig_name in SIGNAL_NAMES:
            vote = signals.get(sig_name, "HOLD")
            if vote == "HOLD":
                continue
            stats[sig_name]["total"] += 1
            if (vote == "BUY" and change_pct > 0) or (vote == "SELL" and change_pct < 0):
                stats[sig_name]["correct"] += 1

    result = {}
    for sig_name in SIGNAL_NAMES:
        s = stats[sig_name]
        if s["total"] == 0:
            continue
        result[sig_name] = {
            "accuracy": s["correct"] / s["total"],
            "samples": s["total"],
            "correct": s["correct"],
        }
    return result


def direction_probability(ticker, current_votes, horizon="1d", days=7, min_samples=5):
    """Convert signal votes + per-ticker accuracy into P(up) at given horizon.

    Algorithm:
        For each active signal voting BUY or SELL for this ticker:
          acc = that signal's accuracy for THIS ticker at THIS horizon
          if signal says BUY:  p_up = acc       (71% accurate BUY → 71% P(up))
          if signal says SELL: p_up = 1 - acc    (71% accurate SELL → 29% P(up))
        Weighted average of all p_up values (weight = sqrt(sample_count))

    Args:
        ticker: Instrument ticker.
        current_votes: dict {signal_name: "BUY"|"SELL"|"HOLD"} for current cycle.
        horizon: Outcome horizon for accuracy lookup.
        days: Lookback window for accuracy computation (None = all time).
        min_samples: Minimum samples required to use a signal's accuracy.

    Returns:
        dict: {
            "direction": "up"|"down"|"neutral",
            "probability": float (0.0-1.0),
            "signals_used": int,
            "total_samples": int,
            "signal_details": [{name, vote, accuracy, samples, p_up, weight}]
        }
    """
    acc_data = accuracy_by_ticker_signal(ticker, horizon=horizon, days=days)

    weighted_sum = 0.0
    weight_total = 0.0
    signals_used = 0
    total_samples = 0
    details = []

    for sig_name, vote in current_votes.items():
        if vote == "HOLD" or sig_name in DISABLED_SIGNALS:
            continue

        sig_acc = acc_data.get(sig_name)
        if not sig_acc or sig_acc["samples"] < min_samples:
            continue

        accuracy = sig_acc["accuracy"]
        samples = sig_acc["samples"]

        # Compute P(up) based on vote direction and accuracy
        if vote == "BUY":
            p_up = accuracy
        else:  # SELL
            p_up = 1.0 - accuracy

        weight = math.sqrt(samples)
        weighted_sum += p_up * weight
        weight_total += weight
        signals_used += 1
        total_samples += samples

        details.append({
            "name": sig_name,
            "vote": vote,
            "accuracy": round(accuracy, 3),
            "samples": samples,
            "p_up": round(p_up, 3),
            "weight": round(weight, 2),
        })

    if weight_total == 0:
        return {
            "direction": "neutral",
            "probability": 0.5,
            "signals_used": 0,
            "total_samples": 0,
            "signal_details": [],
        }

    p_up_final = weighted_sum / weight_total

    direction = "neutral"
    if p_up_final > 0.52:
        direction = "up"
    elif p_up_final < 0.48:
        direction = "down"

    return {
        "direction": direction,
        "probability": round(p_up_final, 3),
        "signals_used": signals_used,
        "total_samples": total_samples,
        "signal_details": details,
    }


def get_focus_probabilities(tickers, current_data, horizons=None, days=7):
    """Compute probabilities for focus instruments across multiple horizons.

    Args:
        tickers: List of focus ticker names (e.g., ["XAG-USD", "BTC-USD"]).
        current_data: dict {ticker: {signals dict from signal_engine output}}.
            Each ticker entry needs an "extra" dict with "_votes" or
            a "signals" dict mapping signal_name -> vote.
        horizons: List of horizon strings. Defaults to ["3h", "1d", "3d"].
        days: Lookback window for accuracy.

    Returns:
        dict: {
            ticker: {
                "3h": {"direction": "up", "probability": 0.72, ...},
                "1d": {"direction": "up", "probability": 0.68, ...},
                "3d": {"direction": "up", "probability": 0.55, ...},
                "accuracy_1d": 0.71,
                "accuracy_samples": 89,
            }
        }
    """
    if horizons is None:
        horizons = ["3h", "1d", "3d"]

    result = {}

    for ticker in tickers:
        ticker_data = current_data.get(ticker, {})

        # Extract current votes from the signal data
        votes = _extract_votes(ticker_data)
        if not votes:
            continue

        ticker_result = {}
        for h in horizons:
            prob = direction_probability(ticker, votes, horizon=h, days=days)
            ticker_result[h] = prob

        # Overall accuracy summary (1d horizon as primary)
        acc_1d = accuracy_by_ticker_signal(ticker, horizon="1d", days=days)
        total_samples = sum(s["samples"] for s in acc_1d.values())
        if acc_1d:
            # Weighted average accuracy across active signals
            weighted_acc = 0.0
            weight_sum = 0.0
            for sig_name, sig_data in acc_1d.items():
                if sig_name in DISABLED_SIGNALS:
                    continue
                w = math.sqrt(sig_data["samples"])
                weighted_acc += sig_data["accuracy"] * w
                weight_sum += w
            if weight_sum > 0:
                ticker_result["accuracy_1d"] = round(weighted_acc / weight_sum, 3)
            else:
                ticker_result["accuracy_1d"] = 0.0
        else:
            ticker_result["accuracy_1d"] = 0.0

        ticker_result["accuracy_samples"] = total_samples
        result[ticker] = ticker_result

    return result


def _extract_votes(ticker_data):
    """Extract signal votes from various data formats.

    Handles:
    - {"extra": {"_votes": {signal: vote}}} (from signal_engine)
    - {"signals": {signal: vote}} (from signal_log)
    - Direct {signal: vote} dict
    """
    # From signal_engine output
    extra = ticker_data.get("extra", {})
    votes = extra.get("_votes")
    if votes:
        return votes

    # From signal_log format
    signals = ticker_data.get("signals", {})
    if signals and any(v in ("BUY", "SELL") for v in signals.values()):
        return signals

    # Direct dict
    if any(k in SIGNAL_NAMES for k in ticker_data):
        return {k: v for k, v in ticker_data.items() if k in SIGNAL_NAMES}

    return {}
