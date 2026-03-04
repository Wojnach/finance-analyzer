"""
Metals Signal Tracker — per-loop accuracy logging.

Logs a full signal snapshot every loop iteration to data/metals_signal_log.jsonl.
Backfills outcomes at 1h/3h horizons to track accuracy of each signal source.
Provides rolling accuracy stats for the loop status display and Claude context.

Usage from metals_loop.py:
    from metals_signal_tracker import log_snapshot, backfill_outcomes, get_accuracy_report, get_accuracy_summary

    # Every loop iteration:
    log_snapshot(check_count, prices, positions, signal_data, llm_signals, triggered, reasons)

    # Every 10th check:
    backfill_outcomes(current_underlying_prices)

    # For status display:
    summary = get_accuracy_summary()  # "main_XAG:72%(18) chrono_XAG_1h:55%(11)"
"""

import json
import os
import sys
import time
import datetime
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from portfolio.file_utils import atomic_write_json, load_json, atomic_append_jsonl

SIGNAL_LOG = "data/metals_signal_log.jsonl"
OUTCOMES_LOG = "data/metals_signal_outcomes.jsonl"
ACCURACY_CACHE_FILE = "data/metals_signal_accuracy.json"

# Horizons for outcome checking (seconds)
HORIZONS = {
    "1h": 3600,
    "3h": 10800,
}

# Rolling window for accuracy calculation
ACCURACY_WINDOW = 200

# Max entries to scan during backfill (keeps I/O bounded)
BACKFILL_SCAN_LIMIT = 400

# Thread safety
_lock = threading.Lock()
_accuracy_cache = {}  # last computed accuracy stats


def log_snapshot(check_count, prices, positions, signal_data, llm_signals,
                 triggered, trigger_reasons, llm_accuracy=None):
    """Log a full signal snapshot for this loop iteration.

    Args:
        check_count: current loop check number
        prices: {position_key: {bid, ask, underlying, ...}}
        positions: {position_key: {entry, stop, active, units, ...}}
        signal_data: dict from read_signal_data() — main loop signals
        llm_signals: dict from get_llm_signals() — Chronos/Ministral
        triggered: bool — whether triggers fired this check
        trigger_reasons: list of trigger reason strings
        llm_accuracy: dict from get_llm_accuracy() — optional
    """
    now = datetime.datetime.now(datetime.timezone.utc)

    entry = {
        "ts": now.isoformat(),
        "check": check_count,
        "prices": {},
        "signals": {},
        "llm": {},
        "positions": {},
        "triggered": triggered,
        "trigger_reasons": trigger_reasons[:5] if trigger_reasons else [],
    }

    # Prices: warrant bids + underlying USD
    for key, p in (prices or {}).items():
        if isinstance(p, dict):
            entry["prices"][key] = round(p.get("bid", 0), 4)
            und = p.get("underlying")
            if und and und > 0:
                entry["prices"][f"{key}_und"] = round(und, 4)
                # Map to standard ticker names for backfill matching
                if "silver" in key.lower():
                    entry["prices"]["XAG-USD"] = round(und, 4)
                elif "gold" in key.lower():
                    entry["prices"]["XAU-USD"] = round(und, 4)

    # Also store crypto prices directly if present in signal_data
    for crypto_ticker in ["BTC-USD", "ETH-USD"]:
        if signal_data and crypto_ticker in signal_data:
            sig_price = signal_data[crypto_ticker].get("price")
            if sig_price and sig_price > 0:
                entry["prices"][crypto_ticker] = round(sig_price, 4)

    # Main loop signals (from agent_summary.json)
    if signal_data:
        for ticker in ["XAG-USD", "XAU-USD", "BTC-USD", "ETH-USD"]:
            if ticker in signal_data:
                s = signal_data[ticker]
                sig_entry = {
                    "action": s.get("action", "?"),
                    "confidence": round(s.get("confidence", 0), 3),
                    "w_confidence": round(s.get("weighted_confidence", 0), 3),
                    "buy_count": s.get("buy_count", 0),
                    "sell_count": s.get("sell_count", 0),
                    "voters": s.get("voters", 0),
                    "rsi": s.get("rsi"),
                    "regime": s.get("regime", "?"),
                }
                # Parse per-signal votes for individual accuracy tracking
                vote_detail = s.get("vote_detail", "")
                if vote_detail:
                    buy_signals, sell_signals = [], []
                    for part in vote_detail.split("|"):
                        part = part.strip()
                        if part.startswith("B:"):
                            buy_signals = [x.strip() for x in part[2:].split(",") if x.strip()]
                        elif part.startswith("S:"):
                            sell_signals = [x.strip() for x in part[2:].split(",") if x.strip()]
                    sig_entry["_buy_signals"] = buy_signals
                    sig_entry["_sell_signals"] = sell_signals
                entry["signals"][ticker] = sig_entry

    # LLM signals (Chronos/Ministral)
    if llm_signals:
        for ticker, data in llm_signals.items():
            llm_entry = {}
            consensus = data.get("consensus", {})
            if consensus:
                llm_entry["consensus_dir"] = consensus.get("direction", "flat")
                llm_entry["consensus_conf"] = round(consensus.get("confidence", 0), 3)
                llm_entry["consensus_action"] = consensus.get("weighted_action", "HOLD")

            # Store prediction price for deviation tracking
            pred_price = data.get("price", 0)
            if pred_price and pred_price > 0:
                llm_entry["pred_price"] = round(pred_price, 4)

            # Individual model predictions — include pct_move for deviation tracking
            if data.get("ministral"):
                m = data["ministral"]
                llm_entry["ministral"] = m.get("action", "HOLD")
                llm_entry["ministral_conf"] = round(m.get("confidence", 0), 3)
            for h in ["1h", "3h"]:
                key = f"chronos_{h}"
                if data.get(key):
                    c = data[key]
                    llm_entry[f"chronos_{h}"] = c.get("direction", "flat")
                    llm_entry[f"chronos_{h}_conf"] = round(c.get("confidence", 0), 3)
                    # Chronos provides predicted pct_move — store for deviation tracking
                    pct_move = c.get("pct_move", 0)
                    if pct_move:
                        llm_entry[f"chronos_{h}_pct_move"] = round(pct_move, 4)

            if llm_entry:
                entry["llm"][ticker] = llm_entry

    # Position state
    for key, pos in (positions or {}).items():
        p = (prices or {}).get(key, {})
        bid = p.get("bid", 0) if isinstance(p, dict) else 0
        pos_entry = {
            "active": pos.get("active", False),
        }
        if pos.get("active") and bid > 0:
            pos_entry["pnl_pct"] = round(((bid - pos["entry"]) / pos["entry"]) * 100, 2)
            dist_stop = ((bid - pos["stop"]) / bid * 100) if bid > 0 else 999
            pos_entry["dist_stop_pct"] = round(dist_stop, 2)
            pos_entry["bid"] = round(bid, 4)
        entry["positions"][key] = pos_entry

    # Write to log
    try:
        atomic_append_jsonl(SIGNAL_LOG, entry)
    except Exception as e:
        print(f"[TRACKER] log_snapshot error: {e}", flush=True)


def _load_resolved_keys():
    """Load set of already-resolved (snapshot_ts, horizon) pairs from outcomes file."""
    resolved = set()
    if not os.path.exists(OUTCOMES_LOG):
        return resolved
    try:
        with open(OUTCOMES_LOG, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    key = (entry.get("snapshot_ts", ""), entry.get("horizon", ""))
                    resolved.add(key)
                except (json.JSONDecodeError, KeyError):
                    pass
    except Exception as e:
        print(f"[TRACKER] outcomes load error: {e}", flush=True)
    return resolved


def _resolve_outcome(entry, h_key, current_underlying_prices, now):
    """Compute outcome for a single snapshot + horizon. Returns outcome dict or None."""
    h_secs = HORIZONS[h_key]

    try:
        ts = datetime.datetime.fromisoformat(entry["ts"])
        epoch = ts.timestamp()
    except (KeyError, ValueError):
        return None

    elapsed = now - epoch
    if elapsed < h_secs * 0.9:  # 10% tolerance
        return None

    entry_prices = entry.get("prices", {})
    outcomes = {}

    for ticker in ["XAG-USD", "XAU-USD", "BTC-USD", "ETH-USD"]:
        current_price = current_underlying_prices.get(ticker, 0)
        if current_price <= 0:
            continue

        pred_price = entry_prices.get(ticker, 0)
        if pred_price <= 0:
            for pk, pv in entry_prices.items():
                if pk.endswith("_und") and ticker.lower().startswith("xag") and "silver" in pk:
                    pred_price = pv
                    break
                elif pk.endswith("_und") and ticker.lower().startswith("xau") and "gold" in pk:
                    pred_price = pv
                    break

        if pred_price <= 0:
            continue

        actual_dir = "up" if current_price > pred_price else "down"
        move_pct = ((current_price - pred_price) / pred_price) * 100

        outcome = {
            "price_then": round(pred_price, 4),
            "price_now": round(current_price, 4),
            "actual_dir": actual_dir,
            "move_pct": round(move_pct, 3),
        }

        signal_info = entry.get("signals", {}).get(ticker, {})
        main_action = signal_info.get("action", "?")
        if main_action in ("BUY", "SELL"):
            predicted_dir = "up" if main_action == "BUY" else "down"
            outcome["main_predicted"] = main_action
            outcome["main_correct"] = (predicted_dir == actual_dir)
            outcome["main_actual_move_pct"] = round(move_pct, 4)

        llm_info = entry.get("llm", {}).get(ticker, {})
        llm_dir = llm_info.get("consensus_dir", "flat")
        if llm_dir in ("up", "down"):
            outcome["llm_predicted"] = llm_dir
            outcome["llm_correct"] = (llm_dir == actual_dir)
            outcome["llm_actual_move_pct"] = round(move_pct, 4)

        for model in ["chronos_1h", "chronos_3h"]:
            m_dir = llm_info.get(model, "flat")
            if m_dir in ("up", "down"):
                outcome[f"{model}_predicted"] = m_dir
                outcome[f"{model}_correct"] = (m_dir == actual_dir)
                outcome[f"{model}_actual_move_pct"] = round(move_pct, 4)

                pred_pct = llm_info.get(f"{model}_pct_move", 0)
                if pred_pct:
                    signed_pred = abs(pred_pct) if m_dir == "up" else -abs(pred_pct)
                    error = signed_pred - move_pct
                    abs_error = abs(error)
                    outcome[f"{model}_pred_pct_move"] = round(signed_pred, 4)
                    outcome[f"{model}_error_pct"] = round(error, 4)
                    outcome[f"{model}_abs_error_pct"] = round(abs_error, 4)

        ministral_action = llm_info.get("ministral", "HOLD")
        if ministral_action in ("BUY", "SELL"):
            m_dir = "up" if ministral_action == "BUY" else "down"
            outcome["ministral_predicted"] = m_dir
            outcome["ministral_correct"] = (m_dir == actual_dir)
            outcome["ministral_actual_move_pct"] = round(move_pct, 4)
            outcome["ministral_conf"] = llm_info.get("ministral_conf", 0)

        buy_sigs = signal_info.get("_buy_signals", [])
        sell_sigs = signal_info.get("_sell_signals", [])
        per_sig = {}
        for sig_name in buy_sigs:
            per_sig[sig_name] = {"predicted": "up", "correct": actual_dir == "up"}
        for sig_name in sell_sigs:
            per_sig[sig_name] = {"predicted": "down", "correct": actual_dir == "down"}
        if per_sig:
            outcome["per_signal"] = per_sig

        outcomes[ticker] = outcome

    return outcomes if outcomes else None


def backfill_outcomes(current_underlying_prices):
    """Check past snapshots against current prices at 1h/3h horizons.

    Outcomes are written to a SEPARATE append-only file (metals_signal_outcomes.jsonl)
    so the signal log itself is never rewritten. This eliminates the race condition
    where a concurrent log_snapshot() append could be lost during rewrite.

    Args:
        current_underlying_prices: {"XAG-USD": float, "XAU-USD": float}
    """
    if not current_underlying_prices:
        return

    now = time.time()

    try:
        if not os.path.exists(SIGNAL_LOG):
            return

        # Load already-resolved keys to avoid duplicate outcomes
        resolved_keys = _load_resolved_keys()

        # Read recent signal log entries (capped for performance)
        entries = []
        with open(SIGNAL_LOG, "r", encoding="utf-8") as f:
            lines = f.readlines()

        start_idx = max(0, len(lines) - BACKFILL_SCAN_LIMIT)
        for i, line in enumerate(lines):
            if i < start_idx:
                entries.append(None)
                continue
            try:
                entries.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                entries.append(None)

        new_outcomes = []

        for i in range(start_idx, len(entries)):
            entry = entries[i]
            if entry is None:
                continue

            snapshot_ts = entry.get("ts", "")

            for h_key in HORIZONS:
                # Skip if already resolved
                if (snapshot_ts, h_key) in resolved_keys:
                    continue

                outcomes = _resolve_outcome(entry, h_key, current_underlying_prices, now)
                if outcomes:
                    outcome_entry = {
                        "snapshot_ts": snapshot_ts,
                        "horizon": h_key,
                        "resolved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "outcomes": outcomes,
                    }
                    new_outcomes.append(outcome_entry)
                    resolved_keys.add((snapshot_ts, h_key))

        # Append new outcomes to separate file (signal log stays untouched)
        for outcome in new_outcomes:
            atomic_append_jsonl(OUTCOMES_LOG, outcome)

        # Recompute accuracy from outcomes file
        _recompute_accuracy_from_outcomes()

    except Exception as e:
        print(f"[TRACKER] backfill error: {e}", flush=True)


def _recompute_accuracy_from_outcomes():
    """Recompute accuracy stats from the separate outcomes file.

    Reads metals_signal_outcomes.jsonl and computes per-signal accuracy.
    Tracks both directional accuracy and price deviation metrics.
    """
    global _accuracy_cache
    import math

    stats = {}

    def _ensure_key(key):
        if key not in stats:
            stats[key] = {
                "correct": 0, "total": 0,
                "errors": [],
                "abs_errors": [],
                "actual_moves_correct": [],
                "actual_moves_wrong": [],
            }

    resolved_count = 0

    try:
        if not os.path.exists(OUTCOMES_LOG):
            return

        with open(OUTCOMES_LOG, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                except (json.JSONDecodeError, ValueError):
                    continue

                h_key = record.get("horizon", "")
                outcomes = record.get("outcomes", {})

                for ticker, outcome in outcomes.items():
                    resolved_count += 1
                    short_ticker = ticker.split("-")[0]
                    actual_move = outcome.get("move_pct", 0)

                    if "main_correct" in outcome:
                        key = f"main_{short_ticker}_{h_key}"
                        _ensure_key(key)
                        stats[key]["total"] += 1
                        if outcome["main_correct"]:
                            stats[key]["correct"] += 1
                            stats[key]["actual_moves_correct"].append(abs(actual_move))
                        else:
                            stats[key]["actual_moves_wrong"].append(abs(actual_move))

                    if "llm_correct" in outcome:
                        key = f"llm_{short_ticker}_{h_key}"
                        _ensure_key(key)
                        stats[key]["total"] += 1
                        if outcome["llm_correct"]:
                            stats[key]["correct"] += 1
                            stats[key]["actual_moves_correct"].append(abs(actual_move))
                        else:
                            stats[key]["actual_moves_wrong"].append(abs(actual_move))

                    for model in ["chronos_1h", "chronos_3h", "ministral"]:
                        correct_key = f"{model}_correct"
                        if correct_key in outcome:
                            key = f"{model}_{short_ticker}_{h_key}"
                            _ensure_key(key)
                            stats[key]["total"] += 1
                            if outcome[correct_key]:
                                stats[key]["correct"] += 1
                                stats[key]["actual_moves_correct"].append(abs(actual_move))
                            else:
                                stats[key]["actual_moves_wrong"].append(abs(actual_move))

                            error_key = f"{model}_error_pct"
                            abs_error_key = f"{model}_abs_error_pct"
                            if error_key in outcome:
                                stats[key]["errors"].append(outcome[error_key])
                                stats[key]["abs_errors"].append(outcome[abs_error_key])

                    per_sig = outcome.get("per_signal", {})
                    for sig_name, sig_data in per_sig.items():
                        key = f"sig_{sig_name}_{short_ticker}_{h_key}"
                        _ensure_key(key)
                        stats[key]["total"] += 1
                        if sig_data.get("correct"):
                            stats[key]["correct"] += 1
                            stats[key]["actual_moves_correct"].append(abs(actual_move))
                        else:
                            stats[key]["actual_moves_wrong"].append(abs(actual_move))

    except Exception as e:
        print(f"[TRACKER] outcomes file read error: {e}", flush=True)
        return

    # Compute final stats
    result = {}
    for key, s in stats.items():
        total = min(s["total"], ACCURACY_WINDOW)
        if total <= 0:
            continue

        entry = {
            "correct": s["correct"],
            "total": s["total"],
            "accuracy": round(s["correct"] / s["total"], 3),
        }

        if s["abs_errors"]:
            n = len(s["abs_errors"])
            entry["mae"] = round(sum(s["abs_errors"]) / n, 4)
            entry["bias"] = round(sum(s["errors"]) / n, 4)
            entry["rmse"] = round(math.sqrt(sum(e**2 for e in s["errors"]) / n), 4)
            entry["deviation_samples"] = n

        if s["actual_moves_correct"]:
            entry["avg_move_correct"] = round(
                sum(s["actual_moves_correct"]) / len(s["actual_moves_correct"]), 4)
        if s["actual_moves_wrong"]:
            entry["avg_move_wrong"] = round(
                sum(s["actual_moves_wrong"]) / len(s["actual_moves_wrong"]), 4)

        result[key] = entry

    with _lock:
        _accuracy_cache = result

    try:
        atomic_write_json(ACCURACY_CACHE_FILE, {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "resolved_snapshots": resolved_count,
            "stats": result,
        })
    except Exception as e:
        print(f"[TRACKER] accuracy cache write error: {e}", flush=True)


def get_accuracy_report():
    """Get full accuracy report dict.

    Returns:
        {signal_key: {correct, total, accuracy}} e.g.
        {"main_XAG_1h": {"correct": 13, "total": 18, "accuracy": 0.722}, ...}
    """
    with _lock:
        if _accuracy_cache:
            return dict(_accuracy_cache)

    # Try loading from disk
    data = load_json(ACCURACY_CACHE_FILE)
    if data:
        return data.get("stats", {})

    return {}


def get_accuracy_summary():
    """Get a one-line summary string for status display.

    Includes direction accuracy + MAE where available.
    Returns e.g.: "main_XAG_1h:72%(18) chronos_1h_XAG_1h:55%(11)MAE=0.32%"
    """
    report = get_accuracy_report()
    if not report:
        return "no data"

    parts = []
    # Sort by total samples descending
    for key in sorted(report.keys(), key=lambda k: -report[k]["total"]):
        s = report[key]
        if s["total"] >= 5:  # only show meaningful samples
            pct = int(s["accuracy"] * 100)
            part = f"{key}:{pct}%({s['total']})"
            # Add MAE for models with price deviation data
            if "mae" in s:
                part += f" MAE={s['mae']:.2f}%"
            parts.append(part)

    if not parts:
        return "< 5 samples"

    return " | ".join(parts[:6])  # top 6 to keep it readable


def get_accuracy_for_context():
    """Get accuracy data formatted for metals_context.json (Claude consumption).

    Returns dict suitable for inclusion in context file.
    """
    report = get_accuracy_report()
    if not report:
        return {"status": "no_data", "stats": {}}

    # Group by signal source
    by_source = {}
    for key, stats in report.items():
        parts = key.rsplit("_", 1)  # split off horizon
        if len(parts) == 2:
            source = parts[0]
            horizon = parts[1]
        else:
            source = key
            horizon = "?"

        if source not in by_source:
            by_source[source] = {}
        entry = {
            "accuracy": stats["accuracy"],
            "samples": stats["total"],
        }
        # Include deviation metrics if available
        if "mae" in stats:
            entry["mae"] = stats["mae"]
            entry["bias"] = stats["bias"]
            entry["rmse"] = stats["rmse"]
        if "avg_move_correct" in stats:
            entry["avg_move_correct"] = stats["avg_move_correct"]
        if "avg_move_wrong" in stats:
            entry["avg_move_wrong"] = stats["avg_move_wrong"]
        by_source[source][horizon] = entry

    # Find best/worst
    best_key = max(report.keys(), key=lambda k: report[k]["accuracy"]) if report else None
    worst_key = min(report.keys(), key=lambda k: report[k]["accuracy"]) if report else None

    return {
        "status": "active",
        "total_signals_tracked": sum(s["total"] for s in report.values()),
        "by_source": by_source,
        "best": {"signal": best_key, **report[best_key]} if best_key else None,
        "worst": {"signal": worst_key, **report[worst_key]} if worst_key else None,
    }


def get_snapshot_count():
    """Get total number of snapshots logged."""
    try:
        if not os.path.exists(SIGNAL_LOG):
            return 0
        with open(SIGNAL_LOG, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception as e:
        print(f"[TRACKER] snapshot count error: {e}", flush=True)
        return 0
