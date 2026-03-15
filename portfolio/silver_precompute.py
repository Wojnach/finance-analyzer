"""Silver Deep Context Precomputer.

Aggregates slowly-changing silver market data into a single cached file
for fast consumption by the /fin-silver command.

Run manually: .venv/Scripts/python.exe portfolio/silver_precompute.py
Could be scheduled daily via Task Scheduler.

Output: data/silver_deep_context.json
"""

import datetime
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio.file_utils import atomic_write_json, load_json, load_jsonl

logger = logging.getLogger("portfolio.silver_precompute")

_STATE_FILE = "data/silver_precompute_state.json"
_DEFAULT_INTERVAL_SEC = 4 * 3600  # 4 hours


def maybe_precompute_silver(config=None):
    """Run silver precompute if enough time has elapsed since last run.

    Called from main loop's _run_post_cycle(). Self-checking — safe to call
    every cycle; will only execute when interval has elapsed.

    Also runs on first call if state file doesn't exist (fresh start / crash recovery).
    """
    import time

    interval = _DEFAULT_INTERVAL_SEC
    if config:
        interval = config.get("silver", {}).get(
            "precompute_interval_sec", _DEFAULT_INTERVAL_SEC
        )

    state = load_json(_STATE_FILE, default={})
    last_run = state.get("last_run_epoch", 0)
    now = time.time()

    if (now - last_run) < interval:
        return None  # Not yet

    try:
        result = precompute()
        atomic_write_json(_STATE_FILE, {
            "last_run_epoch": now,
            "last_run_iso": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "ok",
        })
        logger.info("Silver precompute completed (interval=%ds)", interval)
        return result
    except Exception as e:
        logger.warning("Silver precompute failed: %s", e)
        # Write failure state so we retry next cycle, not wait full interval
        atomic_write_json(_STATE_FILE, {
            "last_run_epoch": last_run,  # Keep old timestamp so we retry soon
            "last_run_iso": state.get("last_run_iso", ""),
            "status": f"error: {e}",
            "last_error_epoch": now,
        })
        return None


def precompute():
    """Aggregate all silver context into a single cached JSON file."""
    context = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "version": 1,
    }

    # 1. Extract XAG signal accuracy history from agent_summary_compact.json
    context["signal_accuracy"] = _compute_xag_signal_accuracy()

    # 2. Extract price trajectory from price_snapshots_hourly.jsonl
    context["price_trajectory"] = _compute_price_trajectory()

    # 3. Extract XAG entries from layer2_journal.jsonl
    context["journal_history"] = _extract_xag_journal_entries()

    # 4. Compute rolling performance metrics
    context["performance"] = _compute_rolling_performance()

    # 5. Cache external research (manually updated section)
    context["external_research"] = _load_or_seed_external_research()

    # 6. Prophecy snapshot
    context["prophecy"] = _extract_prophecy()

    # 7. Historical G/S ratio data points
    context["gs_ratio_history"] = _compute_gs_ratio_history()

    atomic_write_json("data/silver_deep_context.json", context)
    print(f"Silver deep context written to data/silver_deep_context.json")
    print(f"  Generated: {context['generated_at']}")
    print(f"  Sections: {list(context.keys())}")
    return context


def _compute_xag_signal_accuracy():
    """Extract XAG-specific signal accuracy from agent_summary_compact.json."""
    summary = load_json("data/agent_summary_compact.json")
    if not summary:
        return {}

    reliability = summary.get("signal_reliability", {}).get("XAG-USD", {})
    focus_probs = summary.get("focus_probabilities", {}).get("XAG-USD", {})
    forecast = summary.get("forecast_signals", {}).get("XAG-USD", {})
    forecast_acc = summary.get("forecast_accuracy", {}).get("accuracy", {})

    # Get XAG-specific Chronos accuracy
    chronos_xag = {}
    for key in ("chronos_24h", "kronos_24h"):
        by_ticker = forecast_acc.get(key, {}).get("by_ticker", {})
        data = by_ticker.get("XAG-USD", {}) if isinstance(by_ticker, dict) else {}
        if data:
            chronos_xag[key] = data

    return {
        "best_signals": reliability.get("best", {}),
        "worst_signals": reliability.get("worst", {}),
        "focus_probabilities": focus_probs,
        "forecast": forecast,
        "chronos_accuracy": chronos_xag,
    }


def _compute_price_trajectory():
    """Extract XAG price history from price_snapshots_hourly.jsonl."""
    from pathlib import Path

    path = Path("data/price_snapshots_hourly.jsonl")
    if not path.exists():
        return {}

    prices = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    price_data = entry.get("prices", {})
                    xag = price_data.get("XAG-USD")
                    xau = price_data.get("XAU-USD")
                    if xag is not None:
                        prices.append({
                            "ts": entry.get("ts", ""),
                            "xag": xag,
                            "xau": xau,
                            "gs_ratio": round(xau / xag, 2) if xau and xag and xag > 0 else None,
                        })
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
    except OSError as e:
        logger.warning("Failed to read price snapshots: %s", e)
        return {}

    if not prices:
        return {}

    # Keep last 7 days of hourly data + summary stats
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=7)
    cutoff_iso = cutoff.isoformat()
    recent = [p for p in prices if p["ts"] >= cutoff_iso]

    if len(recent) < 2:
        return {"data_points": len(recent)}

    first_price = recent[0]["xag"]
    last_price = recent[-1]["xag"]
    high = max(p["xag"] for p in recent)
    low = min(p["xag"] for p in recent)

    return {
        "data_points": len(recent),
        "period_start": recent[0]["ts"],
        "period_end": recent[-1]["ts"],
        "start_price": first_price,
        "end_price": last_price,
        "high_7d": high,
        "low_7d": low,
        "change_pct": round((last_price - first_price) / first_price * 100, 2) if first_price else 0,
        "range_pct": round((high - low) / low * 100, 2) if low else 0,
        "gs_ratio_start": recent[0].get("gs_ratio"),
        "gs_ratio_end": recent[-1].get("gs_ratio"),
        # Keep every 6th point for charting (~4hr intervals)
        "hourly_prices": [
            {"ts": p["ts"], "xag": p["xag"], "gs": p.get("gs_ratio")}
            for p in recent[::6]
        ],
    }


def _extract_xag_journal_entries():
    """Extract journal entries that mention XAG or silver."""
    entries = load_jsonl("data/layer2_journal.jsonl")
    if not entries:
        return []

    xag_entries = []
    for entry in entries:
        has_xag = (
            "XAG-USD" in entry.get("tickers", {})
            or "XAG-USD" in entry.get("prices", {})
        )
        if has_xag:
            xag_entries.append({
                "ts": entry.get("ts"),
                "regime": entry.get("regime"),
                "xag_outlook": entry.get("tickers", {}).get("XAG-USD", {}).get("outlook"),
                "xag_thesis": entry.get("tickers", {}).get("XAG-USD", {}).get("thesis"),
                "xag_conviction": entry.get("tickers", {}).get("XAG-USD", {}).get("conviction"),
                "xag_price": entry.get("prices", {}).get("XAG-USD"),
            })

    # Return last 20 XAG entries
    return xag_entries[-20:]


def _compute_rolling_performance():
    """Compute rolling signal performance for XAG from metals_signal_log.jsonl."""
    entries = load_jsonl("data/metals_signal_log.jsonl")
    if not entries:
        return {}

    # Count signal occurrences for XAG
    signal_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
    last_entries = []
    for entry in entries:
        entry_str = json.dumps(entry)
        if "XAG" in entry_str:
            action = entry.get("action", entry.get("consensus", "HOLD"))
            if action in signal_counts:
                signal_counts[action] += 1
            last_entries.append({"ts": entry.get("ts", ""), "action": action})

    return {
        "total_signals": sum(signal_counts.values()),
        "signal_distribution": signal_counts,
        "last_10_signals": last_entries[-10:],
    }


def _load_or_seed_external_research():
    """Load cached external research or seed with known data."""
    # Try to load existing external cache
    existing = load_json("data/silver_external_cache.json")
    if existing and existing.get("analyst_targets"):
        return existing

    # Seed with known data (user/system can update this file)
    return {
        "last_updated": "2026-03-14",
        "analyst_targets": {
            "citi": {
                "target": 110,
                "timeframe": "H2 2026",
                "note": "Most bullish major bank",
            },
            "goldman_sachs": {
                "target_range": [85, 100],
                "note": "Primary strategic metal of green transition",
            },
            "jpm": {
                "target_avg": 81,
                "target_q4": 85,
                "note": "Conservative",
            },
            "td_securities": {
                "target_avg": 65.5,
                "target_high": 118,
            },
            "neumeyer": {
                "target_range": [100, 130],
                "note": "First Majestic CEO, called $100",
            },
            "model_avg": {
                "target_range": [112, 119],
                "note": "Algorithmic/model forecasts",
            },
        },
        "supply_demand": {
            "deficit_2026_moz": "67-200",
            "consecutive_deficit_years": 6,
            "cumulative_deficit_moz": 820,
            "mine_production_moz": 820,
            "mine_peak_year": 2016,
            "byproduct_pct": 72,
            "new_supply_earliest": 2033,
            "industrial_demand_record_moz": 680.5,
            "solar_demand_moz": 200,
            "solar_2030_projected_moz": "321-450",
            "ev_demand_moz": 90,
            "ai_datacenter_moz": 78,
        },
        "physical_market": {
            "comex_registered_moz": 88,
            "comex_decline_from_2020_pct": 70,
            "paper_physical_ratio": "356-408:1",
            "backwardation_since": "2025-10",
            "efp_spread": 1.10,
            "efp_historical": 0.25,
            "tokyo_premium_pct": 60,
            "dubai_premium_pct": 40,
        },
        "cot_positioning": {
            "large_spec_contracts": 32500,
            "large_spec_note": "13-year low participation",
            "managed_funds_net_long": 8500,
            "rally_driver": "physical demand, not speculation",
        },
        "china_export_controls": {
            "effective_date": "2026-01-01",
            "impact": "Restricts 60-70% of China silver exports",
            "china_global_production_pct": 13,
        },
        "gs_ratio": {
            "current": 62,
            "historical_avg": "30-35",
            "bull_market_avg": "60-70",
            "2011_low": 32,
            "1980_low": 16,
        },
        "key_risks": [
            "CME margin hikes (Jan +47%, Feb +15-18%) — caused 31% crash from $121",
            "DXY strength above 100",
            "Solar thrifting -7% per-panel YoY",
            "Substitution R&D accelerating at $80+",
            "Exhaustion signals — Iran escalation didn't move silver",
        ],
    }


def _extract_prophecy():
    """Extract silver prophecy data."""
    prophecy = load_json("data/prophecy.json")
    if not prophecy:
        return {}

    beliefs = prophecy.get("beliefs", [])
    if not isinstance(beliefs, list):
        return {}

    for belief in beliefs:
        if not isinstance(belief, dict):
            continue
        if belief.get("id") == "silver_bull_2026":
            checkpoints = []
            for cp in belief.get("checkpoints", []):
                if not isinstance(cp, dict):
                    continue
                checkpoints.append({
                    "condition": cp.get("condition"),
                    "target": cp.get("target_value"),
                    "status": cp.get("status"),
                    "triggered_at": cp.get("triggered_at"),
                })
            return {
                "thesis": belief.get("thesis"),
                "target": belief.get("target_price"),
                "conviction": belief.get("conviction"),
                "entry_price": belief.get("entry_price"),
                "checkpoints": checkpoints,
                "supporting": belief.get("supporting_evidence", []),
                "opposing": belief.get("opposing_evidence", []),
            }
    return {}


def _compute_gs_ratio_history():
    """Compute gold/silver ratio from recent price snapshots."""
    from pathlib import Path

    path = Path("data/price_snapshots_hourly.jsonl")
    if not path.exists():
        return []

    ratios = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    price_data = entry.get("prices", {})
                    xag = price_data.get("XAG-USD")
                    xau = price_data.get("XAU-USD")
                    if xag and xau and xag > 0:
                        ratios.append({
                            "ts": entry.get("ts", ""),
                            "ratio": round(xau / xag, 2),
                        })
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
    except OSError as e:
        logger.warning("Failed to read price snapshots for G/S ratio: %s", e)
        return []

    # Return last 30 data points (about 1.5 days at hourly)
    return ratios[-30:]


if __name__ == "__main__":
    precompute()
