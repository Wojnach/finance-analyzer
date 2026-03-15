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

    # 5. Cache external research (seed + auto-refresh from live sources)
    base_research = _load_or_seed_external_research()
    context["external_research"] = _refresh_external_data(base_research)

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


_REFRESH_STATE_FILE = "data/silver_refresh_state.json"

# Per-source refresh intervals (seconds)
_REFRESH_INTERVALS = {
    "slv": 24 * 3600,           # SLV ETF: daily
    "silver_futures": 24 * 3600, # Silver futures: daily
    "gold": 24 * 3600,          # Gold context: daily
    "cot": 7 * 24 * 3600,       # CFTC COT: weekly
}

_REQUEST_TIMEOUT = 15  # seconds per HTTP request


def _refresh_external_data(base_research):
    """Auto-refresh external data from free public sources.

    Each source has its own refresh interval and failure handling.
    Failures never block the precompute — old cached data persists.
    """
    import time

    updated = dict(base_research)
    refresh_state = load_json(_REFRESH_STATE_FILE, default={})
    now = time.time()
    refreshed = []

    # 1. SLV ETF data (daily) — investment flow proxy
    if _should_refresh(refresh_state, "slv", now):
        data = _safe_fetch("SLV ETF", _fetch_slv_data)
        if data:
            updated["slv_etf"] = data
            refresh_state["slv"] = {"ts": now, "ok": True}
            refreshed.append("slv")
        else:
            refresh_state["slv"] = {
                **refresh_state.get("slv", {}),
                "last_error_ts": now,
            }

    # 2. Silver futures price context (daily)
    if _should_refresh(refresh_state, "silver_futures", now):
        data = _safe_fetch("Silver futures", _fetch_silver_futures)
        if data:
            updated["futures_context"] = data
            refresh_state["silver_futures"] = {"ts": now, "ok": True}
            refreshed.append("silver_futures")
        else:
            refresh_state["silver_futures"] = {
                **refresh_state.get("silver_futures", {}),
                "last_error_ts": now,
            }

    # 3. Gold context for live G/S ratio (daily)
    if _should_refresh(refresh_state, "gold", now):
        data = _safe_fetch("Gold context", _fetch_gold_context)
        if data:
            updated["gold_context"] = data
            refresh_state["gold"] = {"ts": now, "ok": True}
            refreshed.append("gold")
        else:
            refresh_state["gold"] = {
                **refresh_state.get("gold", {}),
                "last_error_ts": now,
            }

    # 4. CFTC COT data (weekly)
    if _should_refresh(refresh_state, "cot", now):
        data = _safe_fetch("CFTC COT", _fetch_cftc_cot)
        if data:
            # Merge live COT with static seed (keep seed as fallback context)
            existing_cot = updated.get("cot_positioning", {})
            updated["cot_positioning"] = {**existing_cot, "live": data}
            refresh_state["cot"] = {"ts": now, "ok": True}
            refreshed.append("cot")
        else:
            refresh_state["cot"] = {
                **refresh_state.get("cot", {}),
                "last_error_ts": now,
            }

    if refreshed:
        updated["last_auto_refresh"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
        updated["auto_refreshed_sources"] = refreshed
        atomic_write_json(_REFRESH_STATE_FILE, refresh_state)
        logger.info("Silver external refresh: %s", ", ".join(refreshed))

    # Always include refresh status so /fin-silver knows what's fresh
    updated["refresh_status"] = {
        src: {
            "age_hours": round((now - s.get("ts", 0)) / 3600, 1),
            "ok": s.get("ok", False),
        }
        for src, s in refresh_state.items()
    }

    return updated


def _should_refresh(state, key, now):
    """Check if a source is due for refresh."""
    interval = _REFRESH_INTERVALS.get(key, 24 * 3600)
    last = state.get(key, {}).get("ts", 0)
    return (now - last) >= interval


def _safe_fetch(name, fetch_fn):
    """Call a fetch function with error handling. Returns None on failure."""
    try:
        return fetch_fn()
    except Exception as e:
        logger.warning("Auto-refresh %s failed: %s", name, e)
        return None


def _fetch_slv_data():
    """Fetch SLV ETF data via yfinance — investment flow proxy."""
    import yfinance as yf

    slv = yf.Ticker("SLV")
    hist = slv.history(period="1mo")

    if hist.empty:
        return None

    closes = hist["Close"].dropna()
    volumes = hist["Volume"].dropna()

    if len(closes) < 2:
        return None

    result = {
        "last_price": round(float(closes.iloc[-1]), 2),
        "price_1mo_ago": round(float(closes.iloc[0]), 2),
        "change_1mo_pct": round(
            float((closes.iloc[-1] / closes.iloc[0] - 1) * 100), 2
        ),
        "high_1mo": round(float(closes.max()), 2),
        "low_1mo": round(float(closes.min()), 2),
        "avg_volume_1mo": int(volumes.mean()) if len(volumes) > 0 else None,
        "latest_volume": int(volumes.iloc[-1]) if len(volumes) > 0 else None,
        "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    # Volume trend (last 5d vs 1mo avg) — rising volume = conviction
    if len(volumes) >= 5:
        vol_5d = float(volumes.tail(5).mean())
        vol_1mo = float(volumes.mean())
        result["volume_trend"] = round(vol_5d / vol_1mo, 2) if vol_1mo > 0 else None
        result["volume_trend_note"] = (
            "rising" if result["volume_trend"] and result["volume_trend"] > 1.2
            else "falling" if result["volume_trend"] and result["volume_trend"] < 0.8
            else "stable"
        )

    return result


def _fetch_silver_futures():
    """Fetch silver futures (SI=F) price context via yfinance."""
    import yfinance as yf

    si = yf.Ticker("SI=F")
    hist = si.history(period="3mo")

    if hist.empty:
        return None

    closes = hist["Close"].dropna()
    if len(closes) < 5:
        return None

    # Compute key levels
    high_3mo = float(closes.max())
    low_3mo = float(closes.min())
    current = float(closes.iloc[-1])

    # Fibonacci retracement from 3mo high/low
    fib_range = high_3mo - low_3mo
    fibs = {
        "0.0_low": round(low_3mo, 2),
        "0.236": round(high_3mo - fib_range * 0.236, 2),
        "0.382": round(high_3mo - fib_range * 0.382, 2),
        "0.500": round(high_3mo - fib_range * 0.5, 2),
        "0.618": round(high_3mo - fib_range * 0.618, 2),
        "0.786": round(high_3mo - fib_range * 0.786, 2),
        "1.0_high": round(high_3mo, 2),
    }

    # Simple moving averages
    sma20 = round(float(closes.tail(20).mean()), 2) if len(closes) >= 20 else None
    sma50 = round(float(closes.tail(50).mean()), 2) if len(closes) >= 50 else None

    # 1-week and 1-month changes
    change_1w = round(
        float((current / closes.iloc[-5] - 1) * 100), 2
    ) if len(closes) >= 5 else None
    change_1mo = round(
        float((current / closes.iloc[-20] - 1) * 100), 2
    ) if len(closes) >= 20 else None

    return {
        "current": round(current, 2),
        "high_3mo": round(high_3mo, 2),
        "low_3mo": round(low_3mo, 2),
        "change_1w_pct": change_1w,
        "change_1mo_pct": change_1mo,
        "sma20": sma20,
        "sma50": sma50,
        "fibonacci_3mo": fibs,
        "position_in_range_pct": round(
            (current - low_3mo) / fib_range * 100, 1
        ) if fib_range > 0 else None,
        "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def _fetch_gold_context():
    """Fetch gold futures (GC=F) for G/S ratio computation."""
    import yfinance as yf

    gc = yf.Ticker("GC=F")
    hist = gc.history(period="1mo")

    if hist.empty:
        return None

    closes = hist["Close"].dropna()
    if len(closes) < 2:
        return None

    current_gold = float(closes.iloc[-1])

    # Also fetch silver for ratio
    si = yf.Ticker("SI=F")
    si_hist = si.history(period="5d")
    current_silver = None
    gs_ratio = None
    if not si_hist.empty:
        si_closes = si_hist["Close"].dropna()
        if len(si_closes) > 0:
            current_silver = float(si_closes.iloc[-1])
            gs_ratio = round(current_gold / current_silver, 2) if current_silver > 0 else None

    return {
        "gold_price": round(current_gold, 2),
        "silver_price": round(current_silver, 2) if current_silver else None,
        "gs_ratio": gs_ratio,
        "gold_change_1mo_pct": round(
            float((closes.iloc[-1] / closes.iloc[0] - 1) * 100), 2
        ),
        "gold_high_1mo": round(float(closes.max()), 2),
        "gold_low_1mo": round(float(closes.min()), 2),
        # Ratio context
        "gs_ratio_vs_historical": (
            "severely undervalued" if gs_ratio and gs_ratio > 80
            else "undervalued" if gs_ratio and gs_ratio > 65
            else "fair value" if gs_ratio and gs_ratio > 50
            else "overvalued" if gs_ratio and gs_ratio > 35
            else "extremely overvalued" if gs_ratio else None
        ),
        "implied_silver_at_50": round(current_gold / 50, 2) if current_gold else None,
        "implied_silver_at_40": round(current_gold / 40, 2) if current_gold else None,
        "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def _fetch_cftc_cot():
    """Fetch CFTC COT data for silver from the SOCRATA Open Data API.

    Uses the legacy futures-only report. Silver CFTC commodity code: 084691.
    Free public API, no key needed (rate-limited to ~1000 req/hour).
    """
    import requests

    # Legacy Futures Only report — silver commodity name
    url = (
        "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
        "?$where=commodity_name='SILVER'"
        "&$order=report_date_as_yyyy_mm_dd DESC"
        "&$limit=5"
    )

    resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()
    rows = resp.json()

    if not rows:
        return None

    latest = rows[0]
    prev = rows[1] if len(rows) > 1 else {}

    def _int(val):
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    oi = _int(latest.get("open_interest_all"))
    nc_long = _int(latest.get("noncomm_positions_long_all"))
    nc_short = _int(latest.get("noncomm_positions_short_all"))
    comm_long = _int(latest.get("comm_positions_long_all"))
    comm_short = _int(latest.get("comm_positions_short_all"))

    nc_net = (nc_long - nc_short) if nc_long is not None and nc_short is not None else None
    comm_net = (comm_long - comm_short) if comm_long is not None and comm_short is not None else None

    # Previous week for comparison
    prev_nc_long = _int(prev.get("noncomm_positions_long_all"))
    prev_nc_short = _int(prev.get("noncomm_positions_short_all"))
    prev_nc_net = (prev_nc_long - prev_nc_short) if prev_nc_long is not None and prev_nc_short is not None else None

    result = {
        "report_date": latest.get("report_date_as_yyyy_mm_dd"),
        "open_interest": oi,
        "noncomm_long": nc_long,
        "noncomm_short": nc_short,
        "noncomm_net": nc_net,
        "comm_long": comm_long,
        "comm_short": comm_short,
        "comm_net": comm_net,
        "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    # Week-over-week changes
    if prev_nc_net is not None and nc_net is not None:
        result["noncomm_net_change"] = nc_net - prev_nc_net
        result["prev_report_date"] = prev.get("report_date_as_yyyy_mm_dd")

    # Also fetch disaggregated for managed money if available
    try:
        disagg_url = (
            "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
            "?$where=commodity_name='SILVER'"
            "&$order=report_date_as_yyyy_mm_dd DESC"
            "&$limit=3"
        )
        disagg_resp = requests.get(disagg_url, timeout=_REQUEST_TIMEOUT)
        disagg_resp.raise_for_status()
        disagg_rows = disagg_resp.json()

        if disagg_rows:
            d = disagg_rows[0]
            mm_long = _int(d.get("m_money_positions_long_all"))
            mm_short = _int(d.get("m_money_positions_short_all"))
            if mm_long is not None and mm_short is not None:
                result["managed_money_long"] = mm_long
                result["managed_money_short"] = mm_short
                result["managed_money_net"] = mm_long - mm_short
    except Exception as e:
        logger.debug("Disaggregated COT fetch failed (non-critical): %s", e)

    return result


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
