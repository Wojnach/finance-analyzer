"""Gold Deep Context Precomputer.

Aggregates slowly-changing gold market data into a single cached file
for fast consumption by the /fin-gold command.

Run manually: .venv/Scripts/python.exe portfolio/gold_precompute.py
Auto-runs every 4h via main loop integration.

Output: data/gold_deep_context.json
"""

import datetime
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio.file_utils import atomic_write_json, load_json, load_jsonl

logger = logging.getLogger("portfolio.gold_precompute")

_STATE_FILE = "data/gold_precompute_state.json"
_REFRESH_STATE_FILE = "data/gold_refresh_state.json"
_DEFAULT_INTERVAL_SEC = 4 * 3600  # 4 hours

# Per-source refresh intervals (seconds)
_REFRESH_INTERVALS = {
    "gld": 24 * 3600,            # GLD ETF: daily
    "gold_futures": 24 * 3600,   # Gold futures: daily
    "cot": 7 * 24 * 3600,       # CFTC COT: weekly
}

_REQUEST_TIMEOUT = 15


def maybe_precompute_gold(config=None):
    """Run gold precompute if enough time has elapsed since last run.

    Called from main loop's _run_post_cycle(). Self-checking — safe to call
    every cycle; will only execute when interval has elapsed.
    """
    import time

    interval = _DEFAULT_INTERVAL_SEC
    if config:
        interval = config.get("gold", {}).get(
            "precompute_interval_sec", _DEFAULT_INTERVAL_SEC
        )

    state = load_json(_STATE_FILE, default={})
    last_run = state.get("last_run_epoch", 0)
    now = time.time()

    if (now - last_run) < interval:
        return None

    try:
        result = precompute()
        atomic_write_json(_STATE_FILE, {
            "last_run_epoch": now,
            "last_run_iso": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "ok",
        })
        logger.info("Gold precompute completed (interval=%ds)", interval)
        return result
    except Exception as e:
        logger.warning("Gold precompute failed: %s", e)
        atomic_write_json(_STATE_FILE, {
            "last_run_epoch": last_run,
            "last_run_iso": state.get("last_run_iso", ""),
            "status": f"error: {e}",
            "last_error_epoch": now,
        })
        return None


def precompute():
    """Aggregate all gold context into a single cached JSON file."""
    context = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "version": 1,
    }

    context["signal_accuracy"] = _compute_xau_signal_accuracy()
    context["price_trajectory"] = _compute_price_trajectory()
    context["journal_history"] = _extract_xau_journal_entries()

    # External research with auto-refresh
    base_research = _load_or_seed_external_research()
    context["external_research"] = _refresh_external_data(base_research)

    context["gs_ratio_history"] = _compute_gs_ratio_history()

    atomic_write_json("data/gold_deep_context.json", context)
    print(f"Gold deep context written to data/gold_deep_context.json")
    print(f"  Generated: {context['generated_at']}")
    print(f"  Sections: {list(context.keys())}")
    return context


def _compute_xau_signal_accuracy():
    """Extract XAU-specific signal accuracy from agent_summary_compact.json."""
    summary = load_json("data/agent_summary_compact.json")
    if not summary:
        return {}

    reliability = summary.get("signal_reliability", {}).get("XAU-USD", {})
    focus_probs = summary.get("focus_probabilities", {}).get("XAU-USD", {})
    forecast = summary.get("forecast_signals", {}).get("XAU-USD", {})
    forecast_acc = summary.get("forecast_accuracy", {}).get("accuracy", {})

    chronos_xau = {}
    for key in ("chronos_24h", "kronos_24h"):
        by_ticker = forecast_acc.get(key, {}).get("by_ticker", {})
        data = by_ticker.get("XAU-USD", {}) if isinstance(by_ticker, dict) else {}
        if data:
            chronos_xau[key] = data

    return {
        "best_signals": reliability.get("best", {}),
        "worst_signals": reliability.get("worst", {}),
        "focus_probabilities": focus_probs,
        "forecast": forecast,
        "chronos_accuracy": chronos_xau,
    }


def _compute_price_trajectory():
    """Extract XAU price history from price_snapshots_hourly.jsonl."""
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
                    xau = price_data.get("XAU-USD")
                    xag = price_data.get("XAG-USD")
                    if xau is not None:
                        prices.append({
                            "ts": entry.get("ts", ""),
                            "xau": xau,
                            "xag": xag,
                            "gs_ratio": round(xau / xag, 2) if xau and xag and xag > 0 else None,
                        })
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
    except OSError as e:
        logger.warning("Failed to read price snapshots: %s", e)
        return {}

    if not prices:
        return {}

    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=7)
    cutoff_iso = cutoff.isoformat()
    recent = [p for p in prices if p["ts"] >= cutoff_iso]

    if len(recent) < 2:
        return {"data_points": len(recent)}

    first_price = recent[0]["xau"]
    last_price = recent[-1]["xau"]
    high = max(p["xau"] for p in recent)
    low = min(p["xau"] for p in recent)

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
        "hourly_prices": [
            {"ts": p["ts"], "xau": p["xau"], "gs": p.get("gs_ratio")}
            for p in recent[::6]
        ],
    }


def _extract_xau_journal_entries():
    """Extract journal entries that mention XAU or gold."""
    entries = load_jsonl("data/layer2_journal.jsonl")
    if not entries:
        return []

    xau_entries = []
    for entry in entries:
        has_xau = (
            "XAU-USD" in entry.get("tickers", {})
            or "XAU-USD" in entry.get("prices", {})
        )
        if has_xau:
            xau_entries.append({
                "ts": entry.get("ts"),
                "regime": entry.get("regime"),
                "xau_outlook": entry.get("tickers", {}).get("XAU-USD", {}).get("outlook"),
                "xau_thesis": entry.get("tickers", {}).get("XAU-USD", {}).get("thesis"),
                "xau_conviction": entry.get("tickers", {}).get("XAU-USD", {}).get("conviction"),
                "xau_price": entry.get("prices", {}).get("XAU-USD"),
            })

    return xau_entries[-20:]


# ---------------------------------------------------------------------------
# Auto-refresh: live data from public sources
# ---------------------------------------------------------------------------

def _refresh_external_data(base_research):
    """Auto-refresh gold external data from free public sources."""
    import time

    updated = dict(base_research)
    refresh_state = load_json(_REFRESH_STATE_FILE, default={})
    now = time.time()
    refreshed = []

    # 1. GLD ETF (daily)
    if _should_refresh(refresh_state, "gld", now):
        data = _safe_fetch("GLD ETF", _fetch_gld_data)
        if data:
            updated["gld_etf"] = data
            refresh_state["gld"] = {"ts": now, "ok": True}
            refreshed.append("gld")
        else:
            refresh_state["gld"] = {
                **refresh_state.get("gld", {}), "last_error_ts": now,
            }

    # 2. Gold futures context (daily)
    if _should_refresh(refresh_state, "gold_futures", now):
        data = _safe_fetch("Gold futures", _fetch_gold_futures)
        if data:
            updated["futures_context"] = data
            refresh_state["gold_futures"] = {"ts": now, "ok": True}
            refreshed.append("gold_futures")
        else:
            refresh_state["gold_futures"] = {
                **refresh_state.get("gold_futures", {}), "last_error_ts": now,
            }

    # 3. CFTC COT for gold (weekly)
    if _should_refresh(refresh_state, "cot", now):
        data = _safe_fetch("CFTC COT Gold", _fetch_cftc_cot_gold)
        if data:
            existing_cot = updated.get("cot_positioning", {})
            updated["cot_positioning"] = {**existing_cot, "live": data}
            refresh_state["cot"] = {"ts": now, "ok": True}
            refreshed.append("cot")
        else:
            refresh_state["cot"] = {
                **refresh_state.get("cot", {}), "last_error_ts": now,
            }

    if refreshed:
        updated["last_auto_refresh"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
        updated["auto_refreshed_sources"] = refreshed
        atomic_write_json(_REFRESH_STATE_FILE, refresh_state)
        logger.info("Gold external refresh: %s", ", ".join(refreshed))

    updated["refresh_status"] = {
        src: {
            "age_hours": round((now - s.get("ts", 0)) / 3600, 1),
            "ok": s.get("ok", False),
        }
        for src, s in refresh_state.items()
    }

    return updated


def _should_refresh(state, key, now):
    interval = _REFRESH_INTERVALS.get(key, 24 * 3600)
    last = state.get(key, {}).get("ts", 0)
    return (now - last) >= interval


def _safe_fetch(name, fetch_fn):
    try:
        return fetch_fn()
    except Exception as e:
        logger.warning("Auto-refresh %s failed: %s", name, e)
        return None


def _fetch_gld_data():
    """Fetch GLD ETF data via yfinance."""
    import yfinance as yf

    gld = yf.Ticker("GLD")
    hist = gld.history(period="1mo")

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


def _fetch_gold_futures():
    """Fetch gold futures (GC=F) price context via yfinance."""
    import yfinance as yf

    gc = yf.Ticker("GC=F")
    hist = gc.history(period="3mo")

    if hist.empty:
        return None

    closes = hist["Close"].dropna()
    if len(closes) < 5:
        return None

    high_3mo = float(closes.max())
    low_3mo = float(closes.min())
    current = float(closes.iloc[-1])

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

    sma20 = round(float(closes.tail(20).mean()), 2) if len(closes) >= 20 else None
    sma50 = round(float(closes.tail(50).mean()), 2) if len(closes) >= 50 else None

    change_1w = round(
        float((current / closes.iloc[-5] - 1) * 100), 2
    ) if len(closes) >= 5 else None
    change_1mo = round(
        float((current / closes.iloc[-20] - 1) * 100), 2
    ) if len(closes) >= 20 else None

    # Also compute silver ratio
    si = yf.Ticker("SI=F")
    si_hist = si.history(period="5d")
    gs_ratio = None
    silver_price = None
    if not si_hist.empty:
        si_closes = si_hist["Close"].dropna()
        if len(si_closes) > 0:
            silver_price = float(si_closes.iloc[-1])
            gs_ratio = round(current / silver_price, 2) if silver_price > 0 else None

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
        "distance_from_high_pct": round(
            (high_3mo - current) / high_3mo * 100, 2
        ),
        "gs_ratio": gs_ratio,
        "silver_price": round(silver_price, 2) if silver_price else None,
        "implied_silver_at_50": round(current / 50, 2),
        "implied_silver_at_40": round(current / 40, 2),
        "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def _fetch_cftc_cot_gold():
    """Fetch CFTC COT data for gold from SOCRATA API."""
    import requests

    url = (
        "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
        "?$where=commodity_name='GOLD'"
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

    if prev_nc_net is not None and nc_net is not None:
        result["noncomm_net_change"] = nc_net - prev_nc_net
        result["prev_report_date"] = prev.get("report_date_as_yyyy_mm_dd")

    # Disaggregated for managed money
    try:
        disagg_url = (
            "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
            "?$where=commodity_name='GOLD'"
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
        logger.debug("Disaggregated COT gold fetch failed (non-critical): %s", e)

    return result


# ---------------------------------------------------------------------------
# Static seed data
# ---------------------------------------------------------------------------

def _load_or_seed_external_research():
    """Load cached external research or seed with known data."""
    existing = load_json("data/gold_external_cache.json")
    if existing and existing.get("analyst_targets"):
        return existing

    return {
        "last_updated": "2026-03-15",
        "analyst_targets": {
            "goldman_sachs": {
                "target": 4900,
                "timeframe": "Dec 2026",
                "note": "Structural bull target",
            },
            "jpm": {
                "target_avg": 5055,
                "target_peak": 5300,
                "note": "Q4 avg $5,055, peaks $5,200-5,300",
            },
            "bank_of_america": {
                "target": 5000,
                "note": "Clear path to $5,000",
            },
            "citi": {
                "target": 5000,
                "timeframe": "2026",
                "note": "In line with consensus",
            },
            "ubs": {
                "target": 4800,
                "note": "Conservative, below consensus",
            },
        },
        "central_bank_buying": {
            "2024_tonnes": 1037,
            "2023_tonnes": 1037,
            "2022_tonnes": 1136,
            "trend": "Record pace since 2022, 3rd consecutive year >1000 tonnes",
            "top_buyers": ["China (PBOC)", "Poland", "Turkey", "India", "Czech Republic"],
            "note": "De-dollarization + sanctions risk driving diversification from USD reserves",
        },
        "supply_demand": {
            "mine_production_tonnes_2024": 3644,
            "total_supply_tonnes_2024": 4974,
            "total_demand_tonnes_2024": 4898,
            "jewelry_demand_tonnes": 2086,
            "investment_demand_tonnes": 1180,
            "central_bank_tonnes": 1037,
            "technology_tonnes": 326,
        },
        "macro_drivers": {
            "primary": "Real yields (10Y - CPI). Negative real yields = bullish.",
            "secondary": "DXY inverse correlation. Dollar weakness = gold strength.",
            "structural": "Central bank buying (de-dollarization), fiscal deficits ($36T US debt)",
            "geopolitical": "Ukraine/Russia, Middle East, US-China tensions — safe haven bid",
        },
        "key_risks": [
            "Rising real yields if inflation cools faster than Fed cuts",
            "DXY strength above 105 (significant headwind)",
            "Profit-taking from ATH — gold near $5,000+",
            "Crowded long positioning in COT if specs pile in",
            "Fed hawkish pivot (fewer cuts than priced)",
        ],
    }


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

    return ratios[-30:]


if __name__ == "__main__":
    precompute()
