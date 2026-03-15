"""Metals Deep Context Precomputer -- Silver + Gold consolidated.

Fetches shared market data once, writes both silver_deep_context.json and
gold_deep_context.json.  Tracks COT history for trend analysis.  Fetches FRED
real yields.

Run manually: .venv/Scripts/python.exe portfolio/metals_precompute.py
Auto-runs every 4h via main loop.
"""

import datetime
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio.file_utils import (
    atomic_append_jsonl,
    atomic_write_json,
    load_json,
    load_jsonl,
)

logger = logging.getLogger("portfolio.metals_precompute")

_STATE_FILE = "data/metals_precompute_state.json"
_REFRESH_STATE_FILE = "data/metals_refresh_state.json"
_COT_HISTORY_FILE = "data/cot_history.jsonl"
_DEFAULT_INTERVAL_SEC = 4 * 3600  # 4 hours

# Per-source refresh intervals (seconds)
_REFRESH_INTERVALS = {
    "gld": 24 * 3600,             # GLD ETF: daily
    "slv": 24 * 3600,             # SLV ETF: daily
    "gold_futures": 24 * 3600,    # Gold futures (GC=F): daily
    "silver_futures": 24 * 3600,  # Silver futures (SI=F): daily
    "cot": 7 * 24 * 3600,        # CFTC COT: weekly
    "fred": 24 * 3600,           # FRED yields: daily
}

_REQUEST_TIMEOUT = 15  # seconds per HTTP request


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def maybe_precompute_metals(config=None):
    """Run metals precompute if enough time has elapsed since last run.

    Called from main loop's _run_post_cycle().  Self-checking -- safe to call
    every cycle; will only execute when interval has elapsed.
    """
    interval = _DEFAULT_INTERVAL_SEC
    if config:
        interval = config.get("metals", {}).get(
            "precompute_interval_sec", _DEFAULT_INTERVAL_SEC
        )

    state = load_json(_STATE_FILE, default={})
    last_run = state.get("last_run_epoch", 0)
    now = time.time()

    if (now - last_run) < interval:
        return None  # Not yet

    try:
        result = precompute(config)
        atomic_write_json(_STATE_FILE, {
            "last_run_epoch": now,
            "last_run_iso": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "ok",
        })
        logger.info("Metals precompute completed (interval=%ds)", interval)
        return result
    except Exception as e:
        logger.warning("Metals precompute failed: %s", e)
        atomic_write_json(_STATE_FILE, {
            "last_run_epoch": last_run,  # Keep old timestamp so we retry soon
            "last_run_iso": state.get("last_run_iso", ""),
            "status": f"error: {e}",
            "last_error_epoch": now,
        })
        return None


def precompute(config=None):
    """Aggregate all metals context into cached JSON files.

    Fetches shared market data ONCE, then writes both:
      - data/silver_deep_context.json
      - data/gold_deep_context.json
    """
    generated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Fetch all shared market data in one pass
    market = _fetch_market_data(config)

    # Record COT history
    _record_cot_history(market.get("cot_silver"), "silver")
    _record_cot_history(market.get("cot_gold"), "gold")

    # Build silver context
    silver_ctx = _build_silver_context(market, generated_at)
    atomic_write_json("data/silver_deep_context.json", silver_ctx)
    print(f"Silver deep context written to data/silver_deep_context.json")
    print(f"  Generated: {generated_at}")
    print(f"  Sections: {list(silver_ctx.keys())}")

    # Build gold context
    gold_ctx = _build_gold_context(market, generated_at)
    atomic_write_json("data/gold_deep_context.json", gold_ctx)
    print(f"Gold deep context written to data/gold_deep_context.json")
    print(f"  Generated: {generated_at}")
    print(f"  Sections: {list(gold_ctx.keys())}")

    return {"silver": silver_ctx, "gold": gold_ctx}


# ---------------------------------------------------------------------------
# Shared market data fetcher
# ---------------------------------------------------------------------------

def _fetch_market_data(config=None):
    """Fetch all shared market data once and return as dict.

    Each sub-fetch is wrapped in _safe_fetch() and has per-source interval
    checking via _REFRESH_STATE_FILE.
    """
    refresh_state = load_json(_REFRESH_STATE_FILE, default={})
    now = time.time()
    refreshed = []

    result = {
        "gc_f": None,
        "si_f": None,
        "gld": None,
        "slv": None,
        "cot_gold": None,
        "cot_silver": None,
        "gs_ratio": None,
        "fred": None,
    }

    # 1. Gold futures (GC=F) -- 3mo OHLCV + Fibonacci + SMAs
    if _should_refresh(refresh_state, "gold_futures", now):
        data = _safe_fetch("Gold futures", _fetch_futures, "GC=F")
        if data:
            result["gc_f"] = data
            refresh_state["gold_futures"] = {"ts": now, "ok": True}
            refreshed.append("gold_futures")
        else:
            refresh_state["gold_futures"] = {
                **refresh_state.get("gold_futures", {}),
                "last_error_ts": now,
            }

    # 2. Silver futures (SI=F) -- 3mo OHLCV + Fibonacci + SMAs
    if _should_refresh(refresh_state, "silver_futures", now):
        data = _safe_fetch("Silver futures", _fetch_futures, "SI=F")
        if data:
            result["si_f"] = data
            refresh_state["silver_futures"] = {"ts": now, "ok": True}
            refreshed.append("silver_futures")
        else:
            refresh_state["silver_futures"] = {
                **refresh_state.get("silver_futures", {}),
                "last_error_ts": now,
            }

    # 3. GLD ETF -- 1mo data + volume trend
    if _should_refresh(refresh_state, "gld", now):
        data = _safe_fetch("GLD ETF", _fetch_etf_data, "GLD")
        if data:
            result["gld"] = data
            refresh_state["gld"] = {"ts": now, "ok": True}
            refreshed.append("gld")
        else:
            refresh_state["gld"] = {
                **refresh_state.get("gld", {}),
                "last_error_ts": now,
            }

    # 4. SLV ETF -- 1mo data + volume trend
    if _should_refresh(refresh_state, "slv", now):
        data = _safe_fetch("SLV ETF", _fetch_etf_data, "SLV")
        if data:
            result["slv"] = data
            refresh_state["slv"] = {"ts": now, "ok": True}
            refreshed.append("slv")
        else:
            refresh_state["slv"] = {
                **refresh_state.get("slv", {}),
                "last_error_ts": now,
            }

    # 5. CFTC COT for gold
    if _should_refresh(refresh_state, "cot", now):
        cot_gold = _safe_fetch("CFTC COT Gold", _fetch_cftc_cot, "GOLD")
        cot_silver = _safe_fetch("CFTC COT Silver", _fetch_cftc_cot, "SILVER")
        if cot_gold or cot_silver:
            result["cot_gold"] = cot_gold
            result["cot_silver"] = cot_silver
            refresh_state["cot"] = {"ts": now, "ok": True}
            refreshed.append("cot")
        else:
            refresh_state["cot"] = {
                **refresh_state.get("cot", {}),
                "last_error_ts": now,
            }

    # 6. FRED real yields
    fred_api_key = None
    if config:
        fred_api_key = config.get("golddigger", {}).get("fred_api_key")
    if _should_refresh(refresh_state, "fred", now):
        data = _safe_fetch("FRED yields", _fetch_fred_data, fred_api_key)
        if data:
            result["fred"] = data
            refresh_state["fred"] = {"ts": now, "ok": True}
            refreshed.append("fred")
        else:
            refresh_state["fred"] = {
                **refresh_state.get("fred", {}),
                "last_error_ts": now,
            }

    # Compute G/S ratio from futures data
    gc_price = None
    si_price = None
    if result["gc_f"]:
        gc_price = result["gc_f"].get("current")
    if result["si_f"]:
        si_price = result["si_f"].get("current")
    if gc_price and si_price and si_price > 0:
        result["gs_ratio"] = round(gc_price / si_price, 2)

    # Persist refresh state
    if refreshed:
        atomic_write_json(_REFRESH_STATE_FILE, refresh_state)
        logger.info("Metals market data refreshed: %s", ", ".join(refreshed))

    # Attach refresh status so consumers know what's fresh
    result["refresh_status"] = {
        src: {
            "age_hours": round((now - s.get("ts", 0)) / 3600, 1),
            "ok": s.get("ok", False),
        }
        for src, s in refresh_state.items()
    }
    result["refreshed_sources"] = refreshed

    return result


# ---------------------------------------------------------------------------
# Individual data fetchers (shared)
# ---------------------------------------------------------------------------

def _should_refresh(state, key, now):
    """Check if a source is due for refresh."""
    interval = _REFRESH_INTERVALS.get(key, 24 * 3600)
    last = state.get(key, {}).get("ts", 0)
    return (now - last) >= interval


def _safe_fetch(name, fetch_fn, *args):
    """Call a fetch function with error handling.  Returns None on failure."""
    try:
        return fetch_fn(*args)
    except Exception as e:
        logger.warning("Auto-refresh %s failed: %s", name, e)
        return None


def _fetch_futures(symbol):
    """Fetch futures price context (GC=F or SI=F) via yfinance.

    Returns 3-month OHLCV with Fibonacci retracements and SMAs.
    """
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="3mo")

    if hist.empty:
        return None

    closes = hist["Close"].dropna()
    if len(closes) < 5:
        return None

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

    sma20 = round(float(closes.tail(20).mean()), 2) if len(closes) >= 20 else None
    sma50 = round(float(closes.tail(50).mean()), 2) if len(closes) >= 50 else None

    change_1w = round(
        float((current / closes.iloc[-5] - 1) * 100), 2
    ) if len(closes) >= 5 else None
    change_1mo = round(
        float((current / closes.iloc[-20] - 1) * 100), 2
    ) if len(closes) >= 20 else None

    return {
        "symbol": symbol,
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
        ) if high_3mo > 0 else None,
        "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def _fetch_etf_data(symbol):
    """Fetch ETF data (GLD or SLV) via yfinance -- investment flow proxy."""
    import yfinance as yf

    etf = yf.Ticker(symbol)
    hist = etf.history(period="1mo")

    if hist.empty:
        return None

    closes = hist["Close"].dropna()
    volumes = hist["Volume"].dropna()

    if len(closes) < 2:
        return None

    result = {
        "symbol": symbol,
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

    # Volume trend (last 5d vs 1mo avg)
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


def _fetch_cftc_cot(commodity_name):
    """Fetch CFTC COT data from the SOCRATA Open Data API.

    Args:
        commodity_name: 'GOLD' or 'SILVER' (case-sensitive for CFTC API).

    Returns legacy + disaggregated data for the requested commodity.
    """
    import requests

    # Legacy Futures Only report
    url = (
        "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
        f"?$where=commodity_name='{commodity_name}'"
        "&$order=report_date_as_yyyy_mm_dd DESC"
        "&$limit=8"
    )

    resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()
    rows = resp.json()

    if not rows:
        return None

    latest = rows[0]
    prev = rows[1] if len(rows) > 1 else {}

    oi = _int(latest.get("open_interest_all"))
    nc_long = _int(latest.get("noncomm_positions_long_all"))
    nc_short = _int(latest.get("noncomm_positions_short_all"))
    comm_long = _int(latest.get("comm_positions_long_all"))
    comm_short = _int(latest.get("comm_positions_short_all"))

    nc_net = (nc_long - nc_short) if nc_long is not None and nc_short is not None else None
    comm_net = (comm_long - comm_short) if comm_long is not None and comm_short is not None else None

    prev_nc_long = _int(prev.get("noncomm_positions_long_all"))
    prev_nc_short = _int(prev.get("noncomm_positions_short_all"))
    prev_nc_net = (
        (prev_nc_long - prev_nc_short)
        if prev_nc_long is not None and prev_nc_short is not None
        else None
    )

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

    # Disaggregated report for managed money
    try:
        disagg_url = (
            "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
            f"?$where=commodity_name='{commodity_name}'"
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
        logger.debug(
            "Disaggregated COT %s fetch failed (non-critical): %s",
            commodity_name, e,
        )

    return result


def _fetch_fred_data(api_key):
    """Fetch FRED data for real yield computation.

    Returns 10Y Treasury yield, CPI, and computed real yield.
    """
    import requests

    if not api_key:
        logger.debug("FRED API key not configured, skipping")
        return None

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Fetch DGS10 (10-Year Treasury Constant Maturity Rate)
    dgs10_url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id=DGS10&api_key={api_key}"
        "&sort_order=desc&limit=30&file_type=json"
    )
    resp = requests.get(dgs10_url, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()
    obs = resp.json().get("observations", [])

    # Filter out "." entries (no data for that day)
    dgs10_vals = []
    for o in obs:
        val = o.get("value", ".")
        if val != ".":
            try:
                dgs10_vals.append({"date": o["date"], "value": float(val)})
            except (ValueError, KeyError):
                continue

    if not dgs10_vals:
        return None

    treasury_10y = dgs10_vals[0]["value"]
    # Find value from ~1 month ago
    treasury_10y_1mo_ago = dgs10_vals[-1]["value"] if len(dgs10_vals) > 1 else treasury_10y

    # Fetch CPIAUCSL (Consumer Price Index, seasonally adjusted)
    cpi_url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id=CPIAUCSL&api_key={api_key}"
        "&sort_order=desc&limit=13&file_type=json"
    )
    cpi_resp = requests.get(cpi_url, timeout=_REQUEST_TIMEOUT)
    cpi_resp.raise_for_status()
    cpi_obs = cpi_resp.json().get("observations", [])

    cpi_latest = None
    cpi_annual_rate = None
    if len(cpi_obs) >= 2:
        try:
            current_cpi = float(cpi_obs[0]["value"])
            cpi_latest = current_cpi

            # Compute YoY CPI change if we have 12+ months of data
            if len(cpi_obs) >= 13:
                cpi_1y_ago = float(cpi_obs[12]["value"])
                cpi_annual_rate = round(
                    (current_cpi / cpi_1y_ago - 1) * 100, 2
                )
        except (ValueError, KeyError, IndexError):
            pass

    real_yield = None
    if treasury_10y is not None and cpi_annual_rate is not None:
        real_yield = round(treasury_10y - cpi_annual_rate, 3)

    # Determine direction from recent trend
    real_yield_direction = None
    if len(dgs10_vals) >= 5:
        recent_avg = sum(v["value"] for v in dgs10_vals[:5]) / 5
        older_avg = sum(v["value"] for v in dgs10_vals[5:10]) / min(
            5, len(dgs10_vals[5:10])
        ) if len(dgs10_vals) > 5 else recent_avg
        if recent_avg > older_avg + 0.05:
            real_yield_direction = "rising"
        elif recent_avg < older_avg - 0.05:
            real_yield_direction = "falling"
        else:
            real_yield_direction = "stable"

    return {
        "treasury_10y": treasury_10y,
        "treasury_10y_date": dgs10_vals[0]["date"],
        "treasury_10y_1mo_ago": treasury_10y_1mo_ago,
        "cpi_latest": cpi_latest,
        "cpi_annual_rate": cpi_annual_rate,
        "real_yield": real_yield,
        "real_yield_direction": real_yield_direction,
        "fetched_at": now_iso,
    }


# ---------------------------------------------------------------------------
# COT history tracking
# ---------------------------------------------------------------------------

def _record_cot_history(cot_data, metal):
    """Append COT data to history file, deduplicating by (metal, report_date)."""
    if not cot_data:
        return

    report_date = cot_data.get("report_date")
    if not report_date:
        return

    # Check for duplicates
    existing = load_jsonl(_COT_HISTORY_FILE)
    for entry in existing:
        if entry.get("metal") == metal and entry.get("report_date") == report_date:
            return  # Already logged

    record = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "metal": metal,
        "report_date": report_date,
        "oi": cot_data.get("open_interest"),
        "nc_net": cot_data.get("noncomm_net"),
        "comm_net": cot_data.get("comm_net"),
        "mm_net": cot_data.get("managed_money_net"),
    }

    atomic_append_jsonl(_COT_HISTORY_FILE, record)
    logger.info("COT history: appended %s %s", metal, report_date)


def _compute_cot_trend(metal):
    """Compute COT trend from historical data for a given metal."""
    entries = load_jsonl(_COT_HISTORY_FILE)
    metal_entries = [e for e in entries if e.get("metal") == metal]

    if not metal_entries:
        return {}

    # Sort by report_date
    metal_entries.sort(key=lambda e: e.get("report_date", ""))

    # Keep last 8 weeks
    recent = metal_entries[-8:]

    result = {
        "weeks_available": len(recent),
        "history": recent,
    }

    # Compute direction from last 3 weeks
    if len(recent) >= 3:
        last_3 = recent[-3:]
        nc_nets = [e.get("nc_net") for e in last_3 if e.get("nc_net") is not None]
        mm_nets = [e.get("mm_net") for e in last_3 if e.get("mm_net") is not None]

        if len(nc_nets) >= 2:
            nc_change = nc_nets[-1] - nc_nets[0]
            result["nc_net_direction"] = (
                "rising" if nc_change > 0
                else "falling" if nc_change < 0
                else "flat"
            )
            result["nc_net_3w_change"] = nc_change

        if len(mm_nets) >= 2:
            mm_change = mm_nets[-1] - mm_nets[0]
            result["mm_net_direction"] = (
                "rising" if mm_change > 0
                else "falling" if mm_change < 0
                else "flat"
            )
            result["mm_net_3w_change"] = mm_change

    return result


# ---------------------------------------------------------------------------
# Per-metal context builders
# ---------------------------------------------------------------------------

def _build_silver_context(market, generated_at):
    """Build silver_deep_context.json from shared market data."""
    context = {
        "generated_at": generated_at,
        "version": 2,
    }

    # 1. Signal accuracy
    context["signal_accuracy"] = _compute_signal_accuracy("XAG-USD")

    # 2. Price trajectory
    context["price_trajectory"] = _compute_price_trajectory("XAG-USD", "xag")

    # 3. Journal history
    context["journal_history"] = _extract_journal_entries("XAG-USD", "xag")

    # 4. Rolling performance
    context["performance"] = _compute_rolling_performance("XAG")

    # 5. External research with live data overlay
    base_research = _load_silver_seed_research()
    external = dict(base_research)

    # Overlay live market data
    if market.get("slv"):
        external["slv_etf"] = market["slv"]
    if market.get("si_f"):
        external["futures_context"] = market["si_f"]
    if market.get("cot_silver"):
        existing_cot = external.get("cot_positioning", {})
        external["cot_positioning"] = {**existing_cot, "live": market["cot_silver"]}
    if market.get("fred"):
        external["fred"] = market["fred"]
    if market.get("gs_ratio") is not None:
        external["gs_ratio_live"] = market["gs_ratio"]

    # Gold context for ratio analysis
    if market.get("gc_f"):
        gc_current = market["gc_f"].get("current")
        if gc_current:
            external["gold_context"] = {
                "gold_price": gc_current,
                "gs_ratio": market.get("gs_ratio"),
                "gs_ratio_vs_historical": _gs_ratio_label(market.get("gs_ratio")),
                "implied_silver_at_50": round(gc_current / 50, 2),
                "implied_silver_at_40": round(gc_current / 40, 2),
            }

    # COT trend
    cot_trend = _compute_cot_trend("silver")
    if cot_trend:
        external["cot_trend"] = cot_trend

    # Refresh status
    external["refresh_status"] = market.get("refresh_status", {})
    if market.get("refreshed_sources"):
        external["last_auto_refresh"] = generated_at
        external["auto_refreshed_sources"] = market["refreshed_sources"]

    context["external_research"] = external

    # 6. Prophecy
    context["prophecy"] = _extract_prophecy("silver_bull_2026")

    # 7. G/S ratio history
    context["gs_ratio_history"] = _compute_gs_ratio_history()

    return context


def _build_gold_context(market, generated_at):
    """Build gold_deep_context.json from shared market data."""
    context = {
        "generated_at": generated_at,
        "version": 2,
    }

    # 1. Signal accuracy
    context["signal_accuracy"] = _compute_signal_accuracy("XAU-USD")

    # 2. Price trajectory
    context["price_trajectory"] = _compute_price_trajectory("XAU-USD", "xau")

    # 3. Journal history
    context["journal_history"] = _extract_journal_entries("XAU-USD", "xau")

    # 4. External research with live data overlay
    base_research = _load_gold_seed_research()
    external = dict(base_research)

    # Overlay live market data
    if market.get("gld"):
        external["gld_etf"] = market["gld"]
    if market.get("gc_f"):
        external["futures_context"] = market["gc_f"]
    if market.get("cot_gold"):
        existing_cot = external.get("cot_positioning", {})
        external["cot_positioning"] = {**existing_cot, "live": market["cot_gold"]}
    if market.get("fred"):
        external["fred"] = market["fred"]
    if market.get("gs_ratio") is not None:
        external["gs_ratio_live"] = market["gs_ratio"]

    # Silver context for ratio analysis
    if market.get("si_f"):
        si_current = market["si_f"].get("current")
        gc_current = market["gc_f"].get("current") if market.get("gc_f") else None
        if si_current and gc_current:
            external["silver_context"] = {
                "silver_price": si_current,
                "gs_ratio": market.get("gs_ratio"),
                "implied_silver_at_50": round(gc_current / 50, 2),
                "implied_silver_at_40": round(gc_current / 40, 2),
            }

    # COT trend
    cot_trend = _compute_cot_trend("gold")
    if cot_trend:
        external["cot_trend"] = cot_trend

    # Refresh status
    external["refresh_status"] = market.get("refresh_status", {})
    if market.get("refreshed_sources"):
        external["last_auto_refresh"] = generated_at
        external["auto_refreshed_sources"] = market["refreshed_sources"]

    context["external_research"] = external

    # 5. G/S ratio history
    context["gs_ratio_history"] = _compute_gs_ratio_history()

    return context


# ---------------------------------------------------------------------------
# Generic helpers (shared by both metals)
# ---------------------------------------------------------------------------

def _compute_signal_accuracy(ticker):
    """Extract per-ticker signal accuracy from agent_summary_compact.json."""
    summary = load_json("data/agent_summary_compact.json")
    if not summary:
        return {}

    reliability = summary.get("signal_reliability", {}).get(ticker, {})
    focus_probs = summary.get("focus_probabilities", {}).get(ticker, {})
    forecast = summary.get("forecast_signals", {}).get(ticker, {})
    forecast_acc = summary.get("forecast_accuracy", {}).get("accuracy", {})

    chronos_data = {}
    for key in ("chronos_24h", "kronos_24h"):
        by_ticker = forecast_acc.get(key, {}).get("by_ticker", {})
        data = by_ticker.get(ticker, {}) if isinstance(by_ticker, dict) else {}
        if data:
            chronos_data[key] = data

    return {
        "best_signals": reliability.get("best", {}),
        "worst_signals": reliability.get("worst", {}),
        "focus_probabilities": focus_probs,
        "forecast": forecast,
        "chronos_accuracy": chronos_data,
    }


def _compute_price_trajectory(ticker, price_key):
    """Extract price history from price_snapshots_hourly.jsonl.

    Args:
        ticker: e.g. 'XAG-USD' or 'XAU-USD'
        price_key: short key for output, e.g. 'xag' or 'xau'
    """
    from pathlib import Path

    path = Path("data/price_snapshots_hourly.jsonl")
    if not path.exists():
        return {}

    other_key = "xau" if price_key == "xag" else "xag"
    other_ticker = "XAU-USD" if ticker == "XAG-USD" else "XAG-USD"

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
                    main_price = price_data.get(ticker)
                    other_price = price_data.get(other_ticker)
                    if main_price is not None:
                        xag = price_data.get("XAG-USD")
                        xau = price_data.get("XAU-USD")
                        prices.append({
                            "ts": entry.get("ts", ""),
                            price_key: main_price,
                            other_key: other_price,
                            "gs_ratio": round(xau / xag, 2) if xau and xag and xag > 0 else None,
                        })
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
    except OSError as e:
        logger.warning("Failed to read price snapshots: %s", e)
        return {}

    if not prices:
        return {}

    # Keep last 7 days of hourly data
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=7)
    cutoff_iso = cutoff.isoformat()
    recent = [p for p in prices if p["ts"] >= cutoff_iso]

    if len(recent) < 2:
        return {"data_points": len(recent)}

    first_price = recent[0][price_key]
    last_price = recent[-1][price_key]
    high = max(p[price_key] for p in recent)
    low = min(p[price_key] for p in recent)

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
            {"ts": p["ts"], price_key: p[price_key], "gs": p.get("gs_ratio")}
            for p in recent[::6]
        ],
    }


def _extract_journal_entries(ticker, prefix):
    """Extract journal entries that mention the given ticker."""
    entries = load_jsonl("data/layer2_journal.jsonl")
    if not entries:
        return []

    matching = []
    for entry in entries:
        has_ticker = (
            ticker in entry.get("tickers", {})
            or ticker in entry.get("prices", {})
        )
        if has_ticker:
            ticker_data = entry.get("tickers", {}).get(ticker, {})
            matching.append({
                "ts": entry.get("ts"),
                "regime": entry.get("regime"),
                f"{prefix}_outlook": ticker_data.get("outlook"),
                f"{prefix}_thesis": ticker_data.get("thesis"),
                f"{prefix}_conviction": ticker_data.get("conviction"),
                f"{prefix}_price": entry.get("prices", {}).get(ticker),
            })

    # Return last 20 entries
    return matching[-20:]


def _compute_rolling_performance(symbol_prefix):
    """Compute rolling signal performance from metals_signal_log.jsonl."""
    entries = load_jsonl("data/metals_signal_log.jsonl")
    if not entries:
        return {}

    signal_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
    last_entries = []
    for entry in entries:
        entry_str = json.dumps(entry)
        if symbol_prefix in entry_str:
            action = entry.get("action", entry.get("consensus", "HOLD"))
            if action in signal_counts:
                signal_counts[action] += 1
            last_entries.append({"ts": entry.get("ts", ""), "action": action})

    return {
        "total_signals": sum(signal_counts.values()),
        "signal_distribution": signal_counts,
        "last_10_signals": last_entries[-10:],
    }


def _extract_prophecy(belief_id):
    """Extract prophecy data for a specific belief."""
    prophecy = load_json("data/prophecy.json")
    if not prophecy:
        return {}

    beliefs = prophecy.get("beliefs", [])
    if not isinstance(beliefs, list):
        return {}

    for belief in beliefs:
        if not isinstance(belief, dict):
            continue
        if belief.get("id") == belief_id:
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


def _gs_ratio_label(gs_ratio):
    """Human-readable G/S ratio assessment."""
    if gs_ratio is None:
        return None
    if gs_ratio > 80:
        return "severely undervalued"
    if gs_ratio > 65:
        return "undervalued"
    if gs_ratio > 50:
        return "fair value"
    if gs_ratio > 35:
        return "overvalued"
    return "extremely overvalued"


def _int(val):
    """Safe int conversion."""
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Seed data (static research for each metal)
# ---------------------------------------------------------------------------

def _load_silver_seed_research():
    """Load cached external research or seed with known data for silver."""
    existing = load_json("data/silver_external_cache.json")
    if existing and existing.get("analyst_targets"):
        return existing

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
            "CME margin hikes (Jan +47%, Feb +15-18%) -- caused 31% crash from $121",
            "DXY strength above 100",
            "Solar thrifting -7% per-panel YoY",
            "Substitution R&D accelerating at $80+",
            "Exhaustion signals -- Iran escalation didn't move silver",
        ],
    }


def _load_gold_seed_research():
    """Load cached external research or seed with known data for gold."""
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
            "geopolitical": "Ukraine/Russia, Middle East, US-China tensions -- safe haven bid",
        },
        "key_risks": [
            "Rising real yields if inflation cools faster than Fed cuts",
            "DXY strength above 105 (significant headwind)",
            "Profit-taking from ATH -- gold near $5,000+",
            "Crowded long positioning in COT if specs pile in",
            "Fed hawkish pivot (fewer cuts than priced)",
        ],
    }


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = load_json("config.json")
    precompute(config)
