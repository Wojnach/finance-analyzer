"""Oil Deep Context Precomputer — WTI + Brent quant signals.

Fetches WTI/Brent futures, OVX, CFTC COT for crude oil, FRED macro,
USO ETF flows, and crack spread.  Computes TSMOM, Donchian, MA crossovers,
realised vol, and term structure signals.

Run manually: .venv/Scripts/python.exe portfolio/oil_precompute.py
Auto-runs every 2h via main loop.
"""

import datetime
import logging
import math
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

logger = logging.getLogger("portfolio.oil_precompute")

_STATE_FILE = "data/oil_precompute_state.json"
_REFRESH_STATE_FILE = "data/oil_refresh_state.json"
_COT_HISTORY_FILE = "data/oil_cot_history.jsonl"
_OUTPUT_FILE = "data/oil_deep_context.json"
_DEFAULT_INTERVAL_SEC = 2 * 3600  # 2 hours

# Per-source refresh intervals (seconds)
_REFRESH_INTERVALS = {
    "wti_futures": 4 * 3600,
    "brent_futures": 4 * 3600,
    "ovx": 4 * 3600,
    "uso_etf": 4 * 3600,
    "rbob_futures": 4 * 3600,
    "cot": 7 * 24 * 3600,       # CFTC COT: weekly
    "fred": 24 * 3600,           # FRED: daily
}

_REQUEST_TIMEOUT = 15


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def maybe_precompute_oil(config=None):
    """Run oil precompute if enough time has elapsed since last run.

    Called from main loop's _run_post_cycle().  Self-checking — safe to call
    every cycle; will only execute when interval has elapsed.
    """
    interval = _DEFAULT_INTERVAL_SEC
    if config:
        interval = config.get("oil", {}).get(
            "precompute_interval_sec", _DEFAULT_INTERVAL_SEC
        )

    state = load_json(_STATE_FILE, default={})
    last_run = state.get("last_run_epoch", 0)
    now = time.time()

    if (now - last_run) < interval:
        return None

    try:
        result = precompute(config)
        atomic_write_json(_STATE_FILE, {
            "last_run_epoch": now,
            "last_run_iso": datetime.datetime.now(datetime.UTC).isoformat(),
            "status": "ok",
        })
        logger.info("Oil precompute completed (interval=%ds)", interval)
        return result
    except Exception as e:
        logger.warning("Oil precompute failed: %s", e)
        atomic_write_json(_STATE_FILE, {
            "last_run_epoch": last_run,
            "last_run_iso": state.get("last_run_iso", ""),
            "status": f"error: {e}",
            "last_error_epoch": now,
        })
        return None


def precompute(config=None):
    """Aggregate all oil context into data/oil_deep_context.json."""
    generated_at = datetime.datetime.now(datetime.UTC).isoformat()

    market = _fetch_market_data(config)

    # Record COT history
    _record_cot_history(market.get("cot_crude"))

    # Build oil context
    ctx = _build_oil_context(market, generated_at)
    atomic_write_json(_OUTPUT_FILE, ctx)
    print(f"Oil deep context written to {_OUTPUT_FILE}")
    print(f"  Generated: {generated_at}")
    print(f"  Sections: {list(ctx.keys())}")

    return ctx


# ---------------------------------------------------------------------------
# Shared market data fetcher
# ---------------------------------------------------------------------------

def _fetch_market_data(config=None):
    """Fetch all oil market data once and return as dict."""
    refresh_state = load_json(_REFRESH_STATE_FILE, default={})
    now = time.time()
    refreshed = []

    result = {
        "cl_f": None,       # WTI futures (CL=F)
        "bz_f": None,       # Brent futures (BZ=F)
        "ovx": None,        # Oil VIX (^OVX)
        "uso": None,        # USO ETF
        "rb_f": None,       # RBOB gasoline (RB=F)
        "cot_crude": None,  # CFTC COT crude oil
        "fred": None,       # FRED macro data
    }

    # 1. WTI futures (CL=F)
    if _should_refresh(refresh_state, "wti_futures", now):
        data = _safe_fetch("WTI futures", _fetch_oil_futures, "CL=F", "6mo")
        if data:
            result["cl_f"] = data
            refresh_state["wti_futures"] = {"ts": now, "ok": True}
            refreshed.append("wti_futures")
        else:
            refresh_state["wti_futures"] = {
                **refresh_state.get("wti_futures", {}),
                "last_error_ts": now,
            }

    # 2. Brent futures (BZ=F)
    if _should_refresh(refresh_state, "brent_futures", now):
        data = _safe_fetch("Brent futures", _fetch_oil_futures, "BZ=F", "6mo")
        if data:
            result["bz_f"] = data
            refresh_state["brent_futures"] = {"ts": now, "ok": True}
            refreshed.append("brent_futures")
        else:
            refresh_state["brent_futures"] = {
                **refresh_state.get("brent_futures", {}),
                "last_error_ts": now,
            }

    # 3. OVX (Oil VIX)
    if _should_refresh(refresh_state, "ovx", now):
        data = _safe_fetch("OVX", _fetch_ovx)
        if data:
            result["ovx"] = data
            refresh_state["ovx"] = {"ts": now, "ok": True}
            refreshed.append("ovx")
        else:
            refresh_state["ovx"] = {
                **refresh_state.get("ovx", {}),
                "last_error_ts": now,
            }

    # 4. USO ETF
    if _should_refresh(refresh_state, "uso_etf", now):
        data = _safe_fetch("USO ETF", _fetch_etf_data, "USO")
        if data:
            result["uso"] = data
            refresh_state["uso_etf"] = {"ts": now, "ok": True}
            refreshed.append("uso_etf")
        else:
            refresh_state["uso_etf"] = {
                **refresh_state.get("uso_etf", {}),
                "last_error_ts": now,
            }

    # 5. RBOB gasoline (for crack spread)
    if _should_refresh(refresh_state, "rbob_futures", now):
        data = _safe_fetch("RBOB futures", _fetch_oil_futures, "RB=F", "3mo")
        if data:
            result["rb_f"] = data
            refresh_state["rbob_futures"] = {"ts": now, "ok": True}
            refreshed.append("rbob_futures")
        else:
            refresh_state["rbob_futures"] = {
                **refresh_state.get("rbob_futures", {}),
                "last_error_ts": now,
            }

    # 6. CFTC COT for crude oil
    if _should_refresh(refresh_state, "cot", now):
        cot = _safe_fetch("CFTC COT Crude Oil", _fetch_cftc_cot_crude)
        if cot:
            result["cot_crude"] = cot
            refresh_state["cot"] = {"ts": now, "ok": True}
            refreshed.append("cot")
        else:
            refresh_state["cot"] = {
                **refresh_state.get("cot", {}),
                "last_error_ts": now,
            }

    # 7. FRED macro data (oil-specific series)
    fred_api_key = None
    if config:
        fred_api_key = config.get("golddigger", {}).get("fred_api_key")
    if _should_refresh(refresh_state, "fred", now):
        data = _safe_fetch("FRED oil data", _fetch_fred_oil, fred_api_key)
        if data:
            result["fred"] = data
            refresh_state["fred"] = {"ts": now, "ok": True}
            refreshed.append("fred")
        else:
            refresh_state["fred"] = {
                **refresh_state.get("fred", {}),
                "last_error_ts": now,
            }

    # Persist refresh state
    if refreshed:
        atomic_write_json(_REFRESH_STATE_FILE, refresh_state)
        logger.info("Oil market data refreshed: %s", ", ".join(refreshed))

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
# Individual data fetchers
# ---------------------------------------------------------------------------

def _should_refresh(state, key, now):
    """Check if a source is due for refresh."""
    interval = _REFRESH_INTERVALS.get(key, 24 * 3600)
    last = state.get(key, {}).get("ts", 0)
    return (now - last) >= interval


def _safe_fetch(name, fetch_fn, *args):
    """Call a fetch function with error handling."""
    try:
        return fetch_fn(*args)
    except Exception as e:
        logger.warning("Oil auto-refresh %s failed: %s", name, e)
        return None


def _fetch_oil_futures(symbol, period="6mo"):
    """Fetch oil futures with extended quant signals.

    Returns price data + TSMOM + Donchian + MA crossovers + realised vol +
    Fibonacci + RSI.
    """
    # 2026-04-14: route via price_source (CL=F → Binance FAPI, USO → Alpaca)
    from portfolio.price_source import fetch_klines

    _LIMIT = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730}
    limit = _LIMIT.get(period, 90)
    hist = fetch_klines(symbol, interval="1d", limit=limit, period=period)

    if hist is None or hist.empty:
        return None

    closes = hist["close"].dropna()
    volumes = hist["volume"].dropna()
    if len(closes) < 10:
        return None

    current = float(closes.iloc[-1])
    high_3mo = float(closes.max())
    low_3mo = float(closes.min())

    # --- Basic price context ---
    sma20 = round(float(closes.tail(20).mean()), 2) if len(closes) >= 20 else None
    sma50 = round(float(closes.tail(50).mean()), 2) if len(closes) >= 50 else None
    sma200 = round(float(closes.tail(200).mean()), 2) if len(closes) >= 200 else None

    change_1w = round(
        float((current / closes.iloc[-5] - 1) * 100), 2
    ) if len(closes) >= 5 else None
    change_1mo = round(
        float((current / closes.iloc[-20] - 1) * 100), 2
    ) if len(closes) >= 20 else None

    # --- Fibonacci retracement from 3mo high/low ---
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

    # --- RSI(14) ---
    rsi_14 = _compute_rsi(closes, 14)

    # --- TSMOM: Multi-horizon momentum ---
    tsmom = {}
    for L in (21, 63, 126):
        if len(closes) >= L + 1:
            log_ret = math.log(closes.iloc[-1] / closes.iloc[-L])
            tsmom[f"{L}d"] = {
                "log_return": round(log_ret, 4),
                "signal": 1 if log_ret > 0 else (-1 if log_ret < 0 else 0),
            }

    # --- Donchian channels ---
    donchian = {}
    for lookback in (20, 55):
        if len(closes) >= lookback:
            window = closes.iloc[-lookback:]
            d_high = float(window.max())
            d_low = float(window.min())
            signal = 0
            if current >= d_high:
                signal = 1
            elif current <= d_low:
                signal = -1
            donchian[f"{lookback}d"] = {
                "high": round(d_high, 2),
                "low": round(d_low, 2),
                "signal": signal,
            }

    # --- MA crossover strength: x_t = (MA_fast - MA_slow) / P_t ---
    ma_crossover = {}
    for fast, slow in ((20, 50), (50, 200)):
        if len(closes) >= slow:
            ma_fast = float(closes.tail(fast).mean())
            ma_slow = float(closes.tail(slow).mean())
            ma_crossover[f"{fast}_{slow}"] = round(
                (ma_fast - ma_slow) / current * 100, 3
            )

    # --- Realised volatility ---
    rv = {}
    log_returns = []
    for i in range(1, len(closes)):
        log_returns.append(math.log(closes.iloc[i] / closes.iloc[i - 1]))

    for window in (10, 20, 60):
        if len(log_returns) >= window:
            recent = log_returns[-window:]
            sum_sq = sum(r * r for r in recent)
            rv_val = math.sqrt(252 / window * sum_sq) * 100
            rv[f"{window}d"] = round(rv_val, 2)

    # --- Volume ratio ---
    vol_ratio = None
    if len(volumes) >= 20:
        vol_5d = float(volumes.tail(5).mean())
        vol_20d = float(volumes.mean())
        vol_ratio = round(vol_5d / vol_20d, 2) if vol_20d > 0 else None

    result = {
        "symbol": symbol,
        "current": round(current, 2),
        "high_3mo": round(high_3mo, 2),
        "low_3mo": round(low_3mo, 2),
        "change_1w_pct": change_1w,
        "change_1mo_pct": change_1mo,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "rsi_14": rsi_14,
        "fibonacci_3mo": fibs,
        "tsmom": tsmom,
        "donchian": donchian,
        "ma_crossover": ma_crossover,
        "realised_vol": rv,
        "volume_ratio_5d_20d": vol_ratio,
        "position_in_range_pct": round(
            (current - low_3mo) / fib_range * 100, 1
        ) if fib_range > 0 else None,
        "distance_from_high_pct": round(
            (high_3mo - current) / high_3mo * 100, 2
        ) if high_3mo > 0 else None,
        "fetched_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }

    return result


def _fetch_ovx():
    """Fetch OVX (CBOE Oil Volatility Index).

    2026-04-14: Routed through price_source — ^OVX is CBOE-proprietary,
    so the router uses yfinance for it (no free real-time alternative).
    """
    from portfolio.price_source import fetch_klines

    hist = fetch_klines("^OVX", interval="1d", limit=90, period="3mo")

    if hist is None or hist.empty:
        return None

    closes = hist["close"].dropna()
    if len(closes) < 2:
        return None

    current = float(closes.iloc[-1])

    # Percentile rank within 3mo
    percentile = round(
        float((closes < current).sum()) / len(closes) * 100, 1
    )

    # Regime classification
    regime = "normal"
    if current > 40:
        regime = "crisis"
    elif current > 30:
        regime = "high_vol"
    elif current < 20:
        regime = "low_vol"

    return {
        "current": round(current, 2),
        "high_3mo": round(float(closes.max()), 2),
        "low_3mo": round(float(closes.min()), 2),
        "mean_3mo": round(float(closes.mean()), 2),
        "percentile_3mo": percentile,
        "regime": regime,
        "change_1w_pct": round(
            float((current / closes.iloc[-5] - 1) * 100), 2
        ) if len(closes) >= 5 else None,
        "fetched_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }


def _fetch_etf_data(symbol):
    """Fetch ETF data (USO etc.) — investment flow proxy.

    2026-04-14: Routed via price_source — ETFs use Alpaca primary
    (real-time IEX feed) instead of yfinance's 15-min delay.
    """
    from portfolio.price_source import fetch_klines

    hist = fetch_klines(symbol, interval="1d", limit=30, period="1mo")

    if hist is None or hist.empty:
        return None

    closes = hist["close"].dropna()
    volumes = hist["volume"].dropna()

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
        "fetched_at": datetime.datetime.now(datetime.UTC).isoformat(),
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


def _fetch_cftc_cot_crude():
    """Fetch CFTC COT data for crude oil from SOCRATA Open Data API.

    Uses 'CRUDE OIL' commodity name in the legacy futures-only report.
    """
    import requests

    # Legacy Futures Only report — crude oil
    commodity = "CRUDE OIL"
    url = (
        "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
        f"?$where=commodity_name='{commodity}'"
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

    # Net/OI ratio
    nc_net_oi_ratio = round(nc_net / oi * 100, 2) if nc_net is not None and oi else None

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
        "noncomm_net_oi_ratio": nc_net_oi_ratio,
        "comm_long": comm_long,
        "comm_short": comm_short,
        "comm_net": comm_net,
        "fetched_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }

    if prev_nc_net is not None and nc_net is not None:
        result["noncomm_net_change"] = nc_net - prev_nc_net
        result["prev_report_date"] = prev.get("report_date_as_yyyy_mm_dd")

    # Compute z-score from 8 weeks of data
    nc_nets = []
    for row in rows:
        ncl = _int(row.get("noncomm_positions_long_all"))
        ncs = _int(row.get("noncomm_positions_short_all"))
        if ncl is not None and ncs is not None:
            nc_nets.append(ncl - ncs)

    if len(nc_nets) >= 4:
        mean_nc = sum(nc_nets) / len(nc_nets)
        std_nc = (sum((x - mean_nc) ** 2 for x in nc_nets) / len(nc_nets)) ** 0.5
        if std_nc > 0 and nc_net is not None:
            result["noncomm_net_zscore"] = round((nc_net - mean_nc) / std_nc, 2)

    # Disaggregated report for managed money
    try:
        disagg_url = (
            "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
            f"?$where=commodity_name='{commodity}'"
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
            "Disaggregated COT crude oil fetch failed (non-critical): %s", e
        )

    return result


def _fetch_fred_oil(api_key):
    """Fetch FRED data for oil analysis.

    Series: DGS10 (10Y Treasury), DCOILWTICO (WTI spot), DCOILBRENTEU (Brent spot).
    """
    import requests

    if not api_key:
        logger.debug("FRED API key not configured, skipping oil FRED data")
        return None

    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    result = {"fetched_at": now_iso}

    # Fetch each series
    series_ids = {
        "DGS10": "treasury_10y",
        "DCOILWTICO": "wti_spot",
        "DCOILBRENTEU": "brent_spot",
    }

    for series_id, key in series_ids.items():
        try:
            url = (
                "https://api.stlouisfed.org/fred/series/observations"
                f"?series_id={series_id}&api_key={api_key}"
                "&sort_order=desc&limit=30&file_type=json"
            )
            resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            obs = resp.json().get("observations", [])

            vals = []
            for o in obs:
                val = o.get("value", ".")
                if val != ".":
                    try:
                        vals.append({"date": o["date"], "value": float(val)})
                    except (ValueError, KeyError):
                        continue

            if vals:
                latest = vals[0]["value"]
                oldest = vals[-1]["value"] if len(vals) > 1 else latest
                direction = None
                if len(vals) >= 5:
                    recent_avg = sum(v["value"] for v in vals[:5]) / 5
                    older_vals = vals[5:10]
                    if older_vals:
                        older_avg = sum(v["value"] for v in older_vals) / len(older_vals)
                        if recent_avg > older_avg + 0.05:
                            direction = "rising"
                        elif recent_avg < older_avg - 0.05:
                            direction = "falling"
                        else:
                            direction = "stable"

                result[key] = {
                    "latest": latest,
                    "date": vals[0]["date"],
                    "1mo_ago": oldest,
                    "direction": direction,
                }
        except Exception as e:
            logger.debug("FRED %s fetch failed: %s", series_id, e)

    return result if len(result) > 1 else None


# ---------------------------------------------------------------------------
# COT history tracking
# ---------------------------------------------------------------------------

def _record_cot_history(cot_data):
    """Append COT data to oil-specific history file."""
    if not cot_data:
        return

    report_date = cot_data.get("report_date")
    if not report_date:
        return

    existing = load_jsonl(_COT_HISTORY_FILE)
    for entry in existing[-20:]:
        if entry.get("report_date") == report_date:
            return  # Already logged

    if len(existing) > 52:
        from portfolio.file_utils import prune_jsonl
        try:
            prune_jsonl(_COT_HISTORY_FILE, max_entries=52)
        except Exception as e:
            logger.debug("Oil COT history prune failed: %s", e)

    record = {
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "report_date": report_date,
        "oi": cot_data.get("open_interest"),
        "nc_net": cot_data.get("noncomm_net"),
        "nc_net_oi_ratio": cot_data.get("noncomm_net_oi_ratio"),
        "comm_net": cot_data.get("comm_net"),
        "mm_net": cot_data.get("managed_money_net"),
    }

    atomic_append_jsonl(_COT_HISTORY_FILE, record)
    logger.info("Oil COT history: appended %s", report_date)


def _compute_cot_trend():
    """Compute COT trend from historical data for crude oil."""
    entries = load_jsonl(_COT_HISTORY_FILE)
    if not entries:
        return {}

    entries.sort(key=lambda e: e.get("report_date", ""))
    recent = entries[-8:]

    result = {
        "weeks_available": len(recent),
    }

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
# Oil context builder
# ---------------------------------------------------------------------------

def _build_oil_context(market, generated_at):
    """Build data/oil_deep_context.json from fetched market data."""
    ctx = {
        "generated_at": generated_at,
        "version": 1,
    }

    # --- WTI section ---
    ctx["wti"] = market.get("cl_f") or {}

    # --- Brent section ---
    ctx["brent"] = market.get("bz_f") or {}

    # --- Brent-WTI spread ---
    cl_price = (market.get("cl_f") or {}).get("current")
    bz_price = (market.get("bz_f") or {}).get("current")
    if cl_price and bz_price:
        spread = round(bz_price - cl_price, 2)
        ctx["brent_wti_spread"] = {
            "current": spread,
            "note": (
                "Brent premium (normal)" if spread > 0
                else "WTI premium (unusual)"
            ),
        }

        # Curve slope proxy: ln(F_brent) - ln(F_wti)
        log_slope = round(math.log(bz_price) - math.log(cl_price), 4)
        ctx["curve_slope_proxy"] = {
            "log_slope": log_slope,
            "note": (
                "backwardation proxy (tight supply)" if log_slope > 0.03
                else "contango proxy (oversupply)" if log_slope < -0.01
                else "neutral"
            ),
        }

        # Carry: annualised (slope / delta_T). Assuming ~1 month between
        # front-month WTI and Brent as proxy.
        carry_annual = round(log_slope * 12, 4)
        ctx["carry_annual"] = {
            "value": carry_annual,
            "note": "Annualised carry from Brent-WTI slope proxy",
        }
    else:
        ctx["brent_wti_spread"] = {}
        ctx["curve_slope_proxy"] = {}
        ctx["carry_annual"] = {}

    # --- OVX ---
    ovx_data = market.get("ovx") or {}
    ctx["ovx"] = ovx_data

    # --- IV-RV spread ---
    ovx_val = ovx_data.get("current")
    rv_20d = (market.get("cl_f") or {}).get("realised_vol", {}).get("20d")
    if ovx_val is not None and rv_20d is not None:
        iv_rv = round(ovx_val - rv_20d, 2)
        ctx["iv_rv_spread"] = {
            "spread": iv_rv,
            "note": (
                "fear premium (IV >> RV)" if iv_rv > 5
                else "complacency (IV << RV)" if iv_rv < -5
                else "neutral"
            ),
        }
    else:
        ctx["iv_rv_spread"] = {}

    # --- Crack spread: RBOB * 42 - WTI (per barrel) ---
    rb_price = (market.get("rb_f") or {}).get("current")
    if rb_price and cl_price:
        crack = round(rb_price * 42 - cl_price, 2)
        ctx["crack_spread"] = {
            "crack_per_bbl": crack,
            "trend": (
                "strong" if crack > 25
                else "normal" if crack > 10
                else "weak"
            ),
            "note": "Widening = refinery demand pull; narrowing = demand weakness",
        }
    else:
        ctx["crack_spread"] = {}

    # --- USO ETF ---
    ctx["uso_etf"] = market.get("uso") or {}

    # --- COT positioning ---
    cot_data = market.get("cot_crude") or {}
    cot_trend = _compute_cot_trend()
    ctx["cot_positioning"] = {
        "live": cot_data,
        "trend": cot_trend,
    }

    # Crowding assessment
    nc_zscore = cot_data.get("noncomm_net_zscore")
    if nc_zscore is not None:
        crowding = "low"
        if abs(nc_zscore) > 1.5:
            crowding = "high"
        elif abs(nc_zscore) > 0.8:
            crowding = "medium"
        ctx["cot_positioning"]["crowding"] = crowding
        ctx["cot_positioning"]["crowding_note"] = (
            f"z-score {nc_zscore}: "
            + ("contrarian sell signal" if nc_zscore > 1.5
               else "contrarian buy signal" if nc_zscore < -1.5
               else "moderate — no extreme" if abs(nc_zscore) > 0.8
               else "light positioning — room to add")
        )

    # --- FRED ---
    ctx["fred"] = market.get("fred") or {}

    # --- Event calendar (hardcoded 2026 key dates) ---
    ctx["event_calendar"] = _build_event_calendar()

    # --- External / seed research ---
    ctx["external_research"] = _load_oil_seed_research()

    # --- GARCH (conditional on arch library) ---
    ctx["garch_forecast"] = _try_garch(market)

    # --- Refresh status ---
    ctx["refresh_status"] = market.get("refresh_status", {})
    if market.get("refreshed_sources"):
        ctx["last_auto_refresh"] = generated_at
        ctx["auto_refreshed_sources"] = market["refreshed_sources"]

    return ctx


# ---------------------------------------------------------------------------
# Event calendar
# ---------------------------------------------------------------------------

def _build_event_calendar():
    """Build upcoming oil-relevant event calendar."""
    today = datetime.date.today()
    events = []

    # EIA Weekly Petroleum Status Report — every Wednesday 10:30 ET
    # Find next 4 Wednesdays
    d = today
    eia_count = 0
    while eia_count < 4:
        d += datetime.timedelta(days=1)
        if d.weekday() == 2:  # Wednesday
            days_until = (d - today).days
            events.append({
                "type": "EIA_WPSR",
                "date": d.isoformat(),
                "days_until": days_until,
                "time_et": "10:30",
            })
            eia_count += 1

    # OPEC+ meetings 2026 (approximate — typically quarterly)
    opec_dates = [
        "2026-04-03", "2026-06-05", "2026-09-04", "2026-12-04",
    ]
    for ds in opec_dates:
        d = datetime.date.fromisoformat(ds)
        days_until = (d - today).days
        if days_until >= -1:
            events.append({
                "type": "OPEC",
                "date": ds,
                "days_until": days_until,
            })

    # FOMC meetings 2026
    fomc_dates = [
        "2026-01-29", "2026-03-19", "2026-05-06", "2026-06-17",
        "2026-07-29", "2026-09-17", "2026-10-28", "2026-12-09",
    ]
    for ds in fomc_dates:
        d = datetime.date.fromisoformat(ds)
        days_until = (d - today).days
        if days_until >= -1:
            events.append({
                "type": "FOMC",
                "date": ds,
                "days_until": days_until,
            })

    events.sort(key=lambda e: e.get("days_until", 999))
    return events[:10]


# ---------------------------------------------------------------------------
# GARCH(1,1) — conditional on arch library
# ---------------------------------------------------------------------------

def _try_garch(market):
    """Attempt GARCH(1,1) fit if arch library is available."""
    try:
        from arch import arch_model  # noqa: F401
    except ImportError:
        return None

    cl_data = market.get("cl_f")
    if not cl_data:
        return None

    try:
        import numpy as np

        # 2026-04-14: route via price_source (CL=F → Binance FAPI for
        # real-time, yfinance fallback if Binance unavailable).
        from portfolio.price_source import fetch_klines

        hist = fetch_klines("CL=F", interval="1d", limit=365, period="1y")
        if hist is None or hist.empty or len(hist) < 100:
            return None

        closes = hist["close"].dropna()
        returns = np.diff(np.log(closes.values)) * 100  # pct log returns

        from arch import arch_model
        model = arch_model(returns, vol="Garch", p=1, q=1, mean="Zero")
        fit = model.fit(disp="off", show_warning=False)

        forecast = fit.forecast(horizon=5)
        var_1d = forecast.variance.iloc[-1, 0]
        var_5d = forecast.variance.iloc[-1, :5].sum()

        return {
            "sigma_1d": round(float(np.sqrt(var_1d)) / 100, 4),
            "sigma_5d": round(float(np.sqrt(var_5d)) / 100, 4),
            "note": "GARCH(1,1) conditional volatility forecast",
        }
    except Exception as e:
        logger.debug("GARCH fit failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Seed research (static + updateable)
# ---------------------------------------------------------------------------

def _load_oil_seed_research():
    """Load cached external research or seed with baseline oil data."""
    existing = load_json("data/oil_external_cache.json")
    if existing and existing.get("supply_demand"):
        return existing

    return {
        "last_updated": "2026-03-19",
        "supply_demand": {
            "opec_plus_cuts_mbd": 2.2,
            "opec_compliance_pct": 85,
            "us_production_mbd": 13.2,
            "us_spr_mb": 370,
            "global_demand_mbd": 103.5,
            "global_demand_growth_note": "+1.2 mbd YoY (IEA forecast)",
            "global_supply_mbd": 102.8,
            "supply_demand_balance_note": "Modest deficit, OPEC+ managing cuts",
        },
        "key_drivers": {
            "primary": "OPEC+ production policy — controls marginal supply",
            "secondary": "US shale response to price levels — breakeven ~$50-55 WTI",
            "macro": "DXY, global growth, China re-opening trajectory",
            "geopolitical": "Middle East tensions, Russia sanctions, Venezuela",
        },
        "key_risks": [
            "OPEC+ compliance breakdown — cheating by members",
            "US shale production surge above 14 mbd",
            "Global recession reducing demand by 1-2 mbd",
            "DXY strength above 105 (oil priced in USD)",
            "Strategic reserve releases by major consumers",
            "Iran/Venezuela supply returning to market",
        ],
        "seasonal_patterns": {
            "driving_season": "May-Sep (US gasoline demand peak — bullish)",
            "refinery_maintenance": "Feb-Apr, Sep-Oct (bearish — lower crude demand)",
            "winter_heating": "Nov-Feb (heating oil demand — modestly bullish)",
            "hurricane_season": "Jun-Nov (Gulf of Mexico supply disruption risk)",
        },
        "historical_context": {
            "wti_2024_range": "$65-$87",
            "wti_covid_low_apr2020": -37.63,
            "wti_post_covid_high_2022": 123.70,
            "brent_wti_typical_spread": "$3-5",
            "note": "Apr 2020 negative prices caused by storage/logistics crisis. "
                    "Extreme curve stress can cause localized negative prices.",
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_rsi(closes, period=14):
    """Compute RSI from a pandas Series of close prices."""
    if len(closes) < period + 1:
        return None

    deltas = closes.diff().dropna()
    gains = deltas.where(deltas > 0, 0.0)
    losses = (-deltas).where(deltas < 0, 0.0)

    avg_gain = float(gains.iloc[:period].mean())
    avg_loss = float(losses.iloc[:period].mean())

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + float(gains.iloc[i])) / period
        avg_loss = (avg_loss * (period - 1) + float(losses.iloc[i])) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 1)


def _int(val):
    """Safe int conversion."""
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = load_json("config.json")
    precompute(config)
