"""Fin Fish: intraday dip/spike fishing planner for metals warrants.

Computes optimal limit buy levels for metals warrants based on:
- Recent daily range patterns (how deep do dips / how high do spikes go?)
- ATR-based volatility and first-passage-time fill probabilities
- Structural support/resistance levels (Fibonacci, pivot, smart money)
- Chronos/model drift for directional bias
- RSI-based direction selection (BULL vs BEAR fishing)
- Avanza warrant barrier safety checks

Supports both BULL (buy dips) and BEAR (buy spikes) fishing.

The output is a ranked table of fishing levels with fill probability,
expected gain on bounce, EV in SEK, and barrier distance.

Machine-readable output via ``compute_fishing_plan()`` for snipe manager
integration.  CLI one-shot via ``main()``.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import math
from pathlib import Path
from typing import Any

import requests

from portfolio.file_utils import atomic_append_jsonl, load_json
from portfolio.monte_carlo import drift_from_probability, volatility_from_atr
from portfolio.price_targets import (
    fill_probability,
    fill_probability_buy,
    structural_levels,
)

# ---------------------------------------------------------------------------
# External config — import from data.fin_fish_config with inline fallbacks
# ---------------------------------------------------------------------------
try:
    from data.fin_fish_config import (
        FISHING_BUDGET_SEK as _CFG_BUDGET,
    )
    from data.fin_fish_config import (
        FISHING_MIN_FILL_PROB as _CFG_MIN_FILL,
    )
    from data.fin_fish_config import (
        FISHING_PREFER_AVA as _CFG_PREFER_AVA,
    )
    from data.fin_fish_config import (
        FISHING_SL_CASCADE as _CFG_SL_CASCADE,
    )
    from data.fin_fish_config import (
        FISHING_TP_CASCADE as _CFG_TP_CASCADE,
    )
    from data.fin_fish_config import (
        PREFERRED_INSTRUMENTS as _CFG_PREFERRED,
    )
    from data.fin_fish_config import (  # type: ignore[import-untyped]
        WARRANT_CATALOG as _CFG_CATALOG,
    )
except Exception:
    _CFG_CATALOG = None
    _CFG_PREFERRED = None
    _CFG_BUDGET = None
    _CFG_MIN_FILL = None
    _CFG_TP_CASCADE = None
    _CFG_SL_CASCADE = None
    _CFG_PREFER_AVA = None

BASE_DIR = Path(__file__).resolve().parent.parent
SUMMARY_PATH = BASE_DIR / "data" / "agent_summary.json"
FISH_LOG_PATH = BASE_DIR / "data" / "fin_fish_log.jsonl"

logger = logging.getLogger("portfolio.fin_fish")

BINANCE_FAPI_TICKER = "https://fapi.binance.com/fapi/v1/ticker/24hr"
BINANCE_FAPI_PRICE = "https://fapi.binance.com/fapi/v1/ticker/price"
BINANCE_FAPI_KLINES = "https://fapi.binance.com/fapi/v1/klines"

UNDERLYING_SYMBOLS = {"XAG-USD": "XAGUSDT", "XAU-USD": "XAUUSDT"}

# ---------------------------------------------------------------------------
# Inline defaults — used when data.fin_fish_config is absent
# ---------------------------------------------------------------------------

# Warrant catalog — BULL and BEAR instruments
_DEFAULT_CATALOG: dict[str, dict] = {
    # --- Silver BULL ---
    "BULL_SILVER_X5_AVA_3": {
        "ob_id": "1069606",
        "api_type": "certificate",
        "underlying": "XAG-USD",
        "direction": "LONG",
        "leverage": 5.0,
        "barrier": 0,
        "parity": 1,
        "name": "BULL SILVER X5 AVA 3",
        "issuer": "AVA",
        "spread_pct": 0.5,
        "commission_sek": 0,
    },
    # --- Silver BEAR ---
    "BEAR_SILVER_X5_AVA_12": {
        "ob_id": "2286417",
        "api_type": "certificate",
        "underlying": "XAG-USD",
        "direction": "SHORT",
        "leverage": 5.0,
        "barrier": 0,
        "parity": 1,
        "name": "BEAR SILVER X5 AVA 12",
        "issuer": "AVA",
        "spread_pct": 0.5,
        "commission_sek": 0,
    },
    # --- Gold BULL ---
    "BULL_GULD_X5_AVA": {
        "ob_id": "738811",
        "api_type": "certificate",
        "underlying": "XAU-USD",
        "direction": "LONG",
        "leverage": 5.0,
        "barrier": 0,
        "parity": 1,
        "name": "BULL GULD X5 AVA",
        "issuer": "AVA",
        "spread_pct": 0.5,
        "commission_sek": 0,
    },
    # --- Gold BEAR (no viable AVA X5 — gold rally killed them all) ---
    "BEAR_GULD_X5_VON4": {
        "ob_id": "1047859",
        "api_type": "certificate",
        "underlying": "XAU-USD",
        "direction": "SHORT",
        "leverage": 5.0,
        "barrier": 0,
        "parity": 1,
        "name": "BEAR GULD X5 VON4",
        "issuer": "VON",
        "spread_pct": 2.2,
        "commission_sek": 0,
    },
    "BEAR_GULD_X2_AVA": {
        "ob_id": "738805",
        "api_type": "certificate",
        "underlying": "XAU-USD",
        "direction": "SHORT",
        "leverage": 2.0,
        "barrier": 0,
        "parity": 1,
        "name": "BEAR GULD X2 AVA",
        "issuer": "AVA",
        "spread_pct": 0.5,
        "commission_sek": 0,
    },
}

# Preferred instruments per (underlying, direction) — snipe manager picks these first
_DEFAULT_PREFERRED: dict[tuple[str, str], str] = {
    ("XAG-USD", "LONG"): "BULL_SILVER_X5_AVA_3",
    ("XAG-USD", "SHORT"): "BEAR_SILVER_X5_AVA_12",
    ("XAU-USD", "LONG"): "BULL_GULD_X5_AVA",
    ("XAU-USD", "SHORT"): "BEAR_GULD_X5_VON4",
}

_DEFAULT_BUDGET_SEK = 20_000
_DEFAULT_MIN_FILL_PROB = 0.02
_DEFAULT_TP_CASCADE: list[Any] = [
    {"underlying_pct": 1.5, "sell_pct": 40, "action": "move_stop_to_breakeven"},
    {"underlying_pct": 2.5, "sell_pct": 40, "action": "trail_stop_1pct"},
    {"underlying_pct": 4.0, "sell_pct": 20, "action": "close"},
]
_DEFAULT_SL_CASCADE: list[Any] = [
    {"underlying_pct": -1.0, "sell_pct": 50, "action": "partial_stop"},
    {"underlying_pct": -2.0, "sell_pct": 100, "action": "full_stop"},
]
_DEFAULT_PREFER_AVA = True

# Resolve config vs defaults
WARRANT_CATALOG: dict[str, dict] = _CFG_CATALOG if _CFG_CATALOG is not None else _DEFAULT_CATALOG
PREFERRED_INSTRUMENTS: dict[tuple[str, str], str] = (
    _CFG_PREFERRED if _CFG_PREFERRED is not None else _DEFAULT_PREFERRED
)
FISHING_BUDGET_SEK: float = _CFG_BUDGET if _CFG_BUDGET is not None else _DEFAULT_BUDGET_SEK
FISHING_MIN_FILL_PROB: float = _CFG_MIN_FILL if _CFG_MIN_FILL is not None else _DEFAULT_MIN_FILL_PROB
FISHING_TP_CASCADE: list[Any] = _CFG_TP_CASCADE if _CFG_TP_CASCADE is not None else _DEFAULT_TP_CASCADE
FISHING_SL_CASCADE: list[Any] = _CFG_SL_CASCADE if _CFG_SL_CASCADE is not None else _DEFAULT_SL_CASCADE
FISHING_PREFER_AVA: bool = _CFG_PREFER_AVA if _CFG_PREFER_AVA is not None else _DEFAULT_PREFER_AVA

# Avanza warrant hours (CET)
AVANZA_OPEN_H, AVANZA_OPEN_M = 8, 15
AVANZA_CLOSE_H, AVANZA_CLOSE_M = 21, 55

MIN_BARRIER_DISTANCE_PCT = 5.0
DEFAULT_BOUNCE_PCT = 2.0  # +2% underlying = take-profit target


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def fetch_live_spot() -> dict[str, dict]:
    """Fetch live spot prices and 24h stats from Binance FAPI."""
    result = {}
    for ticker, symbol in UNDERLYING_SYMBOLS.items():
        try:
            r = requests.get(f"{BINANCE_FAPI_TICKER}?symbol={symbol}", timeout=5)
            d = r.json()
            result[ticker] = {
                "price": float(d["lastPrice"]),
                "high_24h": float(d["highPrice"]),
                "low_24h": float(d["lowPrice"]),
                "change_pct": float(d["priceChangePercent"]),
                "volume_usd": float(d["quoteVolume"]),
            }
        except Exception as e:
            logger.warning("Binance %s error: %s", ticker, e)
    return result


def fetch_daily_ranges(ticker: str, days: int = 10) -> list[dict]:
    """Fetch recent daily candles for range analysis."""
    symbol = UNDERLYING_SYMBOLS.get(ticker)
    if not symbol:
        return []
    try:
        r = requests.get(
            BINANCE_FAPI_KLINES,
            params={"symbol": symbol, "interval": "1d", "limit": days},
            timeout=10,
        )
        candles = r.json()
        result = []
        for c in candles:
            high = float(c[2])
            low = float(c[3])
            close = float(c[4])
            result.append({
                "date": datetime.datetime.fromtimestamp(c[0] / 1000).strftime("%m-%d"),
                "high": high,
                "low": low,
                "close": close,
                "range_pct": round((high - low) / low * 100, 2) if low > 0 else 0,
            })
        return result
    except Exception as e:
        logger.warning("Daily candles error for %s: %s", ticker, e)
        return []


def session_hours_remaining() -> float:
    """Compute hours remaining in the Avanza warrant session (CET)."""
    try:
        import zoneinfo
        cet = zoneinfo.ZoneInfo("Europe/Stockholm")
    except Exception:
        import dateutil.tz  # type: ignore[import-untyped]
        cet = dateutil.tz.gettz("Europe/Stockholm")

    now = datetime.datetime.now(cet)
    close = now.replace(hour=AVANZA_CLOSE_H, minute=AVANZA_CLOSE_M, second=0, microsecond=0)
    open_time = now.replace(hour=AVANZA_OPEN_H, minute=AVANZA_OPEN_M, second=0, microsecond=0)

    if now < open_time:
        return 0.0  # before market open
    if now >= close:
        return 0.0  # after market close

    remaining = (close - now).total_seconds() / 3600
    return max(0.0, round(remaining, 2))


def load_signal_data(ticker: str) -> dict:
    """Load signal data for a ticker from agent_summary."""
    summary = load_json(SUMMARY_PATH) or {}
    signals = summary.get("signals", {})
    entry = signals.get(ticker, {})
    focus = (summary.get("focus_probabilities") or {}).get(ticker, {})
    extra = entry.get("extra") or {}
    return {
        "entry": entry,
        "focus": focus,
        "price_usd": _safe_float(entry.get("price_usd")),
        "rsi": _safe_float(entry.get("rsi")),
        "atr_pct": _safe_float(entry.get("atr_pct"), 0.5),
        "regime": str(entry.get("regime") or ""),
        "action": str(entry.get("action") or "HOLD"),
        "weighted_confidence": _safe_float(entry.get("weighted_confidence")),
        "fear_greed": extra.get("fear_greed"),
        "econ_action": str(extra.get("econ_calendar_action") or "HOLD"),
        "news_action": str(extra.get("news_event_action") or "HOLD"),
    }


def fetch_fx_rate() -> float:
    """Fetch USD/SEK exchange rate."""
    try:
        from portfolio.fx_rates import fetch_usd_sek
        return fetch_usd_sek()
    except Exception:
        return 10.0  # fallback


def _get_chronos_drift(signal: dict) -> float | None:
    """Extract Chronos 24h drift from signal data, or None if unavailable."""
    extra = (signal["entry"].get("extra") or {}) if signal.get("entry") else None
    if not extra:
        return None
    forecast_ind = extra.get("forecast_indicators") or {}
    chronos_24h = forecast_ind.get("chronos_24h_pct")
    if chronos_24h is None:
        return None
    return float(chronos_24h)


def _compute_vol_and_drift(
    signal: dict,
    daily_ranges: list[dict],
    direction: str,
) -> tuple[float, float]:
    """Compute annualized volatility and drift for GBM.

    Parameters
    ----------
    direction : str
        ``"LONG"`` biases drift downward (we want dips), ``"SHORT"`` biases
        drift upward (we want spikes).
    """
    atr_pct = signal["atr_pct"]
    p_up = _safe_float((signal["focus"].get("3h") or {}).get("probability"), 0.5)

    # Hourly ATR path — volatility_from_atr assumes hourly candles
    vol = volatility_from_atr(atr_pct)

    # Daily range path — annualize daily sigma directly with sqrt(252)
    # (volatility_from_atr uses sqrt(252/14) which is wrong for daily data)
    if daily_ranges and len(daily_ranges) >= 3:
        recent_ranges = [c["range_pct"] for c in daily_ranges[-5:] if c["range_pct"] > 0.5]
        if recent_ranges:
            avg_range = sum(recent_ranges) / len(recent_ranges)
            daily_sigma = avg_range / 1.5 / 100.0
            vol_from_daily = daily_sigma * math.sqrt(252.0)
            vol = max(vol, vol_from_daily)

    # For LONG fishing we want P(dip) — use p_up as-is (lower p_up = more likely to dip)
    # For SHORT fishing we want P(spike) — invert: use 1-p_up
    if direction == "SHORT":
        drift = drift_from_probability(1.0 - p_up, vol)
    else:
        drift = drift_from_probability(p_up, vol)

    # Blend Chronos drift if available
    chronos_pct = _get_chronos_drift(signal)
    if chronos_pct is not None:
        chronos_annual = (chronos_pct / 100.0) * 252
        drift = 0.7 * drift + 0.3 * chronos_annual

    return vol, drift


# ---------------------------------------------------------------------------
# Direction selection
# ---------------------------------------------------------------------------

def choose_fishing_directions(signal: dict) -> list[dict]:
    """Decide whether to fish BULL, BEAR, or both.

    Uses RSI, Chronos, consensus signal, Fear & Greed, econ calendar,
    and news severity to set conviction scores for each direction.

    Returns a list of dicts:
        [{"direction": "LONG", "conviction": 0.8}, ...]
    """
    rsi = signal["rsi"]
    chronos_pct = _get_chronos_drift(signal)

    directions: list[dict] = []

    # --- Base conviction from RSI ---
    if rsi < 45:
        bull_conv = 0.8 if rsi < 30 else 0.65
        bear_conv = 0.0
    elif rsi > 65:
        bear_conv = 0.8 if rsi > 70 else 0.65
        bull_conv = 0.0
    else:
        bull_conv = 0.4
        bear_conv = 0.4

    # --- Chronos 24h forecast ---
    if chronos_pct is not None:
        if chronos_pct < -0.3:
            bear_conv = max(bear_conv, 0.3) + 0.15
            bull_conv = max(0.0, bull_conv - 0.1)
        elif chronos_pct > 0.3:
            bull_conv = max(bull_conv, 0.3) + 0.15
            bear_conv = max(0.0, bear_conv - 0.1)

    # --- Consensus signal (30-signal weighted vote) ---
    action = signal.get("action", "HOLD")
    confidence = signal.get("weighted_confidence", 0)
    if action == "BUY" and confidence > 0.6:
        bull_conv += 0.15
    elif action == "SELL" and confidence > 0.6:
        bear_conv += 0.15

    # --- Fear & Greed (contrarian) ---
    fg = signal.get("fear_greed")
    if fg is not None:
        fg_val = _safe_float(fg)
        if fg_val <= 20:
            bull_conv += 0.15   # extreme fear → buy dips
        elif fg_val >= 80:
            bear_conv += 0.15   # extreme greed → sell peaks

    # --- Econ calendar (FOMC/CPI imminent → risk-off) ---
    econ_action = signal.get("econ_action", "HOLD")
    if econ_action == "SELL":
        bear_conv += 0.10
        bull_conv = max(0.0, bull_conv - 0.10)

    # --- News severity ---
    news_action = signal.get("news_action", "HOLD")
    if news_action == "SELL":
        bear_conv += 0.10
    elif news_action == "BUY":
        bull_conv += 0.10

    # Clamp to [0, 1]
    bull_conv = min(1.0, max(0.0, bull_conv))
    bear_conv = min(1.0, max(0.0, bear_conv))

    if bull_conv > 0.05:
        directions.append({"direction": "LONG", "conviction": round(bull_conv, 2)})
    if bear_conv > 0.05:
        directions.append({"direction": "SHORT", "conviction": round(bear_conv, 2)})

    return directions


# ---------------------------------------------------------------------------
# Core fishing level computation — BULL (dip) and BEAR (spike)
# ---------------------------------------------------------------------------

def compute_fishing_levels_bull(
    ticker: str,
    spot: float,
    signal: dict,
    hours: float,
    daily_ranges: list[dict],
) -> list[dict]:
    """Compute candidate BULL fishing (dip-buy) levels with fill probabilities.

    Looks for levels BELOW current price where price might dip to.
    """
    vol, drift = _compute_vol_and_drift(signal, daily_ranges, direction="LONG")
    extra = (signal["entry"].get("extra") or {}) if signal["entry"] else None

    candidates: dict[float, str] = {}

    # 1. ATR-based offsets below spot
    atr_pct = signal["atr_pct"]
    for n, label in [(0.5, "ATR 0.5x"), (1.0, "ATR 1x"), (1.5, "ATR 1.5x"),
                     (2.0, "ATR 2x"), (3.0, "ATR 3x")]:
        level = spot * (1 - n * atr_pct / 100)
        candidates[round(level, 4)] = label

    # 2. Fixed percentage offsets below spot
    for pct in [1, 2, 3, 5, 7, 10]:
        level = spot * (1 - pct / 100)
        candidates[round(level, 4)] = f"-{pct}%"

    # 3. Recent daily lows
    for candle in daily_ranges[-5:]:
        low = candle["low"]
        if 0 < low < spot:
            candidates[round(low, 4)] = f"Daily low {candle['date']}"

    # 4. Structural levels below current price
    levels = structural_levels(spot, signal["entry"], extra)
    for name, lvl in levels.items():
        if 0 < lvl < spot * 0.99:
            candidates[round(lvl, 4)] = f"Struct: {name}"

    return _score_candidates_buy(candidates, spot, vol, drift, hours)


def compute_fishing_levels_bear(
    ticker: str,
    spot: float,
    signal: dict,
    hours: float,
    daily_ranges: list[dict],
) -> list[dict]:
    """Compute candidate BEAR fishing (spike-buy) levels with fill probabilities.

    Looks for levels ABOVE current price where price might spike to —
    we would buy a BEAR cert at that elevated price.
    """
    vol, drift = _compute_vol_and_drift(signal, daily_ranges, direction="SHORT")
    extra = (signal["entry"].get("extra") or {}) if signal["entry"] else None

    candidates: dict[float, str] = {}

    # 1. ATR-based offsets above spot
    atr_pct = signal["atr_pct"]
    for n, label in [(0.5, "ATR 0.5x"), (1.0, "ATR 1x"), (1.5, "ATR 1.5x"),
                     (2.0, "ATR 2x"), (3.0, "ATR 3x")]:
        level = spot * (1 + n * atr_pct / 100)
        candidates[round(level, 4)] = label

    # 2. Fixed percentage offsets above spot
    for pct in [1, 2, 3, 5, 7, 10]:
        level = spot * (1 + pct / 100)
        candidates[round(level, 4)] = f"+{pct}%"

    # 3. Recent daily highs
    for candle in daily_ranges[-5:]:
        high = candle["high"]
        if high > spot:
            candidates[round(high, 4)] = f"Daily high {candle['date']}"

    # 4. Structural levels above current price (resistance)
    levels = structural_levels(spot, signal["entry"], extra)
    for name, lvl in levels.items():
        if lvl > spot * 1.01:
            candidates[round(lvl, 4)] = f"Struct: {name}"

    return _score_candidates_sell(candidates, spot, vol, drift, hours)


def _score_candidates_buy(
    candidates: dict[float, str],
    spot: float,
    vol: float,
    drift: float,
    hours: float,
) -> list[dict]:
    """Score BULL fishing candidates (below spot) with fill probability and EV."""
    results = []
    for level, source in sorted(candidates.items(), reverse=True):
        if level <= 0 or level >= spot:
            continue
        dip_pct = round((spot - level) / spot * 100, 2)
        if dip_pct < 0.3:
            continue

        fp = fill_probability_buy(spot, level, vol, drift, hours, is_24h=True)
        bounce_pct = dip_pct  # symmetric bounce back to current
        modest_bounce_pct = round(DEFAULT_BOUNCE_PCT, 2)
        modest_bounce_target = round(level * (1 + DEFAULT_BOUNCE_PCT / 100), 4)

        results.append({
            "level": level,
            "source": source,
            "dip_pct": dip_pct,
            "move_pct": dip_pct,  # normalized: how far from spot
            "fill_prob": round(fp, 4),
            "bounce_to_spot_pct": round(bounce_pct, 2),
            "modest_bounce_pct": modest_bounce_pct,
            "modest_bounce_target": modest_bounce_target,
        })

    return _dedupe_and_rank(results, key_field="level")


def _score_candidates_sell(
    candidates: dict[float, str],
    spot: float,
    vol: float,
    drift: float,
    hours: float,
) -> list[dict]:
    """Score BEAR fishing candidates (above spot) with fill probability and EV."""
    results = []
    for level, source in sorted(candidates.items()):
        if level <= spot:
            continue
        spike_pct = round((level - spot) / spot * 100, 2)
        if spike_pct < 0.3:
            continue

        # fill_probability gives P(running max >= target) — exactly what we need
        fp = fill_probability(spot, level, vol, drift, hours, is_24h=True)
        bounce_pct = spike_pct  # symmetric drop back to current
        modest_bounce_pct = round(DEFAULT_BOUNCE_PCT, 2)
        modest_bounce_target = round(level * (1 - DEFAULT_BOUNCE_PCT / 100), 4)

        results.append({
            "level": level,
            "source": source,
            "dip_pct": spike_pct,   # legacy name — represents distance from spot
            "move_pct": spike_pct,
            "fill_prob": round(fp, 4),
            "bounce_to_spot_pct": round(bounce_pct, 2),
            "modest_bounce_pct": modest_bounce_pct,
            "modest_bounce_target": modest_bounce_target,
        })

    return _dedupe_and_rank(results, key_field="level")


def _dedupe_and_rank(results: list[dict], key_field: str = "level") -> list[dict]:
    """Deduplicate levels within 0.2% of each other and rank by EV."""
    sorted_results = sorted(results, key=lambda x: x[key_field], reverse=True)
    deduped: list[dict] = []
    for r in sorted_results:
        if not deduped or abs(r[key_field] - deduped[-1][key_field]) / r[key_field] > 0.002:
            deduped.append(r)

    for r in deduped:
        r["ev_score"] = round(r["fill_prob"] * r["bounce_to_spot_pct"], 4)

    deduped.sort(key=lambda x: x["ev_score"], reverse=True)
    return deduped


# Legacy alias — keep backward compatibility
def compute_fishing_levels(
    ticker: str,
    spot: float,
    signal: dict,
    hours: float,
    daily_ranges: list[dict],
) -> list[dict]:
    """Compute candidate fishing (dip-buy) levels with fill probabilities.

    Legacy wrapper — delegates to ``compute_fishing_levels_bull``.
    """
    return compute_fishing_levels_bull(ticker, spot, signal, hours, daily_ranges)


# ---------------------------------------------------------------------------
# Warrant evaluation
# ---------------------------------------------------------------------------

def _select_warrants(
    ticker: str,
    direction: str,
    spot: float,
) -> list[dict]:
    """Select matching warrants for (ticker, direction), preferring the preferred
    instrument and falling back to catalog search.

    Parameters
    ----------
    direction : str
        ``"LONG"`` for BULL, ``"SHORT"`` for BEAR.
    """
    pref_key = (ticker, direction)
    preferred_id = PREFERRED_INSTRUMENTS.get(pref_key)

    # Collect all matching warrants
    matching = [
        w for w in WARRANT_CATALOG.values()
        if w["underlying"] == ticker and w["direction"] == direction
    ]

    if not matching:
        return []

    # If we have a preferred instrument and it exists, put it first
    if preferred_id and preferred_id in WARRANT_CATALOG:
        pref_warrant = WARRANT_CATALOG[preferred_id]
        if pref_warrant in matching:
            matching.remove(pref_warrant)
            matching.insert(0, pref_warrant)

    # If FISHING_PREFER_AVA, sort AVA warrants before others (after preferred)
    if FISHING_PREFER_AVA:
        ava_first = []
        others = []
        for w in matching:
            if w.get("issuer") == "AVA":
                ava_first.append(w)
            else:
                others.append(w)
        # Preferred is already first if it's AVA; otherwise keep order
        matching = ava_first + others

    return matching


def evaluate_warrants(
    ticker: str,
    spot: float,
    fishing_levels: list[dict],
    budget_sek: float,
    fx_rate: float,
    direction: str = "LONG",
) -> list[dict]:
    """Match fishing levels to available warrants, compute sizing and EV.

    Parameters
    ----------
    direction : str
        ``"LONG"`` for BULL certs, ``"SHORT"`` for BEAR certs.
    """
    matching = _select_warrants(ticker, direction, spot)

    results = []
    for warrant in matching:
        barrier = warrant["barrier"]
        name = warrant["name"]
        is_daily_cert = warrant["api_type"] == "certificate" and barrier == 0

        # Dynamic leverage: compute from spot and barrier for warrants.
        # Config leverage is stale (set when cert was added, not at current price).
        # Daily certs (no barrier) keep config leverage.
        if not is_daily_cert and barrier > 0:
            dist = abs(spot - barrier)
            leverage = spot / dist if dist > 0 else warrant["leverage"]
        else:
            leverage = warrant["leverage"]

        # Barrier checks only for MINI warrants (barrier > 0)
        if not is_daily_cert and barrier > 0:
            if direction == "LONG" and spot <= barrier:
                continue  # knocked out
            if direction == "SHORT" and spot >= barrier:
                # BEAR MINIs get knocked out if underlying goes above barrier
                # (depends on product, but skip if too close)
                pass
            barrier_distance = abs(spot - barrier) / spot * 100
            if barrier_distance < MIN_BARRIER_DISTANCE_PCT:
                continue
        else:
            barrier_distance = 100.0

        for fl in fishing_levels:
            level = fl["level"]

            # Check barrier safety at fishing level (MINI warrants only)
            if not is_daily_cert and barrier > 0:
                if direction == "LONG":
                    fish_barrier_dist = round((level - barrier) / level * 100, 2)
                else:
                    fish_barrier_dist = round(abs(level - barrier) / level * 100, 2)
                if fish_barrier_dist < MIN_BARRIER_DISTANCE_PCT:
                    continue
            else:
                fish_barrier_dist = 100.0

            # Estimate warrant price at fishing level
            parity = warrant.get("parity", 10)
            spread_pct = warrant.get("spread_pct", 1.0)
            commission = warrant.get("commission_sek", 0)
            issuer = warrant.get("issuer", "?")

            if is_daily_cert:
                # Daily leverage cert: we cannot compute price from underlying.
                # Use budget / leverage for sizing.
                # Gain = underlying_move_pct * leverage - spread_pct
                warrant_price_at_fish = None  # unknown without Avanza quote
            else:
                if direction == "LONG":
                    warrant_price_at_fish = max(0.01, (level - barrier) / parity * fx_rate)
                else:
                    warrant_price_at_fish = max(0.01, (barrier - level) / parity * fx_rate)

            # Underlying move that generates profit
            bounce_underlying_pct = fl["bounce_to_spot_pct"]

            if is_daily_cert:
                # gain = underlying_move% * leverage - spread%
                warrant_gain_pct = round(bounce_underlying_pct * leverage, 2)
                net_gain_pct = round(warrant_gain_pct - spread_pct, 2)

                if net_gain_pct <= 0:
                    continue

                # For sizing daily certs, estimate cost from budget
                # We don't know the exact cert price, so compute EV as % of budget
                invest_sek = budget_sek
                gain_sek = round(budget_sek * net_gain_pct / 100, 2)
                spread_cost_sek = round(budget_sek * spread_pct / 100, 2)
                qty = 0  # unknown without live cert price
                display_price = 0.0
            else:
                warrant_gain_pct = round(bounce_underlying_pct * leverage, 2)
                net_gain_pct = round(warrant_gain_pct - spread_pct, 2)

                if net_gain_pct <= 0:
                    continue

                assert warrant_price_at_fish is not None  # MINI warrants always have a price
                qty = max(1, int(budget_sek / warrant_price_at_fish))
                invest_sek = round(qty * warrant_price_at_fish, 0)
                gross_sek = round(qty * warrant_price_at_fish * warrant_gain_pct / 100, 2)
                spread_cost_sek = round(qty * warrant_price_at_fish * spread_pct / 100, 2)
                gain_sek = round(gross_sek - spread_cost_sek - commission, 2)
                display_price = round(warrant_price_at_fish, 2)

            ev_sek = round(fl["fill_prob"] * gain_sek, 2)

            results.append({
                "level": level,
                "source": fl["source"],
                "dip_pct": fl["dip_pct"],
                "move_pct": fl.get("move_pct", fl["dip_pct"]),
                "fill_prob": fl["fill_prob"],
                "warrant": name,
                "ob_id": warrant["ob_id"],
                "issuer": issuer,
                "leverage": leverage,
                "barrier": barrier,
                "barrier_dist_pct": fish_barrier_dist,
                "warrant_price": display_price,
                "qty": qty,
                "invest_sek": invest_sek,
                "bounce_pct": bounce_underlying_pct,
                "warrant_gain_pct": warrant_gain_pct,
                "spread_pct": spread_pct,
                "spread_cost_sek": spread_cost_sek,
                "net_gain_pct": net_gain_pct,
                "gain_sek": gain_sek,
                "ev_sek": ev_sek,
                "direction": direction,
                "is_daily_cert": is_daily_cert,
            })

    results.sort(key=lambda x: x["ev_sek"], reverse=True)

    # Deduplicate: keep only the best warrant per price level (within 0.2%).
    # Without this, the same level appears N times for N different warrants,
    # drowning out other price levels in the output.
    deduped: list[dict] = []
    for r in results:
        is_dup = False
        for kept in deduped:
            if abs(r["level"] - kept["level"]) / max(r["level"], 1e-9) < 0.002:
                is_dup = True
                break
        if not is_dup:
            deduped.append(r)

    return deduped


# ---------------------------------------------------------------------------
# Structured plan output (for snipe manager)
# ---------------------------------------------------------------------------

def _build_instrument_info(warrant_results: list[dict], direction: str) -> dict:
    """Extract instrument metadata from the best warrant result."""
    if not warrant_results:
        return {}
    best_name = warrant_results[0]["warrant"]
    for w in WARRANT_CATALOG.values():
        if w["name"] == best_name:
            return {
                "ob_id": w["ob_id"],
                "name": w["name"],
                "leverage": w["leverage"],
                "barrier": w["barrier"],
                "issuer": w.get("issuer", "?"),
                "spread_pct": w.get("spread_pct", 1.0),
                "api_type": w.get("api_type", "warrant"),
                "direction": direction,
            }
    return {}


def compute_fishing_plan(
    ticker: str,
    spot: float,
    signal: dict,
    hours: float,
    daily_ranges: list[dict],
    budget_sek: float | None = None,
    fx_rate: float | None = None,
) -> list[dict]:
    """Compute a structured fishing plan for a ticker.

    Returns a list of plans (one per direction: LONG and/or SHORT) with
    machine-readable structure for the snipe manager.

    Each plan dict::

        {
            "ticker": "XAG-USD",
            "spot": 69.21,
            "direction": "LONG",      # or "SHORT"
            "conviction": 0.65,
            "levels": [...],           # fishing level dicts with fill_prob, ev_sek
            "instrument": {"ob_id": ..., "name": ..., "leverage": ...},
            "tp_cascade": [...],       # from FISHING_TP_CASCADE config
            "sl_cascade": [...],       # from FISHING_SL_CASCADE config
        }
    """
    if budget_sek is None:
        budget_sek = FISHING_BUDGET_SEK
    if fx_rate is None:
        fx_rate = fetch_fx_rate()

    directions = choose_fishing_directions(signal)
    plans: list[dict] = []

    for d in directions:
        direction = d["direction"]
        conviction = d["conviction"]

        # Compute fishing levels
        if direction == "LONG":
            levels = compute_fishing_levels_bull(ticker, spot, signal, hours, daily_ranges)
        else:
            levels = compute_fishing_levels_bear(ticker, spot, signal, hours, daily_ranges)

        # Evaluate warrants for these levels
        warrant_results = evaluate_warrants(
            ticker, spot, levels, budget_sek, fx_rate, direction=direction,
        )

        # Filter by minimum fill probability
        warrant_results = [
            r for r in warrant_results if r["fill_prob"] >= FISHING_MIN_FILL_PROB
        ]

        if not warrant_results:
            continue

        instrument_info = _build_instrument_info(warrant_results, direction)

        plans.append({
            "ticker": ticker,
            "spot": spot,
            "direction": direction,
            "conviction": conviction,
            "levels": warrant_results,
            "instrument": instrument_info,
            "tp_cascade": list(FISHING_TP_CASCADE),
            "sl_cascade": list(FISHING_SL_CASCADE),
        })

    return plans


# ---------------------------------------------------------------------------
# Telegram summary
# ---------------------------------------------------------------------------

def format_telegram_plan(plans: list[dict], avanza_online: bool = False) -> str:
    """Format fishing plans into a concise Telegram message.

    Example output::

        FISH PLAN XAG-USD $69.21
        BULL X5: fish $67.50 (-2.5%) fill 11% EV 31
        BEAR X5: fish $71.00 (+2.6%) fill 8% EV 22
        Avanza: OFFLINE
    """
    if not plans:
        return "FISH: no viable plans"

    lines: list[str] = []
    ticker = plans[0]["ticker"]
    spot = plans[0]["spot"]
    lines.append(f"FISH PLAN {ticker} ${spot:.2f}")

    for plan in plans:
        direction = plan["direction"]
        label = "BULL" if direction == "LONG" else "BEAR"
        leverage = plan["instrument"].get("leverage", 1)
        best_levels = plan["levels"][:1]  # top level only for Telegram

        for lvl in best_levels:
            level_price = lvl["level"]
            if direction == "LONG":
                move_str = f"-{lvl['move_pct']:.1f}%"
            else:
                move_str = f"+{lvl['move_pct']:.1f}%"
            fill_pct = lvl["fill_prob"] * 100
            ev = lvl["ev_sek"]
            lines.append(
                f"{label} X{leverage:.0f}: fish ${level_price:.2f} ({move_str}) "
                f"fill {fill_pct:.0f}% EV {ev:.0f}"
            )

    avanza_status = "online" if avanza_online else "OFFLINE"
    lines.append(f"Avanza: {avanza_status}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI report formatting
# ---------------------------------------------------------------------------

def format_report(
    ticker: str,
    spot_data: dict,
    signal: dict,
    daily_ranges: list[dict],
    plans: list[dict],
    hours: float,
    max_levels: int = 8,
) -> str:
    """Format the fishing plan as a readable CLI report."""
    spot = spot_data["price"]
    lines: list[str] = []
    lines.append(f"=== {ticker} -- ${spot:.2f} ===")
    lines.append(f"  24h: ${spot_data['low_24h']:.2f} - ${spot_data['high_24h']:.2f} "
                 f"({spot_data['change_pct']:+.2f}%) | Vol ${spot_data['volume_usd']/1e6:.0f}M")
    lines.append(f"  Regime: {signal['regime']} | RSI: {signal['rsi']:.1f} | "
                 f"ATR: {signal['atr_pct']:.2f}% | Signal: {signal['action']}")
    # Signal boosters line
    fg = signal.get("fear_greed")
    fg_str = f"F&G: {fg}" if fg is not None else "F&G: n/a"
    conf_str = f"Consensus: {signal['action']} {signal['weighted_confidence']:.0%}"
    econ_str = f"Econ: {signal.get('econ_action', 'HOLD')}"
    news_str = f"News: {signal.get('news_action', 'HOLD')}"
    lines.append(f"  {fg_str} | {conf_str} | {news_str} | {econ_str}")
    lines.append(f"  Session hours left: {hours:.1f}h")

    # Daily range pattern
    if daily_ranges:
        recent = daily_ranges[-5:]
        lines.append(f"  Daily ranges (last {len(recent)}d):")
        for c in recent:
            lines.append(f"    {c['date']}: ${c['low']:.2f}-${c['high']:.2f} ({c['range_pct']:.1f}%)")
        avg_range = sum(c["range_pct"] for c in recent) / len(recent)
        avg_low_dip = sum((c["high"] - c["low"]) / c["high"] * 100 for c in recent) / len(recent)
        lines.append(f"  Avg daily range: {avg_range:.1f}% | Avg dip from high: {avg_low_dip:.1f}%")

    # Direction analysis
    directions = choose_fishing_directions(signal)
    chronos_pct = _get_chronos_drift(signal)
    dir_labels = ", ".join(
        f"{'BULL' if d['direction'] == 'LONG' else 'BEAR'} ({d['conviction']:.0%})"
        for d in directions
    )
    lines.append(f"  Direction: {dir_labels}")
    if chronos_pct is not None:
        lines.append(f"  Chronos 24h: {chronos_pct:+.2f}%")
    lines.append("")

    if not plans:
        lines.append("  No viable fishing plans found.")
        return "\n".join(lines)

    for plan in plans:
        direction = plan["direction"]
        label = "BULL" if direction == "LONG" else "BEAR"
        conviction = plan["conviction"]
        instrument = plan.get("instrument", {})
        inst_name = instrument.get("name", "?")
        inst_lev = instrument.get("leverage", 1)
        warrant_results = plan["levels"]

        lines.append(f"  --- {label} fishing (conviction {conviction:.0%}) "
                     f"via {inst_name} (X{inst_lev:.1f}) ---")

        if not warrant_results:
            lines.append("    No viable levels.")
            lines.append("")
            continue

        # Table header
        if direction == "LONG":
            lines.append(f"    {'Level':>9} {'Dip%':>6} {'Fill%':>6} {'Gross':>6} {'Sprd':>5} "
                         f"{'Net%':>5} {'EV/SEK':>7} {'Barr%':>6} {'Source':<20}")
        else:
            lines.append(f"    {'Level':>9} {'Spike%':>6} {'Fill%':>6} {'Gross':>6} {'Sprd':>5} "
                         f"{'Net%':>5} {'EV/SEK':>7} {'Barr%':>6} {'Source':<20}")
        lines.append(f"    {'-'*9} {'-'*6} {'-'*6} {'-'*6} {'-'*5} "
                     f"{'-'*5} {'-'*7} {'-'*6} {'-'*20}")

        shown = 0
        for r in warrant_results:
            if shown >= max_levels:
                break
            if r["fill_prob"] < FISHING_MIN_FILL_PROB:
                continue

            lines.append(
                f"    ${r['level']:>8.2f} {r['move_pct']:>5.1f}% {r['fill_prob']:>5.1%} "
                f"{r['warrant_gain_pct']:>5.1f}% {r['spread_pct']:>4.1f}% "
                f"{r['net_gain_pct']:>4.1f}% {r['ev_sek']:>6.0f} {r['barrier_dist_pct']:>5.1f}% "
                f"{r['source']:>20s}"
            )
            shown += 1

        if shown == 0:
            lines.append(f"    All levels have <{FISHING_MIN_FILL_PROB:.0%} fill probability.")

        # TP/SL cascade (handles both dict-style and float-style entries)
        tp_str = ", ".join(
            f"+{t['underlying_pct']:.1f}%" if isinstance(t, dict) else f"+{t:.1f}%"
            for t in plan["tp_cascade"]
        )
        sl_str = ", ".join(
            f"{s['underlying_pct']:.1f}%" if isinstance(s, dict) else f"-{s:.1f}%"
            for s in plan["sl_cascade"]
        )
        lines.append(f"    TP cascade (underlying): {tp_str}")
        lines.append(f"    SL cascade (underlying): {sl_str}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fin Fish: compute optimal dip/spike fishing levels for metals warrants."
    )
    parser.add_argument("--hours", type=float, default=0,
                        help="Override planning horizon (default: auto-compute from session).")
    parser.add_argument("--budget", type=float, default=FISHING_BUDGET_SEK,
                        help=f"Budget per fishing level in SEK (default: {FISHING_BUDGET_SEK}).")
    parser.add_argument("--metals", default="silver",
                        help="Comma-separated: silver,gold (default: silver).")
    parser.add_argument("--max-levels", type=int, default=8,
                        help="Max fishing levels to display per metal (default: 8).")
    parser.add_argument("--direction", choices=["bull", "bear", "auto"], default="auto",
                        help="Force direction: bull, bear, or auto (default: auto).")
    parser.add_argument("--telegram", action="store_true",
                        help="Print Telegram-format summary instead of full report.")
    parser.add_argument("--monitor", action="store_true",
                        help="Enter smart monitoring mode after analysis.")
    parser.add_argument("--entry-price", type=float, default=0,
                        help="Entry price for monitoring (default: current spot).")
    parser.add_argument("--cert-price", type=float, default=0,
                        help="Certificate entry price in SEK for P&L tracking.")
    parser.add_argument("--cert-units", type=int, default=0,
                        help="Number of certificate units held.")
    parser.add_argument("--leverage", type=float, default=5.0,
                        help="Certificate leverage (default: 5x).")
    args = parser.parse_args()

    metals = [m.strip().lower() for m in args.metals.split(",")]
    ticker_map = {"silver": "XAG-USD", "gold": "XAU-USD"}
    tickers = [ticker_map[m] for m in metals if m in ticker_map]

    if not tickers:
        print("No valid metals specified. Use: silver, gold")
        return 1

    # Session hours
    hours = args.hours if args.hours > 0 else session_hours_remaining()
    if hours <= 0:
        # Outside trading hours — use next session (planning mode)
        hours = 13.67  # full session 08:15-21:55
        print("Outside Avanza hours -- showing NEXT SESSION plan (13.7h horizon)\n")
    else:
        print(f"Session: {hours:.1f}h remaining until 21:55 CET\n")

    # Fetch data
    print("Fetching live prices...")
    spot_data = fetch_live_spot()
    fx_rate = fetch_fx_rate()
    print(f"FX rate: {fx_rate:.2f} SEK/USD\n")

    # --- Preflight GO/NO-GO check ---
    print("Running preflight check...")
    preflight_results = {}
    try:
        from scripts.fish_preflight import compute_preflight, print_preflight
        for ticker in tickers:
            pf = compute_preflight(ticker)
            preflight_results[ticker] = pf
            print_preflight(pf)
    except Exception as e:
        print(f"  Preflight unavailable: {e}")

    # --- Instrument profile briefing ---
    try:
        from portfolio.instrument_profile import (
            format_profile_briefing,
            get_profile,
        )
        signal_data = load_json(BASE_DIR / "data" / "agent_summary_compact.json")
        for ticker in tickers:
            profile = get_profile(ticker)
            if profile:
                print(f"\n{'='*60}")
                print(f"  INSTRUMENT PROFILE: {profile['name']}")
                print(f"{'='*60}")
                print(format_profile_briefing(ticker, signal_data))

                # Show signal reliability ranking for this ticker
                reliability = (signal_data or {}).get("signal_reliability", {}).get(ticker, {})
                if reliability:
                    ranked = sorted(
                        [(k, v) for k, v in reliability.items() if isinstance(v, dict) and v.get("total", 0) >= 30],
                        key=lambda x: x[1].get("accuracy", 0),
                        reverse=True,
                    )
                    if ranked:
                        print("\n  Signal reliability (top 10 / bottom 3):")
                        for name, data in ranked[:10]:
                            acc = data.get("accuracy", 0)
                            n = data.get("total", 0)
                            marker = " *" if name in profile.get("trusted_signals", []) else ""
                            print(f"    {name:20s} {acc:5.1%} ({n:4d} samples){marker}")
                        if len(ranked) > 10:
                            print("    ...")
                            for name, data in ranked[-3:]:
                                acc = data.get("accuracy", 0)
                                n = data.get("total", 0)
                                marker = " X" if name in profile.get("ignored_signals", []) else ""
                                print(f"    {name:20s} {acc:5.1%} ({n:4d} samples){marker}")

                # Show deep context summary if available (with staleness check)
                precompute_path = BASE_DIR / profile.get("precompute_file", "")
                deep_ctx = load_json(precompute_path)
                if deep_ctx:
                    import datetime as _dt
                    _gen = deep_ctx.get("generated_at", "")
                    if _gen:
                        try:
                            _age_s = (_dt.datetime.now(_dt.timezone.utc) - _dt.datetime.fromisoformat(_gen)).total_seconds()
                            if _age_s > 7200:  # 2 hours
                                print(f"  ⚠ Deep context STALE ({_age_s/3600:.1f}h old)")
                        except Exception:
                            pass
                    analyst = deep_ctx.get("analyst_targets", {})
                    if analyst:
                        targets = []
                        for bank, data in analyst.items():
                            if isinstance(data, dict) and data.get("target"):
                                targets.append(f"{bank}: ${data['target']}")
                            elif isinstance(data, (int, float)):
                                targets.append(f"{bank}: ${data}")
                        if targets:
                            print(f"\n  Analyst targets: {', '.join(targets[:5])}")

                    cot = deep_ctx.get("cot_positioning", {})
                    if cot:
                        trend = cot.get("trend", "")
                        if trend:
                            print(f"  COT trend: {trend}")

                print()
    except Exception as e:
        logger.debug("Profile briefing error: %s", e)

    print()

    all_plans: list[dict] = []
    reports: list[str] = []
    log_entries: list[dict] = []

    for ticker in tickers:
        if ticker not in spot_data:
            print(f"  {ticker}: no live price available, skipping")
            continue

        sd = spot_data[ticker]
        spot = sd["price"]
        print(f"Fetching daily ranges for {ticker}...")
        daily_ranges = fetch_daily_ranges(ticker, days=10)

        print(f"Loading signals for {ticker}...")
        signal = load_signal_data(ticker)

        # Override direction if requested
        if args.direction != "auto":
            forced_dir = "LONG" if args.direction == "bull" else "SHORT"
            if forced_dir == "LONG":
                levels = compute_fishing_levels_bull(ticker, spot, signal, hours, daily_ranges)
            else:
                levels = compute_fishing_levels_bear(ticker, spot, signal, hours, daily_ranges)

            warrant_results = evaluate_warrants(
                ticker, spot, levels, args.budget, fx_rate, direction=forced_dir,
            )
            warrant_results = [r for r in warrant_results if r["fill_prob"] >= FISHING_MIN_FILL_PROB]

            inst_info = _build_instrument_info(warrant_results, forced_dir)
            plans = [{
                "ticker": ticker, "spot": spot, "direction": forced_dir,
                "conviction": 1.0, "levels": warrant_results,
                "instrument": inst_info,
                "tp_cascade": list(FISHING_TP_CASCADE),
                "sl_cascade": list(FISHING_SL_CASCADE),
            }] if warrant_results else []
        else:
            print(f"Computing fishing plan for {ticker}...")
            plans = compute_fishing_plan(ticker, spot, signal, hours, daily_ranges,
                                         budget_sek=args.budget, fx_rate=fx_rate)

        all_plans.extend(plans)

        report = format_report(ticker, sd, signal, daily_ranges, plans,
                               hours, max_levels=args.max_levels)
        reports.append(report)

        # Build log entry
        log_entries.append({
            "ticker": ticker,
            "spot": spot,
            "hours": hours,
            "regime": signal["regime"],
            "rsi": signal["rsi"],
            "atr_pct": signal["atr_pct"],
            "plans": [{
                "direction": p["direction"],
                "conviction": p["conviction"],
                "top_levels": [{
                    "level": r["level"],
                    "fill_prob": r["fill_prob"],
                    "ev_sek": r["ev_sek"],
                    "warrant": r["warrant"],
                } for r in p["levels"][:args.max_levels]],
            } for p in plans],
        })

    if args.telegram:
        # Print Telegram-format summary for each ticker's plans
        for ticker in tickers:
            ticker_plans = [p for p in all_plans if p["ticker"] == ticker]
            if ticker_plans:
                print(format_telegram_plan(ticker_plans))
                print()
    else:
        print("\n" + "=" * 80)
        print(f"FISHING PLAN -- {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} CET")
        print(f"Budget: {args.budget:.0f} SEK per level | FX: {fx_rate:.2f}")
        print("=" * 80 + "\n")

        for report in reports:
            print(report)
            print()

    # Log
    log_entry = {
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "command": "fin-fish",
        "budget_sek": args.budget,
        "fx_rate": fx_rate,
        "hours_remaining": hours,
        "metals": log_entries,
    }
    atomic_append_jsonl(FISH_LOG_PATH, log_entry)

    # --- Smart monitoring mode ---
    # Auto-detect active positions from Avanza or metals_positions_state
    # Start monitoring automatically when a position exists (no --monitor flag needed)
    should_monitor = args.monitor
    detected_position = None

    if not should_monitor and tickers:
        # Try to detect active positions
        try:
            from portfolio.avanza_session import get_positions
            positions = get_positions()
            silver_keywords = ("silver", "silv", "xag", "mini s silver", "mini l silver",
                               "bull silver", "bear silver")
            gold_keywords = ("guld", "gold", "xau", "bull guld", "bear guld")

            for pos in (positions or []):
                name = (pos.get("name") or "").lower()
                vol = pos.get("volume", 0)
                if vol <= 0:
                    continue

                for ticker in tickers:
                    keywords = silver_keywords if "XAG" in ticker else gold_keywords
                    if any(kw in name for kw in keywords):
                        detected_position = {
                            "ticker": ticker,
                            "name": pos.get("name", ""),
                            "volume": vol,
                            "value": pos.get("value", 0),
                            "avg_price": pos.get("averageAcquiredPrice", 0),
                            "last_price": pos.get("lastPrice", 0),
                            "is_short": any(k in name for k in ("bear", "mini s")),
                        }
                        should_monitor = True
                        print(f"\n  Active position detected: {pos.get('name')} ({vol}u)")
                        print("  Auto-starting smart monitor...\n")
                        break
                if detected_position:
                    break
        except Exception:
            pass  # Avanza unavailable — check persisted state
            try:
                pos_state = load_json(BASE_DIR / "data" / "metals_positions_state.json") or {}
                for key, pos in pos_state.items():
                    if pos.get("active") and any(t.lower().replace("-", "") in key.lower()
                                                  for t in tickers):
                        detected_position = {
                            "ticker": tickers[0],
                            "name": key,
                            "volume": pos.get("units", 0),
                            "value": 0,
                            "avg_price": pos.get("entry", 0),
                            "is_short": pos.get("direction", "").lower() == "short",
                        }
                        should_monitor = True
                        print(f"\n  Active position from state: {key}")
                        print("  Auto-starting smart monitor...\n")
                        break
            except Exception:
                pass

    if should_monitor and tickers:
        ticker = tickers[0]
        if ticker in spot_data:
            spot = spot_data[ticker]["price"]
            entry = args.entry_price if args.entry_price > 0 else spot

            # Use detected position info if available
            cert_price = args.cert_price
            cert_units = args.cert_units
            cert_leverage = args.leverage

            if detected_position and cert_price == 0:
                cert_price = detected_position.get("avg_price", 0) or 0
                cert_units = detected_position.get("volume", 0)

            # Determine direction
            if args.direction != "auto":
                direction = "LONG" if args.direction == "bull" else "SHORT"
            elif detected_position:
                direction = "SHORT" if detected_position.get("is_short") else "LONG"
            elif ticker in preflight_results:
                pf = preflight_results[ticker]
                direction = "LONG" if pf["bull_score"] > pf["bear_score"] else "SHORT"
            else:
                direction = "SHORT"

            entry_conviction = 50
            if ticker in preflight_results:
                pf = preflight_results[ticker]
                entry_conviction = pf["bull_score"] if direction == "LONG" else pf["bear_score"]

            try:
                from portfolio.fish_monitor_smart import SmartFishMonitor
                monitor = SmartFishMonitor(
                    ticker=ticker,
                    entry_price=entry,
                    direction=direction,
                    entry_conviction=entry_conviction,
                    cert_entry_price=cert_price,
                    cert_units=cert_units,
                    cert_leverage=cert_leverage,
                )
                monitor.run()
            except KeyboardInterrupt:
                print("\nMonitoring stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
