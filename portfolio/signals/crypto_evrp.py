"""Crypto Expected Volatility Risk Premium (eVRP) signal module.

Measures the gap between implied volatility (Deribit DVOL 30-day) and
realized volatility (10-day rolling from OHLCV). The VRP is persistently
positive in crypto — IV overstates RV most of the time.

Sub-indicators:
    1. eVRP Level     — raw DVOL - RV spread vs absolute thresholds
    2. eVRP Percentile — where current eVRP sits in 90-day distribution
    3. eVRP Momentum  — 5-day change in eVRP (rising = vol compression)

When eVRP is very high (>10), implied vol far exceeds realized vol — the
market is pricing in risk that isn't materializing.  Historically this
precedes mean-reversion downward in IV, often coinciding with bullish
price action (vol compression = calm = uptrend).

When eVRP is very negative (<-10), realized vol exceeds implied — the
market is underpricing risk.  This typically occurs during panic selloffs
and can signal contrarian BUY opportunities.

Data: Deribit public REST API (free, no authentication).
Applicable: BTC-USD, ETH-USD only (crypto assets with Deribit DVOL).

Source: Zarattini, Mele & Aziz (2025) "The Volatility Edge";
        github.com/pi-mis/btc-dvol-strategy.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote, safe_float

logger = logging.getLogger("portfolio.signals.crypto_evrp")

MIN_ROWS = 20  # Need enough bars for realized vol computation
RV_WINDOW = 10  # 10-day realized vol lookback
EVRP_BUY_THRESHOLD = -10.0  # eVRP below this → BUY (vol expansion)
EVRP_SELL_THRESHOLD = 10.0  # eVRP above this → SELL (vol compression)
PCTILE_WINDOW = 90  # Rolling window for percentile ranking
PCTILE_BUY = 10  # Below 10th percentile → BUY
PCTILE_SELL = 90  # Above 90th percentile → SELL
MOMENTUM_WINDOW = 5  # Days to measure eVRP change
MAX_CONFIDENCE = 0.7  # Cap for external-data signals

# Deribit API config
_DERIBIT_BASE = "https://www.deribit.com/api/v2"
_DVOL_CACHE: dict[str, tuple[float, float]] = {}  # currency -> (timestamp, dvol)
_DVOL_CACHE_TTL = 4 * 3600  # 4 hours
_DVOL_HISTORY_CACHE: dict[str, tuple[float, pd.Series]] = {}
_DVOL_HISTORY_TTL = 6 * 3600  # 6 hours

# Ticker → Deribit currency mapping
_TICKER_TO_CURRENCY = {
    "BTC-USD": "BTC",
    "BTCUSDT": "BTC",
    "ETH-USD": "ETH",
    "ETHUSDT": "ETH",
}


def _fetch_dvol_latest(currency: str = "BTC") -> Optional[float]:
    """Fetch latest DVOL value from Deribit public API.

    Returns DVOL as annualized implied vol percentage, or None on failure.
    """
    cache_key = currency
    cached = _DVOL_CACHE.get(cache_key)
    if cached and (time.time() - cached[0]) < _DVOL_CACHE_TTL:
        return cached[1]

    try:
        from portfolio.http_retry import fetch_with_retry

        now_ms = int(time.time() * 1000)
        start_ms = now_ms - (7 * 24 * 3600 * 1000)  # Last 7 days

        resp = fetch_with_retry(
            f"{_DERIBIT_BASE}/public/get_volatility_index_data",
            params={
                "currency": currency,
                "resolution": 86400,  # Daily
                "start_timestamp": start_ms,
                "end_timestamp": now_ms,
            },
            timeout=15,
            retries=2,
        )

        if resp is None or resp.status_code != 200:
            logger.warning("Deribit DVOL fetch failed: status=%s", getattr(resp, "status_code", None))
            return None

        data = resp.json()
        bars = data.get("result", {}).get("data", [])
        if not bars:
            logger.warning("No DVOL data returned for %s", currency)
            return None

        # Bars are [timestamp, open, high, low, close]
        latest_bar = bars[-1]
        dvol = float(latest_bar[4])  # close

        _DVOL_CACHE[cache_key] = (time.time(), dvol)
        return dvol

    except Exception as e:
        logger.warning("Deribit DVOL fetch error: %s", e)
        return None


def _fetch_dvol_history(currency: str = "BTC", days: int = 120) -> Optional[pd.Series]:
    """Fetch DVOL history for percentile/momentum computation.

    Returns pd.Series indexed by date with DVOL values, or None on failure.
    """
    cache_key = currency
    cached = _DVOL_HISTORY_CACHE.get(cache_key)
    if cached and (time.time() - cached[0]) < _DVOL_HISTORY_TTL:
        return cached[1]

    try:
        from portfolio.http_retry import fetch_with_retry

        now_ms = int(time.time() * 1000)
        start_ms = now_ms - (days * 24 * 3600 * 1000)

        all_bars = []
        chunk_end = now_ms

        for _ in range(5):  # Max 5 chunks
            resp = fetch_with_retry(
                f"{_DERIBIT_BASE}/public/get_volatility_index_data",
                params={
                    "currency": currency,
                    "resolution": 86400,
                    "start_timestamp": start_ms,
                    "end_timestamp": chunk_end,
                },
                timeout=15,
                retries=2,
            )

            if resp is None or resp.status_code != 200:
                break

            data = resp.json()
            bars = data.get("result", {}).get("data", [])
            if not bars:
                break

            all_bars.extend(bars)
            earliest = bars[0][0]
            if earliest <= start_ms:
                break
            chunk_end = earliest - 1
            time.sleep(0.2)  # Rate limiting

        if not all_bars:
            return None

        df = pd.DataFrame(all_bars, columns=["ts", "open", "high", "low", "close"])
        df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.date
        df = df.sort_values("date").drop_duplicates("date")
        series = df.set_index("date")["close"].astype(float)
        series.index = pd.to_datetime(series.index)

        _DVOL_HISTORY_CACHE[cache_key] = (time.time(), series)
        return series

    except Exception as e:
        logger.warning("Deribit DVOL history fetch error: %s", e)
        return None


def _compute_realized_vol(close: pd.Series, window: int = RV_WINDOW) -> float:
    """Compute annualized realized volatility from close prices.

    Uses log returns with rolling standard deviation, annualized for crypto
    (365 trading days).
    """
    if len(close) < window + 1:
        return float("nan")

    log_ret = np.log(close / close.shift(1))
    rv = log_ret.rolling(window).std() * np.sqrt(365) * 100
    val = rv.iloc[-1]

    return float(val) if not np.isnan(val) else float("nan")


def _evrp_level_signal(evrp: float) -> str:
    """Sub-signal 1: eVRP absolute level."""
    if evrp < EVRP_BUY_THRESHOLD:
        return "BUY"
    if evrp > EVRP_SELL_THRESHOLD:
        return "SELL"
    return "HOLD"


def _evrp_percentile_signal(dvol_history: Optional[pd.Series],
                             current_evrp: float,
                             rv_series: pd.Series) -> tuple[str, float]:
    """Sub-signal 2: eVRP percentile rank in recent history.

    Returns (signal, percentile_value).
    """
    if dvol_history is None or len(dvol_history) < PCTILE_WINDOW:
        return "HOLD", 50.0

    # Compute historical eVRP series
    # Align DVOL history with RV from price data
    rv_hist = np.log(rv_series / rv_series.shift(1)).rolling(RV_WINDOW).std() * np.sqrt(365) * 100
    rv_hist = rv_hist.dropna()

    if len(rv_hist) < PCTILE_WINDOW:
        return "HOLD", 50.0

    # Use the last PCTILE_WINDOW RV values to compute historical eVRP
    recent_rv = rv_hist.iloc[-PCTILE_WINDOW:]
    recent_dvol = dvol_history.iloc[-PCTILE_WINDOW:] if len(dvol_history) >= PCTILE_WINDOW else dvol_history

    # If lengths don't align, use simple percentile of current eVRP vs recent DVOL-RV
    if len(recent_dvol) < 30:
        return "HOLD", 50.0

    # Simple approach: rank current eVRP against recent DVOL spread
    # Use just the DVOL percentile as proxy (DVOL carries most of the signal)
    dvol_vals = recent_dvol.values
    current_dvol = dvol_history.iloc[-1] if len(dvol_history) > 0 else None
    if current_dvol is None:
        return "HOLD", 50.0

    pctile = float(np.sum(dvol_vals < current_dvol) / len(dvol_vals) * 100)

    if pctile < PCTILE_BUY:
        return "BUY", pctile  # IV unusually low → vol expansion coming
    if pctile > PCTILE_SELL:
        return "SELL", pctile  # IV unusually high → vol compression coming
    return "HOLD", pctile


def _evrp_momentum_signal(dvol_history: Optional[pd.Series]) -> tuple[str, float]:
    """Sub-signal 3: eVRP 5-day momentum (direction of DVOL change).

    Rising DVOL = market pricing in more risk = bearish.
    Falling DVOL = risk receding = bullish.
    """
    if dvol_history is None or len(dvol_history) < MOMENTUM_WINDOW + 1:
        return "HOLD", 0.0

    recent = dvol_history.iloc[-(MOMENTUM_WINDOW + 1):]
    change = float(recent.iloc[-1] - recent.iloc[0])

    if change < -5.0:  # DVOL dropped 5+ points
        return "BUY", change
    if change > 5.0:  # DVOL rose 5+ points
        return "SELL", change
    return "HOLD", change


def compute_crypto_evrp_signal(df: pd.DataFrame, context: dict = None) -> dict:
    """Compute crypto Expected Volatility Risk Premium signal.

    Args:
        df: DataFrame with columns: open, high, low, close, volume
        context: Optional dict with keys: ticker, asset_class, config

    Returns:
        dict with keys: action, confidence, sub_signals, indicators
    """
    empty = {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}

    if df is None or len(df) < MIN_ROWS:
        return empty

    # Only applicable to crypto assets
    ticker = (context or {}).get("ticker", "")
    asset_class = (context or {}).get("asset_class", "")

    if asset_class and asset_class not in ("crypto",):
        return empty

    currency = _TICKER_TO_CURRENCY.get(ticker, "")
    if not currency:
        # Try to infer from ticker
        if "BTC" in ticker.upper():
            currency = "BTC"
        elif "ETH" in ticker.upper():
            currency = "ETH"
        else:
            return empty

    # Step 1: Compute realized vol from OHLCV
    close = df["close"].dropna()
    rv = _compute_realized_vol(close, RV_WINDOW)

    if np.isnan(rv) or rv <= 0:
        return empty

    # Step 2: Fetch DVOL from Deribit
    dvol = _fetch_dvol_latest(currency)

    if dvol is None or dvol <= 0:
        logger.debug("DVOL unavailable for %s, returning HOLD", currency)
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {},
            "indicators": {"rv_10d": safe_float(rv), "dvol": None, "evrp": None},
        }

    # Step 3: Compute eVRP
    evrp = dvol - rv

    # Step 4: Fetch DVOL history for percentile and momentum
    dvol_history = _fetch_dvol_history(currency, days=120)

    # Sub-signal 1: eVRP absolute level
    level_vote = _evrp_level_signal(evrp)

    # Sub-signal 2: eVRP percentile
    pctile_vote, pctile_val = _evrp_percentile_signal(dvol_history, evrp, close)

    # Sub-signal 3: eVRP momentum
    momentum_vote, momentum_val = _evrp_momentum_signal(dvol_history)

    # Majority vote
    votes = [level_vote, pctile_vote, momentum_vote]
    action, confidence = majority_vote(votes, count_hold=False)

    # Cap confidence
    confidence = min(confidence, MAX_CONFIDENCE)

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": {
            "evrp_level": level_vote,
            "evrp_percentile": pctile_vote,
            "evrp_momentum": momentum_vote,
        },
        "indicators": {
            "dvol_30d": safe_float(dvol),
            "rv_10d": safe_float(rv),
            "evrp": safe_float(evrp),
            "evrp_percentile": safe_float(pctile_val),
            "dvol_momentum_5d": safe_float(momentum_val),
            "currency": currency,
        },
    }
