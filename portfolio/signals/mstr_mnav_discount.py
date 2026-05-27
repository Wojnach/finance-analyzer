"""MSTR mNAV discount signal — valuation arbitrage detector (MSTR only).

Computes the ratio of MicroStrategy's market cap to its Bitcoin treasury
NAV (mNAV). When MSTR trades below 1.0x mNAV, equity holders are buying
BTC at a discount. When above ~1.5x, they pay a premium over direct BTC
exposure (less attractive post-ETF).

Sub-indicators:
    1. mNAV Ratio      -- current market_cap / (btc_holdings * btc_price)
    2. mNAV Velocity   -- 5-day rate-of-change of mNAV ratio
    3. Discount Depth   -- z-scored distance from fair value (1.0x)

Data:
    - BTC price: from the OHLCV DataFrame (if ticker=MSTR, BTC price is
      fetched via yfinance or shared_state cache)
    - MSTR shares outstanding: updated from SEC 10-Q filings (quarterly)
    - BTC holdings: updated from 8-K filings (~weekly purchase disclosures)

Applicable: MSTR only.
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np
import pandas as pd

from portfolio.http_retry import fetch_json
from portfolio.shared_state import _cached
from portfolio.signal_utils import safe_float

logger = logging.getLogger("portfolio.signals.mstr_mnav_discount")

MIN_ROWS = 20
MAX_CONFIDENCE = 0.7

_APPLICABLE_TICKERS = {"MSTR"}

MNAV_BUY = 0.95
MNAV_SELL = 1.50
MNAV_STRONG_BUY = 0.85
MNAV_STRONG_SELL = 2.0

VELOCITY_WINDOW = 5
VELOCITY_BUY = -0.03
VELOCITY_SELL = 0.05

Z_LOOKBACK = 60
Z_BUY = -1.5
Z_SELL = 1.5

_BTC_HOLDINGS = 843_738
_SHARES_OUTSTANDING = 26_382_000
_HOLDINGS_UPDATED = "2026-05-26"

_CACHE_TTL = 3600


def _fetch_btc_price() -> float | None:
    """Fetch current BTC-USD price. Uses yfinance with 1h cache."""
    def _do_fetch():
        try:
            import yfinance as yf
            data = yf.download("BTC-USD", period="5d", interval="1d", progress=False)
            if data is not None and len(data) > 0:
                return float(data["Close"].iloc[-1])
        except Exception:
            logger.exception("yfinance BTC fetch failed")
        return None

    return _cached("mstr_mnav_btc_price", _CACHE_TTL, _do_fetch)


def _mnav_ratio(mstr_price: float, btc_price: float) -> float:
    """Compute mNAV ratio = market_cap / btc_nav."""
    if btc_price <= 0 or mstr_price <= 0:
        return float("nan")
    market_cap = mstr_price * _SHARES_OUTSTANDING
    btc_nav = _BTC_HOLDINGS * btc_price
    return market_cap / btc_nav


def _mnav_level(ratio: float) -> tuple[float, str]:
    """Vote based on current mNAV ratio."""
    if np.isnan(ratio):
        return float("nan"), "HOLD"
    if ratio <= MNAV_STRONG_BUY:
        return ratio, "BUY"
    if ratio <= MNAV_BUY:
        return ratio, "BUY"
    if ratio >= MNAV_STRONG_SELL:
        return ratio, "SELL"
    if ratio >= MNAV_SELL:
        return ratio, "SELL"
    return ratio, "HOLD"


def _mnav_velocity(ratios: list[float], window: int = VELOCITY_WINDOW) -> tuple[float, str]:
    """Rate-of-change of mNAV ratio over window days."""
    if len(ratios) < window + 1:
        return float("nan"), "HOLD"
    current = ratios[-1]
    past = ratios[-(window + 1)]
    if past == 0 or np.isnan(past) or np.isnan(current):
        return float("nan"), "HOLD"
    roc = (current - past) / past
    roc_val = safe_float(roc)
    if np.isnan(roc_val):
        return float("nan"), "HOLD"
    if roc_val < VELOCITY_BUY:
        return roc_val, "BUY"
    if roc_val > VELOCITY_SELL:
        return roc_val, "SELL"
    return roc_val, "HOLD"


def _discount_depth_zscore(ratios: list[float], lookback: int = Z_LOOKBACK) -> tuple[float, str]:
    """Z-score of mNAV ratio distance from fair value (1.0x)."""
    if len(ratios) < lookback + 1:
        return float("nan"), "HOLD"
    distances = [r - 1.0 for r in ratios[-(lookback + 1):]]
    mean_d = np.nanmean(distances[:-1])
    std_d = np.nanstd(distances[:-1])
    if std_d == 0 or np.isnan(std_d):
        return float("nan"), "HOLD"
    current_d = distances[-1]
    z = (current_d - mean_d) / std_d
    z_val = safe_float(z)
    if np.isnan(z_val):
        return float("nan"), "HOLD"
    if z_val < Z_BUY:
        return z_val, "BUY"
    if z_val > Z_SELL:
        return z_val, "SELL"
    return z_val, "HOLD"


def compute_mstr_mnav_discount_signal(
    df: pd.DataFrame, context: dict = None
) -> dict:
    """Compute MSTR mNAV discount signal.

    Args:
        df: OHLCV DataFrame for MSTR
        context: dict with ticker, config, asset_class, regime
    """
    hold = {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}

    if df is None or len(df) < MIN_ROWS:
        return hold

    ticker = (context or {}).get("ticker", "")
    if ticker and ticker not in _APPLICABLE_TICKERS:
        return hold

    close = df["close"].dropna()
    if len(close) < MIN_ROWS:
        return hold

    mstr_price = safe_float(close.iloc[-1])
    if np.isnan(mstr_price) or mstr_price <= 0:
        return hold

    btc_price = _fetch_btc_price()
    if btc_price is None or btc_price <= 0:
        logger.warning("Cannot fetch BTC price for mNAV calculation")
        return hold

    current_ratio = _mnav_ratio(mstr_price, btc_price)
    if np.isnan(current_ratio):
        return hold

    historical_ratios = []
    for i in range(len(close)):
        r = _mnav_ratio(float(close.iloc[i]), btc_price)
        historical_ratios.append(r)

    level_val, level_vote = _mnav_level(current_ratio)
    vel_val, vel_vote = _mnav_velocity(historical_ratios)
    z_val, z_vote = _discount_depth_zscore(historical_ratios)

    votes = [level_vote, vel_vote, z_vote]
    buy_count = votes.count("BUY")
    sell_count = votes.count("SELL")

    if buy_count > sell_count:
        action = "BUY"
        confidence = buy_count / 3.0
    elif sell_count > buy_count:
        action = "SELL"
        confidence = sell_count / 3.0
    else:
        action = "HOLD"
        confidence = 0.0

    if current_ratio <= MNAV_STRONG_BUY:
        confidence = min(confidence + 0.15, 1.0)
    elif current_ratio >= MNAV_STRONG_SELL:
        confidence = min(confidence + 0.15, 1.0)

    confidence = min(confidence, MAX_CONFIDENCE)

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "sub_signals": {
            "mnav_level": level_vote,
            "mnav_velocity": vel_vote,
            "discount_depth_zscore": z_vote,
        },
        "indicators": {
            "mnav_ratio": round(safe_float(current_ratio), 4),
            "mnav_velocity_5d": round(safe_float(vel_val), 4),
            "discount_depth_z": round(safe_float(z_val), 4),
            "btc_price": round(btc_price, 2),
            "mstr_price": round(mstr_price, 2),
            "btc_holdings": _BTC_HOLDINGS,
            "shares_outstanding": _SHARES_OUTSTANDING,
            "holdings_updated": _HOLDINGS_UPDATED,
        },
    }
