"""Compute and persist intraday seasonality profiles for metals.

Fetches 1h klines from Binance FAPI and computes hour-of-day
return/volatility profiles. Called periodically to keep profiles fresh.
"""
from __future__ import annotations

import logging

from portfolio.metals_orderbook import SYMBOL_MAP
from portfolio.seasonality import compute_hourly_profile, save_profiles, load_profiles

logger = logging.getLogger("portfolio.seasonality_updater")


def update_seasonality_profiles(tickers: list[str] | None = None) -> dict:
    """Fetch klines and recompute seasonality profiles for metals tickers.

    Args:
        tickers: List of tickers to update. Defaults to XAG-USD, XAU-USD.

    Returns:
        Dict of ticker -> profile (or None if failed).
    """
    if tickers is None:
        tickers = ["XAG-USD", "XAU-USD"]

    profiles = load_profiles()

    for ticker in tickers:
        try:
            df = _fetch_hourly_klines(ticker, limit=500)  # ~20 days
            if df is None or df.empty:
                logger.warning("No klines for %s, skipping", ticker)
                continue
            profile = compute_hourly_profile(df)
            if profile is not None:
                profiles[ticker] = profile
                logger.info("Updated seasonality profile for %s (%d hours)",
                           ticker, len(profile))
            else:
                logger.warning("Insufficient data for %s profile", ticker)
        except Exception as e:
            logger.warning("Seasonality update failed for %s: %s", ticker, e)

    if profiles:
        save_profiles(profiles)
    return profiles


def _fetch_hourly_klines(ticker: str, limit: int = 500):
    """Fetch 1h klines from Binance FAPI for a metals ticker."""
    import pandas as pd
    from portfolio.api_utils import BINANCE_FAPI_BASE
    from portfolio.http_retry import fetch_json
    from portfolio.shared_state import _binance_limiter

    symbol = SYMBOL_MAP.get(ticker)
    if not symbol:
        return None

    _binance_limiter.wait()
    data = fetch_json(
        f"{BINANCE_FAPI_BASE}/klines",
        params={"symbol": symbol, "interval": "1h", "limit": limit},
        timeout=15,
        label="seasonality_klines",
    )
    if not data:
        return None

    rows = []
    for k in data:
        rows.append({
            "time": pd.Timestamp(k[0], unit="ms", tz="UTC"),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        })
    df = pd.DataFrame(rows)
    df = df.set_index("time")
    return df
