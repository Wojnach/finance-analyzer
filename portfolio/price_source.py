"""Canonical real-time price-source router.

yfinance is **LAST RESORT** only. Use it exclusively for CBOE volatility
indices (^VIX, ^VIX3M, ^OVX, ^GVZ) where no free live alternative
exists. For everything else this module routes to a real-time source:

* Commodity underlyings (XAG, XAU, oil)            → Binance FAPI perps
* Crypto (BTC, ETH)                                → Binance SPOT
* US stocks / ETFs (MSTR, SPY, QQQ, USO, …)        → Alpaca IEX feed
* Treasury yields / macro daily series             → FRED
* FX (EUR/USD)                                     → Alpha Vantage FX_DAILY

Benchmark reference (measured 2026-04-14 13:00 CET):

====== source ===============  data age  api latency
Binance FAPI XAGUSDT             7.7 s       445 ms    ← PRIMARY
Avanza market-guide              tick      1,259 ms
yfinance SI=F                   655.4 s      389 ms    ← 85× stale

Every consumer of historical / intraday OHLCV should call
:func:`fetch_klines` rather than importing yfinance directly. The
function raises :class:`SourceUnavailableError` if no live alternative
exists AND the yfinance fallback is not allowed for the requested
ticker.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger("portfolio.price_source")


# ---------------------------------------------------------------------------
# Ticker classification — keep this table explicit so future additions are
# obvious. The router is alias-aware: 'XAG-USD', 'SI=F', 'XAGUSDT' all
# resolve to the same Binance FAPI symbol.
# ---------------------------------------------------------------------------

_BINANCE_FAPI = {
    # silver
    "XAG-USD": "XAGUSDT", "SI=F": "XAGUSDT", "XAGUSDT": "XAGUSDT",
    # gold
    "XAU-USD": "XAUUSDT", "GC=F": "XAUUSDT", "XAUUSDT": "XAUUSDT",
}

_BINANCE_SPOT = {
    "BTC-USD": "BTCUSDT", "BTCUSDT": "BTCUSDT",
    "ETH-USD": "ETHUSDT", "ETHUSDT": "ETHUSDT",
}

# CBOE-proprietary volatility indices. No free live alternative exists —
# these remain on yfinance by design. If you ever find a real-time feed,
# update this set AND the router.
_CBOE_VOL_INDICES = frozenset({
    "^VIX", "^VIX3M", "^OVX", "^GVZ", "^RVX", "^VXN",
})

# Tickers for which yfinance is the only available free data source.
# The router raises a WARNING when any of these fires so we can quantify
# how much yfinance dependency remains.
_YFINANCE_LAST_RESORT = frozenset({
    "HG=F",          # copper — no Binance perpetual
    "DX-Y.NYB",      # DXY pseudo-ticker (we also have an Alpha Vantage fallback)
    "EURUSD=X",      # FX — Alpha Vantage is paid for intraday
}) | _CBOE_VOL_INDICES


class SourceUnavailableError(RuntimeError):
    """Raised when no price source can serve the requested ticker."""


# ---------------------------------------------------------------------------
# Per-source fetchers — thin wrappers over the existing data_collector
# helpers + a new yfinance helper for the allowed fallback paths.
# ---------------------------------------------------------------------------


def _binance_interval(interval: str) -> str:
    """Normalize yfinance-style intervals to Binance format.

    yfinance uses '60m' / '90m' — Binance uses '1h' / no 90m equivalent.
    This function translates common cases. Unrecognized intervals pass
    through unchanged so a direct Binance interval (e.g. '1h') still works.
    """
    mapping = {
        "60m": "1h",
        "90m": "1h",   # Binance has no 90m — closest down-sample is 1h
        "120m": "2h",
    }
    return mapping.get(interval, interval)


def _fetch_binance_fapi(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    from portfolio.data_collector import binance_fapi_klines

    return binance_fapi_klines(symbol, interval=_binance_interval(interval), limit=limit)


def _fetch_binance_spot(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    from portfolio.data_collector import binance_klines

    return binance_klines(symbol, interval=_binance_interval(interval), limit=limit)


def _fetch_alpaca(ticker: str, interval: str, limit: int) -> pd.DataFrame:
    from portfolio.data_collector import alpaca_klines

    return alpaca_klines(ticker, interval=interval, limit=limit)


def _fetch_yfinance(
    ticker: str, interval: str, period: str | None = None, limit: int | None = None,
) -> pd.DataFrame:
    """Yfinance last-resort fetcher. Emits a WARNING every call so we can see
    how much yfinance residue is left in the system."""
    import yfinance as yf

    logger.warning(
        "price_source: falling back to yfinance for %s (interval=%s, period=%s). "
        "This source lags 10-15 min; upstream caller should be on Binance/Alpaca/FRED if possible.",
        ticker, interval, period,
    )
    p = period or "5d"
    df = yf.download(
        ticker, period=p, interval=interval,
        progress=False, auto_adjust=True,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if df.empty:
        return df
    # Normalize column casing (yfinance uses capitalized; our downstream
    # code expects lowercase after the alpaca/binance path).
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    return df.tail(limit) if limit else df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_source(ticker: str) -> str:
    """Return the routing decision for ``ticker``: one of
    {"binance_fapi", "binance_spot", "alpaca", "yfinance"}.

    Exported for diagnostics and tests.
    """
    if ticker in _BINANCE_FAPI:
        return "binance_fapi"
    if ticker in _BINANCE_SPOT:
        return "binance_spot"
    if ticker in _YFINANCE_LAST_RESORT:
        return "yfinance"
    # Default assumption: a bare uppercase symbol is a US stock/ETF → Alpaca.
    # Anything starting with '^' that isn't a CBOE vol index is an index we
    # don't have a mapping for — route to yfinance with warning.
    if ticker.startswith("^"):
        return "yfinance"
    return "alpaca"


def fetch_klines(
    ticker: str,
    interval: str = "1d",
    limit: int = 100,
    period: str | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV bars for ``ticker`` from the freshest available source.

    Args:
        ticker: Symbol in any recognized alias form (XAG-USD, SI=F,
            XAGUSDT, BTC-USD, MSTR, ^VIX, etc.)
        interval: Binance/Alpaca-style interval string
            ("1m", "5m", "1h", "1d").
        limit: Maximum rows to return.
        period: Optional yfinance-style period string ("5d", "3mo") —
            used only on the yfinance fallback path.

    Returns:
        ``pandas.DataFrame`` with lowercase OHLCV columns
        (``open, high, low, close, volume``).

    Raises:
        SourceUnavailableError: If every applicable source failed.
    """
    source = resolve_source(ticker)

    try:
        if source == "binance_fapi":
            return _fetch_binance_fapi(_BINANCE_FAPI[ticker], interval, limit)
        if source == "binance_spot":
            return _fetch_binance_spot(_BINANCE_SPOT[ticker], interval, limit)
        if source == "alpaca":
            return _fetch_alpaca(ticker, interval, limit)
        # yfinance fallback
        return _fetch_yfinance(ticker, interval, period=period, limit=limit)
    except Exception as exc:
        # If a primary source (Binance/Alpaca) fails AND the ticker isn't
        # explicitly yfinance-only, fall through to yfinance as emergency
        # backup so the loop doesn't lose data entirely. Emit an error log
        # so we can see how often the primary is flaking.
        if source in ("binance_fapi", "binance_spot", "alpaca"):
            logger.error(
                "price_source: primary source %s FAILED for %s (%r). "
                "Falling back to yfinance. Investigate the primary outage.",
                source, ticker, exc,
            )
            try:
                return _fetch_yfinance(ticker, interval, period=period, limit=limit)
            except Exception as exc2:
                raise SourceUnavailableError(
                    f"All sources failed for {ticker}: primary={source} "
                    f"({exc!r}), fallback=yfinance ({exc2!r})"
                ) from exc2
        raise SourceUnavailableError(
            f"yfinance failed for {ticker}: {exc!r}"
        ) from exc


def is_yfinance_allowed(ticker: str) -> bool:
    """True if yfinance is an approved primary source for this ticker
    (CBOE vol indices + the explicit last-resort list).

    Callers that want a direct yfinance path (for reasons beyond
    :func:`fetch_klines`'s OHLCV contract — e.g. yfinance's earnings
    calendar or fundamentals APIs) should check this first and log a
    WARNING if they still go to yfinance for a ticker outside this list.
    """
    return ticker in _YFINANCE_LAST_RESORT


__all__ = [
    "fetch_klines",
    "resolve_source",
    "is_yfinance_allowed",
    "SourceUnavailableError",
]
