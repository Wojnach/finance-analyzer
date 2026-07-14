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

import pandas as pd

logger = logging.getLogger("portfolio.price_source")


# ---------------------------------------------------------------------------
# Ticker classification — keep this table explicit so future additions are
# obvious. The router is alias-aware: 'XAG-USD', 'SI=F', 'XAGUSDT' all
# resolve to the same Binance FAPI symbol.
# ---------------------------------------------------------------------------

_BINANCE_FAPI = {
    # silver
    "XAG-USD": "XAGUSDT",
    "SI=F": "XAGUSDT",
    "XAGUSDT": "XAGUSDT",
    # gold
    "XAU-USD": "XAUUSDT",
    "GC=F": "XAUUSDT",
    "XAUUSDT": "XAUUSDT",
}

_BINANCE_SPOT = {
    "BTC-USD": "BTCUSDT",
    "BTCUSDT": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "ETHUSDT": "ETHUSDT",
}

# 2026-05-28: yfinance does not recognize the dashed/Binance metal aliases
# (XAG-USD/XAU-USD return an EMPTY frame; XAGUSDT/XAUUSDT are unknown). The
# emergency yfinance fallback below was calling _fetch_yfinance() with the
# ORIGINAL ticker, so a Binance-FAPI outage on silver/gold lost data entirely
# instead of failing over (the docstring advertises alias-awareness but it
# wasn't applied on the fallback path). Translate to the yfinance futures
# symbol first.
_YFINANCE_ALIAS = {
    "XAG-USD": "SI=F",
    "XAGUSDT": "SI=F",
    "XAU-USD": "GC=F",
    "XAUUSDT": "GC=F",
}


def _to_yfinance_symbol(ticker: str) -> str:
    """Map a Binance/dashed alias to a yfinance-valid symbol (identity if none)."""
    return _YFINANCE_ALIAS.get(ticker, ticker)


# CBOE-proprietary volatility indices. No free live alternative exists —
# these remain on yfinance by design. If you ever find a real-time feed,
# update this set AND the router.
_CBOE_VOL_INDICES = frozenset(
    {
        "^VIX",
        "^VIX3M",
        "^OVX",
        "^GVZ",
        "^RVX",
        "^VXN",
    }
)

# Tickers for which yfinance is the only available free data source.
# Calls to these emit DEBUG (not WARNING) so legitimate use doesn't
# pollute the log; calls for non-allowed tickers emit WARNING so we
# can quantify residual leakage.
_YFINANCE_LAST_RESORT = (
    frozenset(
        {
            "HG=F",  # copper — no Binance perpetual
            "DX-Y.NYB",  # DXY pseudo-ticker (Alpha Vantage FX is paid intraday)
            "EURUSD=X",  # FX — Alpha Vantage paid intraday
            "^TNX",  # 10y treasury yield (CBOE; FRED has daily DGS10 fallback)
            "^TYX",  # 30y treasury yield
            "2YY=F",  # 2y treasury yield futures pseudo-ticker
            "^FVX",  # 5y treasury yield
            # Oil futures — no free real-time alternative (Binance has no oil
            # perpetual; Alpaca has the USO ETF but not the underlying futures).
            # 2026-05-01: added when oil_loop went live in DRY_RUN — previously
            # CL=F/BZ=F routed to Alpaca (futures unsupported) → fallback to
            # yfinance with WARNING noise every 60s cycle. oil_precompute had
            # been silently relying on the same fallback path.
            "CL=F",  # WTI front-month
            "BZ=F",  # Brent front-month
            "RB=F",  # RBOB gasoline (used for crack-spread context)
        }
    )
    | _CBOE_VOL_INDICES
)


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
        "90m": "1h",  # Binance has no 90m — closest down-sample is 1h
        "120m": "2h",
    }
    return mapping.get(interval, interval)


def _fetch_binance_fapi(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    from portfolio.data_collector import binance_fapi_klines

    return binance_fapi_klines(
        symbol, interval=_binance_interval(interval), limit=limit
    )


def _fetch_binance_spot(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    from portfolio.data_collector import binance_klines

    return binance_klines(symbol, interval=_binance_interval(interval), limit=limit)


def _fetch_alpaca(ticker: str, interval: str, limit: int) -> pd.DataFrame:
    from portfolio.data_collector import alpaca_klines

    return alpaca_klines(ticker, interval=interval, limit=limit)


def _fetch_yfinance(
    ticker: str,
    interval: str,
    period: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Yfinance fetcher. Allowed-list tickers emit DEBUG; everything else
    emits WARNING so we can quantify residual leakage."""
    import yfinance as yf

    if ticker in _YFINANCE_LAST_RESORT:
        logger.debug(
            "price_source: yfinance for %s (interval=%s, period=%s) — allowed (no live alt)",
            ticker,
            interval,
            period,
        )
    else:
        logger.warning(
            "price_source: falling back to yfinance for %s (interval=%s, period=%s). "
            "This source lags 10-15 min; upstream caller should be on Binance/Alpaca/FRED if possible.",
            ticker,
            interval,
            period,
        )
    p = period or "5d"
    df = yf.download(
        ticker,
        period=p,
        interval=interval,
        progress=False,
        auto_adjust=True,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if df.empty:
        raise SourceUnavailableError(f"yfinance returned empty DataFrame for {ticker}")
    # Normalize column casing (yfinance uses capitalized; our downstream
    # code expects lowercase after the alpaca/binance path).
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    return df.tail(limit) if limit else df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_fail_closed_cache = {"ts": 0.0, "val": None}


def _price_fail_closed() -> bool:
    """Whether traded-instrument price fetches fail closed on primary outage.

    Default True (never serve delayed yfinance into a live trade). Override
    via config.json ``price_source.fail_closed = false`` to restore the old
    emergency-fallback behavior. Re-read every 30s — togglable live.
    """
    import time as _time

    now = _time.time()
    if now - _fail_closed_cache["ts"] < 30 and _fail_closed_cache["val"] is not None:
        return _fail_closed_cache["val"]
    val = True
    try:
        import json as _json
        import os as _os

        path = _os.path.join(_os.path.dirname(__file__), "..", "config.json")
        with open(path) as f:
            cfg = _json.load(f).get("price_source", {})
        if isinstance(cfg, dict) and "fail_closed" in cfg:
            val = bool(cfg["fail_closed"])
    except Exception:
        val = True
    _fail_closed_cache.update(ts=now, val=val)
    return val


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
        return _fetch_yfinance(
            _to_yfinance_symbol(ticker), interval, period=period, limit=limit
        )
    except Exception as exc:
        # Traded instruments (Binance crypto/metals, Alpaca stocks) have a
        # real-time source. If it fails, FAIL CLOSED — abstain this cycle
        # rather than silently serving 15-min-delayed yfinance prices into
        # a live trade decision (user directive 2026-07-14: never trade on
        # delayed data). yfinance-native context tickers (VIX/yields/DXY/
        # oil) are unaffected — they take the direct yfinance path above,
        # never this fallback branch.
        if source in ("binance_fapi", "binance_spot", "alpaca"):
            if _price_fail_closed():
                raise SourceUnavailableError(
                    f"{source} failed for {ticker} ({exc!r}); fail-closed — "
                    f"refusing stale yfinance fallback for a traded instrument"
                ) from exc
            logger.error(
                "price_source: primary source %s FAILED for %s (%r). "
                "Falling back to yfinance. Investigate the primary outage.",
                source,
                ticker,
                exc,
            )
            try:
                df = _fetch_yfinance(
                    _to_yfinance_symbol(ticker), interval, period=period, limit=limit
                )
                df.attrs["_source"] = "yfinance_fallback"
                df.attrs["_primary_failed"] = source
                return df
            except Exception as exc2:
                raise SourceUnavailableError(
                    f"All sources failed for {ticker}: primary={source} "
                    f"({exc!r}), fallback=yfinance ({exc2!r})"
                ) from exc2
        raise SourceUnavailableError(f"yfinance failed for {ticker}: {exc!r}") from exc


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
