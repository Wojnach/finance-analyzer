"""Cross-asset data for metals prediction.

Fetches correlated markets that carry predictive information for
1-3h gold and silver moves:
    - Copper (HG=F): industrial demand proxy, correlated with silver
    - GVZ: CBOE Gold ETF Volatility Index (implied vol for gold)
    - SPY: S&P 500 ETF (risk-on/risk-off gauge)
    - Gold/Silver ratio: mean-reverting ratio, extreme readings signal

All data fetched via yfinance with caching to avoid rate limits.

2026-04-13: Added intraday (60m bar) fetchers next to the existing daily
ones after 4,916-sample measurement showed metals_cross_asset at 29.1%
on XAG 3h — root cause was 5-day lookbacks evaluated against 3h outcomes
(see docs/AVANZA_RESILIENCE_PLAN.md follow-up). Daily fetchers preserved
for longer-horizon callers; the metals_cross_asset signal switched to
intraday by default.
"""
from __future__ import annotations

import logging
from functools import wraps

import pandas as pd

from portfolio.shared_state import _cached, _yfinance_limiter

logger = logging.getLogger("portfolio.metals_cross_assets")

_CROSS_TTL = 300
_GVZ_TTL = 600
# Intraday TTL is shorter — 60m bars refresh at the start of each hour,
# and we want to re-query shortly after the bar closes to pick up the new row.
_CROSS_INTRADAY_TTL = 180


def _yf_download(ticker: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV from yfinance with rate limiting."""
    import yfinance as yf
    _yfinance_limiter.wait()
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df
    except Exception as e:
        logger.warning("yfinance fetch failed for %s: %s", ticker, e)
        return pd.DataFrame()


def _nocache(func):
    """Mark function so tests can bypass _cached via func.__wrapped__."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    wrapper.__wrapped__ = func
    return wrapper


def _pct_change(series: pd.Series, periods: int) -> float:
    """Percentage change over N periods, returns NaN on insufficient data."""
    if len(series) < periods + 1:
        return float("nan")
    return float((series.iloc[-1] / series.iloc[-1 - periods] - 1) * 100)


@_nocache
def get_copper_data() -> dict | None:
    """Copper futures (HG=F) price and momentum."""
    def _fetch():
        df = _yf_download("HG=F", period="3mo", interval="1d")
        if df.empty or "Close" not in df.columns:
            return None
        close = df["Close"].dropna()
        if len(close) < 20:
            return None
        return {
            "price": float(close.iloc[-1]),
            "change_1d_pct": _pct_change(close, 1),
            "change_5d_pct": _pct_change(close, 5),
            "sma20": float(close.rolling(20).mean().iloc[-1]),
            "vs_sma20_pct": float((close.iloc[-1] / close.rolling(20).mean().iloc[-1] - 1) * 100),
        }
    return _cached("cross_copper", _CROSS_TTL, _fetch)


@_nocache
def get_gvz() -> dict | None:
    """CBOE Gold ETF Volatility Index (^GVZ)."""
    def _fetch():
        df = _yf_download("^GVZ", period="3mo", interval="1d")
        if df.empty or "Close" not in df.columns:
            return None
        close = df["Close"].dropna()
        if len(close) < 10:
            return None
        level = float(close.iloc[-1])
        mean20 = float(close.rolling(20, min_periods=10).mean().iloc[-1])
        std20 = float(close.rolling(20, min_periods=10).std().iloc[-1])
        zscore = (level - mean20) / std20 if std20 > 0.01 else 0.0
        return {
            "level": level,
            "change_1d_pct": _pct_change(close, 1),
            "sma20": mean20,
            "zscore": zscore,
        }
    return _cached("cross_gvz", _GVZ_TTL, _fetch)


@_nocache
def get_gold_silver_ratio() -> dict | None:
    """Gold/Silver price ratio and deviation from mean."""
    def _fetch():
        gold_df = _yf_download("GC=F", period="6mo", interval="1d")
        silver_df = _yf_download("SI=F", period="6mo", interval="1d")
        if gold_df.empty or silver_df.empty:
            return None
        gold_close = gold_df["Close"].dropna()
        silver_close = silver_df["Close"].dropna()
        if len(gold_close) < 20 or len(silver_close) < 20:
            return None
        common = gold_close.index.intersection(silver_close.index)
        if len(common) < 20:
            return None
        g = gold_close.loc[common]
        s = silver_close.loc[common]
        ratio = g / s
        current = float(ratio.iloc[-1])
        sma20 = float(ratio.rolling(20).mean().iloc[-1])
        std20 = float(ratio.rolling(20).std().iloc[-1])
        zscore = (current - sma20) / std20 if std20 > 0.01 else 0.0
        return {
            "ratio": current,
            "sma20": sma20,
            "zscore": zscore,
            "change_5d_pct": _pct_change(ratio, 5),
        }
    return _cached("cross_gs_ratio", _CROSS_TTL, _fetch)


@_nocache
def get_oil_data() -> dict | None:
    """WTI Crude Oil futures (CL=F) price and momentum."""
    def _fetch():
        df = _yf_download("CL=F", period="3mo", interval="1d")
        if df.empty or "Close" not in df.columns:
            return None
        close = df["Close"].dropna()
        if len(close) < 10:
            return None
        return {
            "price": float(close.iloc[-1]),
            "change_1d_pct": _pct_change(close, 1),
            "change_5d_pct": _pct_change(close, 5),
        }
    return _cached("cross_oil", _CROSS_TTL, _fetch)


@_nocache
def get_spy_return() -> dict | None:
    """S&P 500 ETF (SPY) recent returns for risk-on/risk-off."""
    def _fetch():
        df = _yf_download("SPY", period="1mo", interval="1d")
        if df.empty or "Close" not in df.columns:
            return None
        close = df["Close"].dropna()
        if len(close) < 5:
            return None
        return {
            "price": float(close.iloc[-1]),
            "change_1d_pct": _pct_change(close, 1),
            "change_5d_pct": _pct_change(close, 5),
        }
    return _cached("cross_spy", _CROSS_TTL, _fetch)


def get_all_cross_asset_data() -> dict:
    """Fetch all cross-asset features in one call (daily bars)."""
    return {
        "copper": get_copper_data(),
        "gvz": get_gvz(),
        "gold_silver_ratio": get_gold_silver_ratio(),
        "spy": get_spy_return(),
        "oil": get_oil_data(),
    }


# --- Intraday variants (60m bars, for 1-3h prediction horizons) ---
#
# yfinance 60m interval supports up to 730 days of history. We use 5d
# period which yields ~35 hourly bars — enough for 3h change (3 bars) and
# intraday rolling stats. On weekends/holidays the last ~2 days of bars
# may be sparse; `_pct_change` returns NaN and signal votes HOLD.


@_nocache
def get_copper_intraday() -> dict | None:
    """Copper 60m bars. Exposes change_1h_pct + change_3h_pct."""
    def _fetch():
        df = _yf_download("HG=F", period="5d", interval="60m")
        if df.empty or "Close" not in df.columns:
            return None
        close = df["Close"].dropna()
        if len(close) < 4:
            return None
        return {
            "price": float(close.iloc[-1]),
            "change_1h_pct": _pct_change(close, 1),
            "change_3h_pct": _pct_change(close, 3),
        }
    return _cached("cross_copper_intraday", _CROSS_INTRADAY_TTL, _fetch)


@_nocache
def get_gold_silver_ratio_intraday() -> dict | None:
    """Gold/Silver ratio 60m bars. Exposes ratio_change_3h_pct."""
    def _fetch():
        gold_df = _yf_download("GC=F", period="5d", interval="60m")
        silver_df = _yf_download("SI=F", period="5d", interval="60m")
        if gold_df.empty or silver_df.empty:
            return None
        gold_close = gold_df["Close"].dropna()
        silver_close = silver_df["Close"].dropna()
        if len(gold_close) < 4 or len(silver_close) < 4:
            return None
        common = gold_close.index.intersection(silver_close.index)
        if len(common) < 4:
            return None
        ratio = gold_close.loc[common] / silver_close.loc[common]
        return {
            "ratio": float(ratio.iloc[-1]),
            "change_1h_pct": _pct_change(ratio, 1),
            "change_3h_pct": _pct_change(ratio, 3),
        }
    return _cached("cross_gs_ratio_intraday", _CROSS_INTRADAY_TTL, _fetch)


@_nocache
def get_oil_intraday() -> dict | None:
    """WTI crude 60m bars."""
    def _fetch():
        df = _yf_download("CL=F", period="5d", interval="60m")
        if df.empty or "Close" not in df.columns:
            return None
        close = df["Close"].dropna()
        if len(close) < 4:
            return None
        return {
            "price": float(close.iloc[-1]),
            "change_1h_pct": _pct_change(close, 1),
            "change_3h_pct": _pct_change(close, 3),
        }
    return _cached("cross_oil_intraday", _CROSS_INTRADAY_TTL, _fetch)


@_nocache
def get_spy_intraday() -> dict | None:
    """SPY 60m bars — captures intraday risk-on/risk-off."""
    def _fetch():
        df = _yf_download("SPY", period="5d", interval="60m")
        if df.empty or "Close" not in df.columns:
            return None
        close = df["Close"].dropna()
        if len(close) < 4:
            return None
        return {
            "price": float(close.iloc[-1]),
            "change_1h_pct": _pct_change(close, 1),
            "change_3h_pct": _pct_change(close, 3),
        }
    return _cached("cross_spy_intraday", _CROSS_INTRADAY_TTL, _fetch)


def get_all_cross_asset_intraday() -> dict:
    """Fetch all intraday (60m) cross-asset features in one call.

    GVZ is intentionally absent — it's a daily-published index with no
    intraday bars. Callers should still read `get_gvz()` for GVZ context.
    """
    return {
        "copper": get_copper_intraday(),
        "gold_silver_ratio": get_gold_silver_ratio_intraday(),
        "spy": get_spy_intraday(),
        "oil": get_oil_intraday(),
    }
