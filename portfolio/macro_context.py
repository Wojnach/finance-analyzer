import logging
from pathlib import Path

import pandas as pd

from portfolio.api_utils import ALPACA_BASE, BINANCE_BASE, BINANCE_FAPI_BASE, get_alpaca_headers
from portfolio.http_retry import fetch_with_retry
from portfolio.shared_state import VOLUME_TTL as _VOLUME_TTL
from portfolio.shared_state import _alpaca_limiter, _binance_limiter, _cached, _yfinance_limiter

logger = logging.getLogger("portfolio.macro_context")

CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.json"

from datetime import UTC

from portfolio.tickers import TICKER_SOURCE_MAP as TICKER_MAP


def _alpaca_headers():
    return get_alpaca_headers()


DXY_TTL = 3600
# 2026-04-13: Intraday DXY cache is 3 min — 60m bars refresh at each hour
# boundary and we want to re-query shortly after the new bar closes.
DXY_INTRADAY_TTL = 180
TREASURY_TTL_VAL = 3600


def _fetch_dxy():
    """Fetch DXY data.

    2026-04-14: routed via price_source — DXY (DX-Y.NYB) is in the
    yfinance allowed-fallback list (no free real-time alternative),
    but this preserves a single upgrade point for the day a real-time
    DXY feed becomes available.
    """
    from portfolio.price_source import fetch_klines

    h = fetch_klines("DX-Y.NYB", interval="1d", limit=30, period="30d")
    if h is None or h.empty:
        return None

    close = h["close"]
    current = float(close.iloc[-1])
    sma20 = float(close.rolling(20).mean().iloc[-1])
    pct_5d = (
        float((close.iloc[-1] / close.iloc[-5] - 1) * 100) if len(close) >= 5 else 0
    )

    if current > sma20:
        trend = "strong"
    else:
        trend = "weak"

    return {
        "value": round(current, 2),
        "sma20": round(sma20, 2),
        "trend": trend,
        "change_5d_pct": round(pct_5d, 2),
    }


def get_dxy():
    return _cached("dxy", DXY_TTL, _fetch_dxy)


# --- Intraday DXY (2026-04-13) ---------------------------------------------
# The daily _fetch_dxy above feeds a daily-resolution sub-signal inside
# macro_regime. For 1-3h metals prediction (silver had 46.4% consensus
# accuracy at 3h despite DXY's R² ~0.6 inverse correlation with silver),
# we need 60m-bar DXY data. Primary source is yfinance DX-Y.NYB intraday;
# fallback is EURUSD=X which makes up ~57.6% of DXY weight and gives us
# a usable synth when the primary pseudo-ticker's intraday feed hiccups.


def _dxy_features_from_close(close, *, source: str) -> dict | None:
    """Build the intraday DXY payload from a pandas Close series."""
    import math

    if close is None or len(close) < 2:
        return None
    last = float(close.iloc[-1])
    if math.isnan(last):
        return None

    def _pct(periods: int) -> float:
        if len(close) < periods + 1:
            return float("nan")
        prior = float(close.iloc[-1 - periods])
        if prior == 0 or math.isnan(prior):
            return float("nan")
        return (last / prior - 1) * 100

    change_1h = _pct(1)
    change_3h = _pct(3)

    return {
        "value": round(last, 4),
        "change_1h_pct": round(change_1h, 4) if not math.isnan(change_1h) else None,
        "change_3h_pct": round(change_3h, 4) if not math.isnan(change_3h) else None,
        "source": source,
    }


def _fetch_dxy_intraday():
    """Fetch intraday DXY (60m bars). Fallback chain: primary index → EURUSD synth.

    2026-04-14: routed via price_source — DXY and EURUSD=X are both in
    the yfinance allowed-fallback list (no free real-time alternative).
    The router preserves the same fallback logic but centralizes it.
    """
    from portfolio.price_source import fetch_klines

    def _download(ticker: str):
        try:
            df = fetch_klines(ticker, interval="60m", limit=120, period="5d")
            if df is None or df.empty or "close" not in df.columns:
                return None
            return df["close"].dropna()
        except Exception as exc:
            logger.debug("price_source intraday fetch failed for %s: %s", ticker, exc)
            return None

    # Primary: DX-Y.NYB intraday 60m
    close = _download("DX-Y.NYB")
    result = _dxy_features_from_close(close, source="DX-Y.NYB")
    if result is not None and result.get("change_1h_pct") is not None:
        return result

    # Fallback: synthesize from EURUSD=X spot.
    # DXY weights EUR at ~57.6%; the single-factor approximation
    # DXY ≈ c × EURUSD^(-0.576) captures the bulk of DXY's directional
    # variance. The constant 58.0 does NOT match real DXY levels (~99) —
    # it is arbitrary. Only ``change_1h_pct`` / ``change_3h_pct`` from
    # this synth path are usable — the ``value`` field is meaningless.
    # Downstream consumers (signals/dxy_cross_asset.py) only read the
    # change fields, so this is safe.
    eurusd = _download("EURUSD=X")
    if eurusd is None or len(eurusd) == 0:
        return None
    synth = 58.0 * (eurusd ** -0.576)
    return _dxy_features_from_close(synth, source="EURUSD=X-synth")


def get_dxy_intraday():
    """Cached accessor for intraday DXY features."""
    return _cached("dxy_intraday", DXY_INTRADAY_TTL, _fetch_dxy_intraday)


def _fetch_klines(ticker):
    source_type, symbol = TICKER_MAP.get(ticker, (None, None))
    if source_type in ("binance", "binance_fapi"):
        base_url = BINANCE_FAPI_BASE if source_type == "binance_fapi" else BINANCE_BASE
        _binance_limiter.wait()
        r = fetch_with_retry(
            f"{base_url}/klines",
            params={"symbol": symbol, "interval": "15m", "limit": 100},
            timeout=10,
        )
        if r is None:
            return None
        r.raise_for_status()
        raw = r.json()
        df = pd.DataFrame(
            raw,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_vol",
                "trades",
                "tb",
                "tq",
                "ignore",
            ],
        )
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df
    elif source_type == "alpaca":
        from datetime import datetime

        _alpaca_limiter.wait()
        end = datetime.now(UTC)
        start = end - pd.Timedelta(days=5)
        r = fetch_with_retry(
            f"{ALPACA_BASE}/stocks/{symbol}/bars",
            headers=_alpaca_headers(),
            params={
                "timeframe": "15Min",
                "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "feed": "iex",
            },
            timeout=10,
        )
        if r is None:
            return None
        r.raise_for_status()
        bars = r.json().get("bars") or []
        if not bars:
            return None
        df = pd.DataFrame(bars)
        df = df.rename(
            columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        )
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df
    return None


def _fetch_volume_signal(ticker):
    """Compute volume signal from klines for a single ticker."""
    klines_df = _fetch_klines(ticker)
    if klines_df is None or klines_df.empty:
        return None
    vol = klines_df["volume"].astype(float)
    close = klines_df["close"].astype(float)
    if len(vol) < 2:
        return None
    last_vol = float(vol.iloc[-2])
    avg20 = (
        float(vol.iloc[:-1].rolling(20).mean().iloc[-1])
        if len(vol) >= 22
        else float(vol.iloc[:-1].mean())
    )
    ratio = last_vol / avg20 if avg20 > 0 else 1.0

    # Price direction over last 3 completed candles
    if len(close) >= 5:
        price_change = float(close.iloc[-2] / close.iloc[-5] - 1)
    else:
        price_change = 0.0

    # Volume spike (>1.5x avg) confirms direction
    # No spike = abstain (HOLD)
    if ratio > 1.5:
        if price_change > 0:
            action = "BUY"
        elif price_change < 0:
            action = "SELL"
        else:
            action = "HOLD"
    else:
        action = "HOLD"

    return {
        "ratio": round(ratio, 2),
        "spike": ratio > 1.5,
        "price_change_3": round(price_change * 100, 2),
        "action": action,
    }


def get_volume_signal(ticker):
    return _cached(f"vol_{ticker}", _VOLUME_TTL, _fetch_volume_signal, ticker)


from portfolio.fomc_dates import FOMC_DATES_ISO as FOMC_DATES


def _fred_10y_fallback():
    """FRED DGS10 fallback when yfinance ^TNX fails.

    Added 2026-04-09 after yfinance ^TNX fetch started returning None for
    extended periods (16h stale), triggering `TypeError('NoneType' object is
    not subscriptable')` from this function's callers.

    Returns a dict in the same shape as the yfinance path would for the "10y"
    key — {yield_pct, change_5d} — or None if FRED is also unavailable.
    Reuses `portfolio.golddigger.data_provider.fetch_us10y`, which already
    has its own 1h cache + circuit breaker.
    """
    try:
        import json as _json

        from portfolio.golddigger.data_provider import fetch_us10y
        fred_key = ""
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                fred_key = _json.load(f).get("golddigger", {}).get("fred_api_key", "") or ""
        except Exception:
            return None
        if not fred_key:
            return None
        yield_decimal = fetch_us10y(fred_key, series_id="DGS10")
        if yield_decimal is None:
            return None
        # fetch_us10y returns decimal (0.0425); yfinance path uses pct (4.25).
        return {"yield_pct": round(yield_decimal * 100, 3), "change_5d": 0.0}
    except Exception:
        logger.warning("FRED fallback failed for 10y", exc_info=True)
        return None


def _fetch_treasury():
    """Fetch treasury yield data, with FRED fallback for 10y.

    2026-04-14: routed via price_source. Treasury tickers (^TNX, ^TYX)
    are CBOE-style indices with no free intraday alternative; the router
    sends them through yfinance. 2YY=F is a futures pseudo-ticker also
    with no free alternative.
    """
    from portfolio.price_source import fetch_klines

    tickers = {"10y": "^TNX", "2y": "2YY=F", "30y": "^TYX"}
    result = {}
    for label, sym in tickers.items():
        try:
            h = fetch_klines(sym, interval="1d", limit=30, period="30d")
            if h is None or h.empty:
                continue
            close = h["close"]
            current = float(close.iloc[-1])
            pct_5d = (
                float((close.iloc[-1] / close.iloc[-5] - 1) * 100)
                if len(close) >= 5
                else 0
            )
            result[label] = {
                "yield_pct": round(current, 3),
                "change_5d": round(pct_5d, 2),
            }
        except Exception:
            logger.warning("Treasury fetch failed for %s", label, exc_info=True)

    # FRED fallback for 10y when yfinance ^TNX is down (common symptom:
    # No data / NoneType errors). Other maturities don't have a clean FRED
    # fallback via this helper, so they stay yfinance-only.
    if "10y" not in result:
        fallback = _fred_10y_fallback()
        if fallback is not None:
            logger.info("Treasury 10y: using FRED fallback (%.3f%%)", fallback["yield_pct"])
            result["10y"] = fallback

    if "10y" in result and "2y" in result:
        spread = result["10y"]["yield_pct"] - result["2y"]["yield_pct"]
        result["spread_2s10s"] = round(spread, 3)
        if spread < 0:
            result["curve"] = "inverted"
        elif spread < 0.2:
            result["curve"] = "flat"
        else:
            result["curve"] = "normal"

    return result or None


def get_treasury():
    return _cached("treasury", TREASURY_TTL_VAL, _fetch_treasury)


def get_fed_calendar():
    from datetime import datetime, timedelta

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    upcoming = [d for d in FOMC_DATES if d >= today]
    if not upcoming:
        return None

    next_date = upcoming[0]
    days_until = (
        datetime.strptime(next_date, "%Y-%m-%d") - datetime.strptime(today, "%Y-%m-%d")
    ).days

    is_meeting_day = today in FOMC_DATES
    is_day_before = any(
        (datetime.strptime(d, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        == today
        for d in FOMC_DATES
    )

    result = {
        "next_fomc": next_date,
        "days_until": days_until,
        "meetings_remaining": len(upcoming) // 2,
    }
    if is_meeting_day:
        result["warning"] = "FOMC meeting TODAY — expect volatility"
    elif is_day_before:
        result["warning"] = "FOMC meeting TOMORROW — positioning risk"
    elif days_until <= 7:
        result["warning"] = f"FOMC in {days_until} days — pre-meeting drift possible"

    return result


if __name__ == "__main__":
    dxy = get_dxy()
    print(f"DXY: {dxy}")
    treasury = get_treasury()
    print(f"Treasury: {treasury}")
    fed = get_fed_calendar()
    print(f"Fed: {fed}")
    for t in list(TICKER_MAP.keys()):
        print(f"{t}: {get_volume_signal(t)}")
