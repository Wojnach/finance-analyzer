"""OHLCV for the Swedbank universe — Avanza primary, Alpaca fallback.

`portfolio.price_source.fetch_klines` cannot serve this universe: it routes any
bare uppercase symbol to Alpaca, which has no Swedish listings, so all 7
Stockholm instruments would simply fail. Avanza's price-chart endpoint covers
every instrument type through one path keyed on orderbook ID.

    GET /_api/price-chart/stock/<ob>?timePeriod=<p>&resolution=<r>
    -> {ohlc: [{timestamp, open, high, low, close, totalVolumeTraded}], metadata, ...}

`resolution` is lowercase and must appear in that period's
`metadata.resolution.availableResolutions`, else HTTP 400. Verified live:
three_years+day gives ~753 bars, five_years+day ~1257, for equities,
certificates and warrants alike.

Read-only (`api_get` only) and sequential, same as pricing.py — the real-money
metals loop shares this Avanza session.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("portfolio.swedbank.ohlcv")

# (timePeriod, resolution) per horizon we care about, chosen so each yields
# comfortably more than compute_indicators' 26-bar floor.
HORIZON_SPEC = {
    "1h": ("one_month", "hour"),
    "1d": ("one_year", "day"),
    "1w": ("five_years", "week"),
}
DEFAULT_HORIZON = "1d"
MIN_BARS = 26


class OhlcvError(RuntimeError):
    pass


_AVAILABILITY_TTL = 60.0
_availability = {"checked_at": 0.0, "ok": None}


def avanza_available(now_fn=None):
    """Whether an Avanza session exists and has not expired.

    Cheap file check, cached for a minute. Without it every US instrument on a
    machine with no session pays a doomed HTTP round-trip before falling back to
    Alpaca — 81s vs 36s for the same 26 instruments, measured on herc.

    Cached rather than per-call because a sweep asks 26 times in a row, and TTL'd
    rather than once-per-process because a login can land mid-run.
    """
    import time

    from portfolio.avanza_session import SESSION_FILE, session_remaining_minutes

    clock = now_fn or time.monotonic
    now = clock()
    if (
        _availability["ok"] is not None
        and (now - _availability["checked_at"]) < _AVAILABILITY_TTL
    ):
        return _availability["ok"]

    ok = SESSION_FILE.exists()
    if ok:
        remaining = session_remaining_minutes()
        # None means the file carries no expiry — usable, let the call decide.
        if remaining is not None and remaining <= 0:
            ok = False
    _availability.update({"checked_at": now, "ok": ok})
    return ok


def _avanza_chart(ob, period, resolution):
    from portfolio.avanza_session import api_get

    path = f"/_api/price-chart/stock/{ob}?timePeriod={period}"
    if resolution:
        path += f"&resolution={resolution}"
    return api_get(path)


def to_frame(payload):
    """Avanza chart payload -> DataFrame with the columns indicators expects."""
    import pandas as pd

    rows = (payload or {}).get("ohlc") or []
    if not rows:
        raise OhlcvError("empty ohlc series")
    df = pd.DataFrame(rows)
    missing = {"open", "high", "low", "close"} - set(df.columns)
    if missing:
        raise OhlcvError(f"chart payload missing columns: {sorted(missing)}")
    # compute_indicators reads a `volume` column; Avanza names it differently and
    # omits it on some instruments. Absent volume must not fail the whole fetch —
    # volume-dependent signals degrade on their own.
    df["volume"] = df.get("totalVolumeTraded", 0)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"])
    if len(df) < MIN_BARS:
        raise OhlcvError(f"only {len(df)} usable bars, need {MIN_BARS}")
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def fetch_by_ticker(ticker, horizon=DEFAULT_HORIZON):
    """OHLCV for a ticker that is NOT a pinned Swedbank instrument.

    The crypto underlyings (BTC-USD, ETH-USD) have no orderbook ID in our table,
    so they cannot go through the Avanza chart path. Route them to the canonical
    price_source router instead of silently falling back to the certificate,
    which would compute indicators on exactly the thin, decaying, SEK-denominated
    series the underlying mapping exists to avoid.
    """
    from portfolio.price_source import fetch_klines

    interval = {"1h": "1h", "1d": "1d", "1w": "1d"}.get(horizon, "1d")
    df = fetch_klines(ticker, interval=interval, limit=400)
    if df is None or len(df) < MIN_BARS:
        raise OhlcvError(
            f"{ticker}: price_source returned "
            f"{0 if df is None else len(df)} bars, need {MIN_BARS}"
        )
    out = df.copy()
    if "volume" not in out.columns:
        out["volume"] = 0
    if "timestamp" not in out.columns:
        out = out.reset_index().rename(columns={out.index.name or "index": "timestamp"})
    return out, "price_source"


def fetch(inst, horizon=DEFAULT_HORIZON, chart_fn=None, alpaca_fn=None):
    """OHLCV for one instrument. Avanza first, Alpaca only for US names.

    Returns (df, source). Raises OhlcvError when no source can supply enough
    bars — callers must treat that as "no signal", never as a neutral signal.
    """
    period, resolution = HORIZON_SPEC.get(horizon, HORIZON_SPEC[DEFAULT_HORIZON])
    fetch_chart = chart_fn or _avanza_chart
    # An explicit chart_fn is a caller override (tests, probes) and must still be
    # honoured even with no session on disk.
    skip_avanza = chart_fn is None and not avanza_available()
    if skip_avanza and not inst.has_fallback:
        raise OhlcvError(
            f"{inst.key}: no Avanza session and no fallback exists (Stockholm listing)"
        )
    if not skip_avanza:
        try:
            return to_frame(fetch_chart(inst.avanza_ob, period, resolution)), "avanza"
        except Exception as exc:
            logger.warning(
                "swedbank ohlcv: avanza chart failed for %s: %s", inst.key, exc
            )

    if not inst.has_fallback:
        raise OhlcvError(
            f"{inst.key}: avanza chart unavailable and no fallback exists "
            f"(Stockholm listing)"
        )
    try:
        get_bars = alpaca_fn or _alpaca_bars
        df = get_bars(inst.alpaca, horizon)
        if df is None or len(df) < MIN_BARS:
            raise OhlcvError(f"alpaca returned {0 if df is None else len(df)} bars")
        return df, "alpaca:fallback"
    except OhlcvError:
        raise
    except Exception as exc:
        raise OhlcvError(f"{inst.key}: no OHLCV from any source ({exc})") from exc


def _alpaca_bars(symbol, horizon):
    from portfolio.price_source import fetch_klines

    interval = {"1h": "1h", "1d": "1d", "1w": "1d"}.get(horizon, "1d")
    limit = {"1h": 400, "1d": 300, "1w": 300}.get(horizon, 300)
    df = fetch_klines(symbol, interval=interval, limit=limit)
    if df is None or df.empty:
        return None
    out = df.copy()
    if "volume" not in out.columns:
        out["volume"] = 0
    if "timestamp" not in out.columns:
        out = out.reset_index().rename(columns={out.index.name or "index": "timestamp"})
    return out
