"""Minimal OIL-USD signal source for the grid market-maker.

agent_summary.json does not currently publish OIL-USD signals (only
XAG/XAU/BTC/ETH/MSTR — see portfolio/tickers.py SYMBOLS). To let the
grid fisher arm oil instruments without dragging the full Layer 1
signal pipeline into oil, this module computes a small standalone
signal from Brent (BZ=F) klines: RSI(14) + EMA(9,21) momentum, packed
into the same ``(direction, confidence)`` shape the grid expects.

It writes the result to ``data/oil_grid_signal.json`` with a TTL so
the metals_loop reads cached values for most cycles and only refreshes
the underlying kline pull every ``REFRESH_INTERVAL_SEC`` seconds. The
fetch itself routes through ``portfolio.price_source.fetch_klines``,
which uses Binance FAPI when available and falls back to yfinance.

Returns ``None`` (and writes ``{"direction": null, "confidence": 0.0}``)
when the fetch fails so the grid stays idle on oil instead of placing
into a signal vacuum.
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
from typing import Optional

import pandas as pd

from portfolio.file_utils import atomic_write_json, load_json
from portfolio.price_source import SourceUnavailableError, fetch_klines

logger = logging.getLogger("portfolio.oil_grid_signal")

SIGNAL_FILE = "data/oil_grid_signal.json"

# Refresh interval — oil moves slow vs metals, so a 5-minute kline pull
# is more than enough for grid placement decisions.
REFRESH_INTERVAL_SEC = 300

# Underlying ticker. OLJAB warrants track Brent.
UNDERLYING = "BZ=F"
INTERVAL = "1h"
HISTORY_LIMIT = 300  # enough for EMA21 + RSI14 to settle


def _utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _bar_ts_iso(df: "pd.DataFrame") -> Optional[str]:
    """ISO-8601 (UTC) timestamp of the LAST bar in the frame, or None.

    2026-06-11 (audit B8 premortem hook 4): this is the bar's OWN timestamp
    (data age), distinct from fetched_ts (when we pulled it / cache
    freshness). BZ=F via yfinance lags 10-15 min, so a fresh fetch can
    return an old bar; consumers must judge data age off this, not the
    fetch time. Returns None if the index isn't a usable datetime.
    """
    try:
        idx = df.index[-1]
        ts = pd.Timestamp(idx)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:  # noqa: BLE001
        return None


def _rsi(series: pd.Series, period: int = 14) -> float:
    """Wilder's RSI on the latest bar; returns NaN if too few points."""
    if len(series) < period + 1:
        return float("nan")
    deltas = series.diff().dropna()
    gains = deltas.clip(lower=0)
    losses = -deltas.clip(upper=0)
    avg_gain = gains.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
    avg_loss = losses.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _signal_from_indicators(close: pd.Series) -> tuple[Optional[str], float, dict]:
    """Compute (direction, confidence, meta) from a close-price series."""
    if len(close) < 22:
        return None, 0.0, {"reason": "insufficient_history",
                           "bars": len(close)}
    ema9 = close.ewm(span=9, adjust=False).mean().iloc[-1]
    ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
    rsi = _rsi(close, 14)
    last = float(close.iloc[-1])

    if math.isnan(rsi):
        return None, 0.0, {"reason": "rsi_nan", "last": last}

    # Direction: EMA alignment + RSI position.
    # EMA9 > EMA21 and RSI < 70 -> LONG bias.
    # EMA9 < EMA21 and RSI > 30 -> SHORT bias.
    # Confidence scales with the trend strength + RSI distance from 50.
    ema_diff_pct = (ema9 - ema21) / ema21 * 100.0
    rsi_distance = abs(rsi - 50.0) / 50.0  # 0..1

    if ema_diff_pct > 0 and rsi < 70:
        # Strength: linear up to 2 % EMA spread + RSI offset
        base = min(0.55, 0.5 + min(abs(ema_diff_pct) / 2.0, 1.0) * 0.05)
        conf = min(0.8, base + rsi_distance * 0.2)
        direction = "LONG"
    elif ema_diff_pct < 0 and rsi > 30:
        base = min(0.55, 0.5 + min(abs(ema_diff_pct) / 2.0, 1.0) * 0.05)
        conf = min(0.8, base + rsi_distance * 0.2)
        direction = "SHORT"
    else:
        return None, 0.0, {
            "reason": "ambiguous", "rsi": round(rsi, 2),
            "ema9": round(ema9, 3), "ema21": round(ema21, 3),
            "last": last,
        }

    return direction, round(conf, 3), {
        "rsi": round(rsi, 2),
        "ema9": round(ema9, 3),
        "ema21": round(ema21, 3),
        "ema_diff_pct": round(ema_diff_pct, 3),
        "last_close": round(last, 3),
        "bars": len(close),
    }


def compute_signal() -> dict:
    """Pull Brent klines and produce a fresh signal dict.

    Returns the same shape as the cached file so callers can round-trip.
    """
    try:
        df = fetch_klines(
            UNDERLYING, interval=INTERVAL, limit=HISTORY_LIMIT,
            period="60d",
        )
    except SourceUnavailableError as exc:
        logger.warning("oil_grid_signal: fetch failed: %s", exc)
        now = _utcnow_iso()
        return {
            "ts": now,           # back-compat alias of fetched_ts
            "fetched_ts": now,   # when we fetched (cache freshness)
            "bar_ts": None,      # last bar's own ts (data age) — none on failure
            "underlying": UNDERLYING,
            "direction": None,
            "confidence": 0.0,
            "meta": {"reason": "fetch_failed", "error": str(exc)},
        }

    if df is None or df.empty or "close" not in df.columns:
        now = _utcnow_iso()
        return {
            "ts": now,
            "fetched_ts": now,
            "bar_ts": None,
            "underlying": UNDERLYING,
            "direction": None,
            "confidence": 0.0,
            "meta": {"reason": "empty_df"},
        }

    direction, confidence, meta = _signal_from_indicators(df["close"])
    now = _utcnow_iso()
    # 2026-06-11 (audit B8 premortem hook 4): split the timestamps.
    #   fetched_ts — when this fetch happened. get_cached_or_refresh keys the
    #     5-min cache TTL off THIS so the cache keeps working against a feed
    #     that lags 10-15 min (using bar_ts would expire the cache instantly
    #     and the oil leg would silently die hammering yfinance).
    #   bar_ts — the last bar's own timestamp (real data age). grid_fisher /
    #     metals_loop consumers judge staleness off this, not fetch time.
    # `ts` is retained as an alias of fetched_ts for back-compat.
    return {
        "ts": now,
        "fetched_ts": now,
        "bar_ts": _bar_ts_iso(df),
        "underlying": UNDERLYING,
        "direction": direction,
        "confidence": confidence,
        "meta": meta,
    }


def get_cached_or_refresh(force: bool = False) -> dict:
    """Return the current oil signal, refreshing from kline data when
    the cached value is older than ``REFRESH_INTERVAL_SEC``.

    The cache file lives at ``data/oil_grid_signal.json`` so other
    processes can read it without re-pulling klines.
    """
    cached = load_json(SIGNAL_FILE, default=None)
    # 2026-06-11 (audit B8 premortem hook 4): the cache TTL keys off
    # fetched_ts (when WE last pulled), NOT bar_ts (the bar's age). The feed
    # lags 10-15 min, so keying off bar_ts would make every cached value look
    # >5 min stale and refresh every cycle — defeating the cache and
    # hammering yfinance. Fall back to legacy `ts` for caches written before
    # the split.
    _fetch_ts = None
    if isinstance(cached, dict):
        _fetch_ts = cached.get("fetched_ts") or cached.get("ts")
    if not force and _fetch_ts:
        try:
            cached_dt = _dt.datetime.strptime(
                _fetch_ts, "%Y-%m-%dT%H:%M:%SZ",
            ).replace(tzinfo=_dt.timezone.utc)
        except ValueError:
            cached_dt = None
        if cached_dt is not None:
            age = (_dt.datetime.now(_dt.timezone.utc) - cached_dt).total_seconds()
            if age < REFRESH_INTERVAL_SEC:
                return cached

    fresh = compute_signal()
    try:
        atomic_write_json(SIGNAL_FILE, fresh)
    except Exception as exc:  # noqa: BLE001
        logger.warning("oil_grid_signal: write failed: %s", exc)
    return fresh


if __name__ == "__main__":
    import json
    sig = get_cached_or_refresh(force=True)
    print(json.dumps(sig, indent=2))
