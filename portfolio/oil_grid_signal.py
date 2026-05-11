"""Minimal OIL-USD signal feed for the grid market-maker.

The main signal pipeline (portfolio/main.py SYMBOLS) does not yet
register OIL-USD as a Tier-1 ticker — adding it requires data-source
wiring, per-signal accuracy backfill, and main-loop cycle headroom we
don't have. This module fills the gap with a small standalone signal
that:

  * fetches BZ=F (Brent front-month) 1h klines via
    portfolio.price_source — matches the OLJAB certs we trade, which
    track Brent
  * computes RSI / MACD / EMA / BB via portfolio.indicators
  * votes a 4-signal consensus
  * persists ``action`` + ``confidence`` to ``data/oil_signal_state.json``

Grid fisher reads that file each tick and merges it into the signal
dict it passes to GridFisher.tick(). When the file is missing or
stale, grid fisher falls back to ``no_direction`` and oil instruments
stay idle.

Hold profile inherits the grid fisher's: minutes to hours, EOD-flat.
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
from pathlib import Path
from typing import Any

from portfolio.file_utils import atomic_write_json

logger = logging.getLogger("portfolio.oil_grid_signal")

OIL_TICKER = "BZ=F"  # Brent front-month — matches the OLJAB X5 certs
                     # the grid fisher trades. WTI (CL=F) was considered
                     # but Brent correlates 1:1 with the certs and avoids
                     # an extra basis-risk layer.
SIGNAL_STATE_FILE = "data/oil_signal_state.json"
FRESHNESS_S = 300  # consumers treat files older than 5 min as stale


def _utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _vote_rsi(rsi: float) -> str:
    if rsi <= 30:
        return "BUY"
    if rsi >= 70:
        return "SELL"
    return "HOLD"


def _vote_macd(macd_hist: float) -> str:
    if macd_hist > 0:
        return "BUY"
    if macd_hist < 0:
        return "SELL"
    return "HOLD"


def _vote_ema(ema9: float, ema21: float) -> str:
    if ema9 > ema21 * 1.005:  # 0.5% deadband matches main loop convention
        return "BUY"
    if ema9 < ema21 * 0.995:
        return "SELL"
    return "HOLD"


def _vote_bb(close: float, bb_lower: float, bb_upper: float) -> str:
    if close <= bb_lower:
        return "BUY"
    if close >= bb_upper:
        return "SELL"
    return "HOLD"


def compute_signal() -> dict[str, Any]:
    """Fetch oil OHLCV, compute indicators, vote consensus.

    Returns a dict with ``action`` in {"BUY","SELL","HOLD"},
    ``confidence`` in [0,1], plus the individual votes and indicator
    values. On any fetch / compute failure, returns ``{"action": "HOLD",
    "confidence": 0.0, "error": "<msg>"}`` so callers always get a
    structurally-valid record.
    """
    try:
        from portfolio.price_source import fetch_klines  # noqa: PLC0415
        from portfolio.indicators import compute_indicators  # noqa: PLC0415
    except ImportError as exc:
        return {
            "action": "HOLD", "confidence": 0.0, "ts": _utcnow_iso(),
            "error": f"import_failed: {exc}",
        }

    try:
        df = fetch_klines(OIL_TICKER, interval="1h", limit=100)
    except Exception as exc:  # noqa: BLE001
        return {
            "action": "HOLD", "confidence": 0.0, "ts": _utcnow_iso(),
            "error": f"fetch_failed: {exc}",
        }
    if df is None or len(df) < 30:
        return {
            "action": "HOLD", "confidence": 0.0, "ts": _utcnow_iso(),
            "error": f"insufficient_data: rows={0 if df is None else len(df)}",
        }

    ind = compute_indicators(df)
    if ind is None:
        return {
            "action": "HOLD", "confidence": 0.0, "ts": _utcnow_iso(),
            "error": "compute_indicators_returned_none",
        }

    rsi = float(ind.get("rsi") or 50)
    macd_hist = float(ind.get("macd_hist") or 0)
    ema9 = float(ind.get("ema9") or ind.get("close") or 0)
    ema21 = float(ind.get("ema21") or ind.get("close") or 0)
    close = float(ind.get("close") or 0)
    bb_lower = float(ind.get("bb_lower") or close)
    bb_upper = float(ind.get("bb_upper") or close)

    if not all(math.isfinite(v) for v in (rsi, macd_hist, ema9, ema21,
                                          close, bb_lower, bb_upper)):
        return {
            "action": "HOLD", "confidence": 0.0, "ts": _utcnow_iso(),
            "error": "non_finite_indicator",
        }

    votes = {
        "rsi": _vote_rsi(rsi),
        "macd": _vote_macd(macd_hist),
        "ema": _vote_ema(ema9, ema21),
        "bb": _vote_bb(close, bb_lower, bb_upper),
    }
    buy_count = sum(1 for v in votes.values() if v == "BUY")
    sell_count = sum(1 for v in votes.values() if v == "SELL")
    total = len(votes)

    if buy_count > sell_count:
        action = "BUY"
        # Confidence: fraction of voters agreeing, scaled to the
        # 0.50-0.85 band so we sit above the grid fisher's 0.56 floor
        # only when at least 3 of 4 signals concur.
        agree = buy_count / total
    elif sell_count > buy_count:
        action = "SELL"
        agree = sell_count / total
    else:
        action = "HOLD"
        agree = 0.0
    # Map agreement [0.5, 1.0] -> confidence [0.50, 0.85] linearly,
    # then clip. 2/4 agreement -> 0.50 (below floor, won't arm); 3/4
    # -> 0.675 (above floor); 4/4 -> 0.85.
    if agree >= 0.5:
        confidence = round(0.50 + (agree - 0.5) * 0.70, 4)
    else:
        confidence = 0.0

    return {
        "action": action,
        "confidence": confidence,
        "ts": _utcnow_iso(),
        "ticker": "OIL-USD",
        "source_ticker": OIL_TICKER,
        "rsi": round(rsi, 2),
        "macd_hist": round(macd_hist, 4),
        "ema9": round(ema9, 4),
        "ema21": round(ema21, 4),
        "close": round(close, 4),
        "bb_lower": round(bb_lower, 4),
        "bb_upper": round(bb_upper, 4),
        "votes": votes,
        "buy_count": buy_count,
        "sell_count": sell_count,
    }


def write_signal(path: str | Path = SIGNAL_STATE_FILE) -> dict[str, Any]:
    """Compute the oil signal and atomically write it to *path*.

    Returns the computed record so the caller can log or telemeter it.
    """
    record = compute_signal()
    try:
        atomic_write_json(path, record)
    except Exception as exc:  # noqa: BLE001
        logger.warning("oil_signal_feed: write to %s failed: %s", path, exc)
        record = {**record, "write_error": str(exc)}
    return record


def load_signal(
    path: str | Path = SIGNAL_STATE_FILE,
    max_age_s: float = FRESHNESS_S,
) -> dict[str, Any] | None:
    """Load the persisted oil signal if fresh.

    Returns ``None`` when the file is missing or older than
    ``max_age_s``. Stale records are treated as missing so consumers
    don't act on data older than one tick interval.
    """
    from portfolio.file_utils import load_json  # noqa: PLC0415

    record = load_json(path, default=None)
    if not isinstance(record, dict):
        return None
    ts_str = record.get("ts")
    if not ts_str:
        return None
    try:
        ts = _dt.datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=_dt.timezone.utc,
        )
    except ValueError:
        return None
    age_s = (_dt.datetime.now(_dt.timezone.utc) - ts).total_seconds()
    if age_s > max_age_s:
        return None
    return record
