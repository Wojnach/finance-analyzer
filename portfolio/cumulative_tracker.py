"""Cumulative price change tracker — rolling 1d/3d/7d price changes.

Logs hourly price snapshots to data/price_snapshots_hourly.jsonl and computes
rolling changes so messages can show "XAG +12.4% 7d".
"""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from portfolio.file_utils import atomic_append_jsonl, load_jsonl
from portfolio.shared_state import _cached

logger = logging.getLogger("portfolio.cumulative_tracker")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SNAPSHOTS_FILE = DATA_DIR / "price_snapshots_hourly.jsonl"

# Minimum interval between snapshots (55 minutes)
_MIN_SNAPSHOT_INTERVAL_SEC = 55 * 60

# Cache TTL for cumulative summary
_CUMULATIVE_CACHE_TTL = 300  # 5 min


def maybe_log_hourly_snapshot(prices_usd):
    """Append a price snapshot if >55 min since last entry.

    Args:
        prices_usd: dict {ticker: price_usd} for all tracked instruments.

    Returns:
        True if a snapshot was logged, False if skipped (too recent).
    """
    if not prices_usd:
        return False

    # Check last snapshot timestamp
    last_ts = _get_last_snapshot_ts()
    now = time.time()
    if last_ts and (now - last_ts) < _MIN_SNAPSHOT_INTERVAL_SEC:
        return False

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "prices": {k: round(v, 4) for k, v in prices_usd.items() if v},
    }

    atomic_append_jsonl(SNAPSHOTS_FILE, entry)
    logger.debug("Logged hourly price snapshot (%d tickers)", len(entry["prices"]))
    return True


def _get_last_snapshot_ts():
    """Get the timestamp of the most recent snapshot as epoch seconds.

    Reads just the last line of the file for efficiency.
    """
    if not SNAPSHOTS_FILE.exists():
        return None

    last_line = None
    try:
        with open(SNAPSHOTS_FILE, "rb") as f:
            # Seek to end and read backwards to find last line
            f.seek(0, 2)
            size = f.tell()
            if size == 0:
                return None
            # Read last 2KB (more than enough for one JSON line)
            read_size = min(size, 2048)
            f.seek(size - read_size)
            chunk = f.read().decode("utf-8", errors="replace")
            lines = chunk.strip().split("\n")
            last_line = lines[-1].strip()
    except (OSError, IndexError):
        return None

    if not last_line:
        return None

    try:
        entry = json.loads(last_line)
        ts = datetime.fromisoformat(entry["ts"])
        return ts.timestamp()
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def compute_rolling_changes(tickers=None, snapshots=None):
    """Compute rolling price changes over 1d, 3d, 7d windows.

    Args:
        tickers: Optional list of tickers to compute for. None = all.
        snapshots: Pre-loaded snapshot list (for testing). None = load from file.

    Returns:
        dict: {ticker: {"change_1d": +2.3, "change_3d": +5.1, "change_7d": +12.4}}
        Changes are in percent. None values if insufficient data.
    """
    if snapshots is None:
        snapshots = load_jsonl(SNAPSHOTS_FILE)

    if not snapshots:
        return {}

    now = datetime.now(timezone.utc)
    latest = snapshots[-1]
    latest_prices = latest.get("prices", {})

    windows = {
        "change_1d": timedelta(days=1),
        "change_3d": timedelta(days=3),
        "change_7d": timedelta(days=7),
    }

    result = {}

    for ticker, current_price in latest_prices.items():
        if tickers and ticker not in tickers:
            continue
        if not current_price or current_price <= 0:
            continue

        changes = {}
        for label, delta in windows.items():
            target_ts = now - delta
            old_price = _find_closest_price(snapshots, ticker, target_ts)
            if old_price and old_price > 0:
                changes[label] = round(((current_price - old_price) / old_price) * 100, 2)
            else:
                changes[label] = None

        result[ticker] = changes

    return result


def _find_closest_price(snapshots, ticker, target_ts, max_hours=6):
    """Find the price closest to target_ts within max_hours tolerance.

    Args:
        snapshots: List of snapshot dicts (sorted by time).
        ticker: Ticker to look up.
        target_ts: Target datetime (UTC).
        max_hours: Maximum acceptable distance in hours.

    Returns:
        float or None: The closest price found, or None if none within range.
    """
    best_price = None
    best_delta = None

    for snap in snapshots:
        try:
            snap_ts = datetime.fromisoformat(snap["ts"])
        except (KeyError, ValueError):
            continue

        price = snap.get("prices", {}).get(ticker)
        if price is None:
            continue

        delta = abs((snap_ts - target_ts).total_seconds()) / 3600
        if delta > max_hours:
            continue

        if best_delta is None or delta < best_delta:
            best_price = price
            best_delta = delta

    return best_price


def get_cumulative_summary(tickers=None):
    """Main entry point. Returns rolling changes + top movers.

    Cached for 5 minutes via shared_state._cached().

    Args:
        tickers: Optional list of tickers. None = all.

    Returns:
        dict: {
            "ticker_changes": {ticker: {"change_1d": ..., "change_3d": ..., "change_7d": ...}},
            "movers": [{"ticker": ..., "change_3d": ..., "change_7d": ...}]
        }
    """
    def _compute():
        changes = compute_rolling_changes(tickers=tickers)

        # Identify movers: abs(3d) > 5% or abs(7d) > 10%
        movers = []
        for ticker, c in changes.items():
            c3d = c.get("change_3d")
            c7d = c.get("change_7d")
            if (c3d is not None and abs(c3d) > 5.0) or \
               (c7d is not None and abs(c7d) > 10.0):
                movers.append({
                    "ticker": ticker,
                    "change_3d": c3d,
                    "change_7d": c7d,
                })

        # Sort movers by absolute 7d change (or 3d if 7d unavailable)
        movers.sort(key=lambda m: abs(m.get("change_7d") or m.get("change_3d") or 0), reverse=True)

        return {
            "ticker_changes": changes,
            "movers": movers,
        }

    cache_key = f"cumulative_summary_{','.join(tickers) if tickers else 'all'}"
    return _cached(cache_key, _CUMULATIVE_CACHE_TTL, _compute)
