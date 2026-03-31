"""Microstructure snapshot accumulator for rolling OFI and spread history.

Called each cycle by metals_loop.py to build order book snapshot history.
The orderbook_flow signal reads the accumulated OFI and spread z-score
from the persisted state.

State is kept in memory (ring buffer) and persisted to
data/microstructure_state.json for cross-process access.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path

from portfolio.file_utils import atomic_write_json, load_json
from portfolio.microstructure import compute_ofi, spread_zscore

logger = logging.getLogger("portfolio.microstructure_state")

_STATE_FILE = Path("data/microstructure_state.json")
_MAX_SNAPSHOTS = 60  # ~30-60 min at 30-60s intervals
_MIN_SNAPSHOTS_FOR_OFI = 3
_MIN_SPREADS_FOR_ZSCORE = 10

# In-memory ring buffers per ticker
_snapshot_buffers: dict[str, deque] = {}
_spread_buffers: dict[str, deque] = {}


def _ensure_buffer(ticker: str) -> None:
    """Initialize ring buffers for a ticker if needed."""
    if ticker not in _snapshot_buffers:
        _snapshot_buffers[ticker] = deque(maxlen=_MAX_SNAPSHOTS)
    if ticker not in _spread_buffers:
        _spread_buffers[ticker] = deque(maxlen=_MAX_SNAPSHOTS)


def accumulate_snapshot(ticker: str, depth: dict) -> None:
    """Add an order book snapshot to the rolling buffer.

    Args:
        ticker: Canonical ticker (e.g. "XAG-USD")
        depth: Order book depth dict from metals_orderbook.get_orderbook_depth()
               Must have: best_bid, best_ask, bids, asks, spread
    """
    if depth is None:
        return
    _ensure_buffer(ticker)
    snapshot = {
        "best_bid": depth["best_bid"],
        "best_ask": depth["best_ask"],
        "bids": depth["bids"][:5],   # keep top 5 levels only
        "asks": depth["asks"][:5],
        "ts": depth.get("ts", int(time.time() * 1000)),
    }
    _snapshot_buffers[ticker].append(snapshot)
    _spread_buffers[ticker].append(depth["spread"])


def get_rolling_ofi(ticker: str) -> float:
    """Compute OFI from accumulated snapshots for a ticker.

    Returns cumulative OFI over the last N snapshots.
    Returns 0.0 if insufficient history.
    """
    _ensure_buffer(ticker)
    snapshots = list(_snapshot_buffers[ticker])
    if len(snapshots) < _MIN_SNAPSHOTS_FOR_OFI:
        return 0.0
    return compute_ofi(snapshots)


def get_spread_zscore(ticker: str) -> float | None:
    """Compute spread z-score from accumulated spread history.

    Returns z-score of current spread vs recent history.
    Returns None if insufficient data.
    """
    _ensure_buffer(ticker)
    spreads = list(_spread_buffers[ticker])
    if len(spreads) < _MIN_SPREADS_FOR_ZSCORE:
        return None
    return spread_zscore(spreads)


def get_microstructure_state(ticker: str) -> dict:
    """Get current accumulated microstructure state for a ticker.

    Returns dict with ofi and spread_zscore ready for the signal module.
    """
    ofi = get_rolling_ofi(ticker)
    sz = get_spread_zscore(ticker)
    _ensure_buffer(ticker)
    return {
        "ofi": ofi,
        "spread_zscore": sz if sz is not None else 0.0,
        "snapshot_count": len(_snapshot_buffers[ticker]),
        "spread_count": len(_spread_buffers[ticker]),
    }


def persist_state() -> None:
    """Write current microstructure state to disk for cross-process access."""
    state = {}
    for ticker in _snapshot_buffers:
        ms = get_microstructure_state(ticker)
        ms["ts"] = int(time.time() * 1000)
        state[ticker] = ms
    if state:
        atomic_write_json(_STATE_FILE, state)


def load_persisted_state(ticker: str) -> dict | None:
    """Read persisted microstructure state for a ticker.

    Used by orderbook_flow signal when running in a different process
    from metals_loop.
    """
    data = load_json(_STATE_FILE)
    if not data or ticker not in data:
        return None
    entry = data[ticker]
    age_ms = int(time.time() * 1000) - entry.get("ts", 0)
    if age_ms > 120_000:  # stale if >2 minutes old
        return None
    return entry


def snapshot_count(ticker: str) -> int:
    """Return current snapshot buffer size for a ticker."""
    _ensure_buffer(ticker)
    return len(_snapshot_buffers[ticker])
