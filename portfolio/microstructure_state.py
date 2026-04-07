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
_MIN_OFI_HISTORY_FOR_ZSCORE = 10
_MAX_OFI_HISTORY = 120  # ~2h of OFI readings for z-score normalization

# Multi-scale OFI windows (snapshot counts)
_OFI_WINDOW_FAST = 5   # ~5 min
_OFI_WINDOW_MEDIUM = 15  # ~15 min
# slow = all snapshots (full buffer)

# In-memory ring buffers per ticker
_snapshot_buffers: dict[str, deque] = {}
_spread_buffers: dict[str, deque] = {}
_ofi_history: dict[str, deque] = {}  # rolling OFI values for z-score


def _ensure_buffer(ticker: str) -> None:
    """Initialize ring buffers for a ticker if needed."""
    if ticker not in _snapshot_buffers:
        _snapshot_buffers[ticker] = deque(maxlen=_MAX_SNAPSHOTS)
    if ticker not in _spread_buffers:
        _spread_buffers[ticker] = deque(maxlen=_MAX_SNAPSHOTS)
    if ticker not in _ofi_history:
        _ofi_history[ticker] = deque(maxlen=_MAX_OFI_HISTORY)


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


def record_ofi(ticker: str, ofi_val: float) -> None:
    """Record an OFI value for z-score history tracking.

    Called once per cycle from get_microstructure_state to avoid
    double-appending if get_rolling_ofi is called multiple times.
    """
    _ensure_buffer(ticker)
    _ofi_history[ticker].append(ofi_val)


def get_ofi_zscore(ticker: str) -> float:
    """Z-score of current OFI relative to its own rolling distribution.

    Normalizes OFI per asset so gold and BTC use comparable thresholds.
    Returns 0.0 if insufficient history.
    """
    _ensure_buffer(ticker)
    history = list(_ofi_history[ticker])
    if len(history) < _MIN_OFI_HISTORY_FOR_ZSCORE:
        return 0.0
    import numpy as np
    arr = np.array(history, dtype=float)
    mean = arr.mean()
    std = arr.std()
    if std < 1e-12:
        return 0.0
    return float((arr[-1] - mean) / std)


def get_multiscale_ofi(ticker: str) -> dict:
    """Compute OFI at 3 time scales: fast (~5min), medium (~15min), slow (full).

    Returns dict with ofi_fast, ofi_medium, ofi_slow, and flow_acceleration
    (fast z-score minus slow z-score — positive = accelerating buying).
    """
    _ensure_buffer(ticker)
    snapshots = list(_snapshot_buffers[ticker])
    n = len(snapshots)

    ofi_slow = compute_ofi(snapshots) if n >= _MIN_SNAPSHOTS_FOR_OFI else 0.0
    ofi_medium = compute_ofi(snapshots[-_OFI_WINDOW_MEDIUM:]) if n >= _OFI_WINDOW_MEDIUM else ofi_slow
    ofi_fast = compute_ofi(snapshots[-_OFI_WINDOW_FAST:]) if n >= _OFI_WINDOW_FAST else ofi_medium

    # Flow acceleration: compare fast to slow (normalized by snapshot counts)
    # Normalize per-snapshot to make windows comparable
    fast_per_snap = ofi_fast / max(_OFI_WINDOW_FAST - 1, 1)
    slow_per_snap = ofi_slow / max(n - 1, 1) if n > 1 else 0.0
    flow_acceleration = fast_per_snap - slow_per_snap

    return {
        "ofi_fast": round(ofi_fast, 4),
        "ofi_medium": round(ofi_medium, 4),
        "ofi_slow": round(ofi_slow, 4),
        "flow_acceleration": round(flow_acceleration, 4),
    }


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

    Returns dict with ofi, ofi_zscore, multiscale OFI, and spread_zscore.
    """
    ofi = get_rolling_ofi(ticker)
    record_ofi(ticker, ofi)  # track for z-score (once per state retrieval)
    ofi_z = get_ofi_zscore(ticker)
    sz = get_spread_zscore(ticker)
    ms_ofi = get_multiscale_ofi(ticker)
    _ensure_buffer(ticker)
    return {
        "ofi": ofi,
        "ofi_zscore": ofi_z,
        "ofi_fast": ms_ofi["ofi_fast"],
        "ofi_medium": ms_ofi["ofi_medium"],
        "ofi_slow": ms_ofi["ofi_slow"],
        "flow_acceleration": ms_ofi["flow_acceleration"],
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
