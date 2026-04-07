"""Tests for microstructure snapshot accumulator."""
from __future__ import annotations

import time

import pytest

from portfolio.microstructure_state import (
    _snapshot_buffers,
    _spread_buffers,
    accumulate_snapshot,
    get_microstructure_state,
    get_rolling_ofi,
    get_spread_zscore,
    snapshot_count,
)


@pytest.fixture(autouse=True)
def _clear_buffers():
    """Clear ring buffers between tests."""
    _snapshot_buffers.clear()
    _spread_buffers.clear()


def _make_depth(best_bid, best_ask, bid_vol=10.0, ask_vol=10.0):
    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "bids": [[best_bid, bid_vol]],
        "asks": [[best_ask, ask_vol]],
        "spread": best_ask - best_bid,
        "ts": int(time.time() * 1000),
    }


class TestAccumulateSnapshot:
    def test_adds_to_buffer(self):
        depth = _make_depth(30.0, 30.5)
        accumulate_snapshot("XAG-USD", depth)
        assert snapshot_count("XAG-USD") == 1

    def test_none_depth_ignored(self):
        accumulate_snapshot("XAG-USD", None)
        assert snapshot_count("XAG-USD") == 0

    def test_multiple_tickers_independent(self):
        accumulate_snapshot("XAG-USD", _make_depth(30.0, 30.5))
        accumulate_snapshot("XAU-USD", _make_depth(3100.0, 3101.0))
        assert snapshot_count("XAG-USD") == 1
        assert snapshot_count("XAU-USD") == 1

    def test_ring_buffer_caps_at_max(self):
        for i in range(100):
            accumulate_snapshot("XAG-USD", _make_depth(30.0 + i * 0.01, 30.5 + i * 0.01))
        assert snapshot_count("XAG-USD") == 60  # _MAX_SNAPSHOTS


class TestGetRollingOFI:
    def test_insufficient_snapshots_returns_zero(self):
        accumulate_snapshot("XAG-USD", _make_depth(30.0, 30.5))
        assert get_rolling_ofi("XAG-USD") == 0.0

    def test_bid_improving_gives_positive_ofi(self):
        # Simulate bid improving over several snapshots
        accumulate_snapshot("XAG-USD", _make_depth(30.0, 30.5, bid_vol=10.0))
        accumulate_snapshot("XAG-USD", _make_depth(30.1, 30.5, bid_vol=8.0))
        accumulate_snapshot("XAG-USD", _make_depth(30.2, 30.5, bid_vol=12.0))
        ofi = get_rolling_ofi("XAG-USD")
        assert ofi > 0  # Bid kept improving → net buying pressure

    def test_ask_improving_gives_negative_ofi(self):
        accumulate_snapshot("XAG-USD", _make_depth(30.0, 30.5, ask_vol=10.0))
        accumulate_snapshot("XAG-USD", _make_depth(30.0, 30.3, ask_vol=8.0))
        accumulate_snapshot("XAG-USD", _make_depth(30.0, 30.1, ask_vol=12.0))
        ofi = get_rolling_ofi("XAG-USD")
        assert ofi < 0  # Ask kept improving → net selling pressure


class TestGetSpreadZScore:
    def test_insufficient_data_returns_none(self):
        for _ in range(5):
            accumulate_snapshot("XAG-USD", _make_depth(30.0, 30.5))
        assert get_spread_zscore("XAG-USD") is None  # need 10

    def test_normal_spread_near_zero(self):
        for _ in range(20):
            accumulate_snapshot("XAG-USD", _make_depth(30.0, 30.5))
        z = get_spread_zscore("XAG-USD")
        assert z is not None
        assert z == pytest.approx(0.0, abs=0.1)

    def test_wide_spread_gives_high_zscore(self):
        for _ in range(19):
            accumulate_snapshot("XAG-USD", _make_depth(30.0, 30.5))  # spread=0.5
        accumulate_snapshot("XAG-USD", _make_depth(30.0, 32.0))  # spread=2.0
        z = get_spread_zscore("XAG-USD")
        assert z is not None
        assert z > 1.5


class TestGetMicrostructureState:
    def test_returns_complete_dict(self):
        for _ in range(15):
            accumulate_snapshot("XAG-USD", _make_depth(30.0, 30.5))
        state = get_microstructure_state("XAG-USD")
        assert "ofi" in state
        assert "spread_zscore" in state
        assert "snapshot_count" in state
        assert state["snapshot_count"] == 15

    def test_empty_ticker_returns_zeros(self):
        state = get_microstructure_state("UNKNOWN")
        assert state["ofi"] == 0.0
        assert state["spread_zscore"] == 0.0
        assert state["snapshot_count"] == 0
