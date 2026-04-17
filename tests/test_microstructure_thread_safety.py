"""Tests for thread-safe buffer access in microstructure_state.py."""

import threading

from unittest.mock import patch

from portfolio import microstructure_state as ms


def _make_depth(bid=100.0, ask=100.1):
    """Create a minimal depth dict for accumulate_snapshot."""
    return {
        "best_bid": bid,
        "best_ask": ask,
        "bids": [[bid, 10]] * 5,
        "asks": [[ask, 10]] * 5,
        "spread": ask - bid,
    }


class TestMicrostructureThreadSafety:
    """Concurrent accumulate_snapshot + get_microstructure_state must not crash."""

    def setup_method(self):
        ms._snapshot_buffers.clear()
        ms._spread_buffers.clear()
        ms._ofi_history.clear()

    @patch("portfolio.microstructure_state.compute_ofi", return_value=0.5)
    @patch("portfolio.microstructure_state.spread_zscore", return_value=0.1)
    def test_concurrent_write_read_no_runtime_error(self, mock_sz, mock_ofi):
        """Two threads hammering buffers for 100 iterations: no RuntimeError."""
        ticker = "XAG-USD"
        errors = []
        iterations = 100

        def writer():
            for i in range(iterations):
                try:
                    ms.accumulate_snapshot(ticker, _make_depth(100 + i * 0.01))
                except Exception as exc:
                    errors.append(("writer", exc))

        def reader():
            for _ in range(iterations):
                try:
                    ms.get_microstructure_state(ticker)
                except Exception as exc:
                    errors.append(("reader", exc))

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
