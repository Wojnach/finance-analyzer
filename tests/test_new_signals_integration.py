"""Integration tests for new microstructure + cross-asset signals."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd


def _make_ohlcv(n=100):
    """Generate realistic OHLCV for signal engine."""
    np.random.seed(42)
    close = 30.0 + np.cumsum(np.random.randn(n) * 0.1)
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.05,
            "high": close + abs(np.random.randn(n) * 0.2),
            "low": close - abs(np.random.randn(n) * 0.2),
            "close": close,
            "volume": np.random.randint(100, 10000, n).astype(float),
        },
        index=pd.date_range("2026-01-01", periods=n, freq="15min"),
    )


class TestOrderbookFlowIntegration:
    """Verify orderbook_flow signal loads and runs for metals."""

    @patch("portfolio.signals.orderbook_flow._get_microstructure_context")
    def test_runs_for_silver(self, mock_ctx):
        mock_ctx.return_value = {
            "depth_imbalance": 0.3,
            "trade_imbalance_ratio": 0.2,
            "vpin": 0.4,
            "ofi": 3.0,
            "spread_zscore": 0.5,
            "spread_bps": 5.0,
        }
        from portfolio.signal_registry import get_enhanced_signals, load_signal_func

        entry = get_enhanced_signals()["orderbook_flow"]
        func = load_signal_func(entry)
        result = func(_make_ohlcv(), ticker="XAG-USD", config={}, macro={})
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0
        assert "sub_signals" in result

    def test_holds_for_stocks(self):
        from portfolio.signal_registry import get_enhanced_signals, load_signal_func

        entry = get_enhanced_signals()["orderbook_flow"]
        func = load_signal_func(entry)
        result = func(_make_ohlcv(), ticker="NVDA", config={}, macro={})
        assert result["action"] == "HOLD"


class TestMetalsCrossAssetIntegration:
    """Verify metals_cross_asset signal loads and runs for metals."""

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_runs_for_gold(self, mock_ctx):
        mock_ctx.return_value = {
            "copper_change_5d": 1.0,
            "gvz_zscore": 0.5,
            "gs_ratio_zscore": 0.3,
            # 2026-04-10: added with the G/S velocity sub-signal (c3903bf).
            # 0.5 sits inside the ±_GS_VELOCITY_PCT band so the sub-signal
            # returns HOLD, keeping this test's overall-result assertion
            # (BUY/SELL/HOLD) trivially true.
            "gs_ratio_velocity": 0.5,
            "spy_change_1d": 0.5,
            "oil_change_5d": 1.0,
        }
        from portfolio.signal_registry import get_enhanced_signals, load_signal_func

        entry = get_enhanced_signals()["metals_cross_asset"]
        func = load_signal_func(entry)
        result = func(_make_ohlcv(), ticker="XAU-USD", config={}, macro={})
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert "sub_signals" in result

    def test_holds_for_crypto(self):
        from portfolio.signal_registry import get_enhanced_signals, load_signal_func

        entry = get_enhanced_signals()["metals_cross_asset"]
        func = load_signal_func(entry)
        result = func(_make_ohlcv(), ticker="BTC-USD", config={}, macro={})
        assert result["action"] == "HOLD"
