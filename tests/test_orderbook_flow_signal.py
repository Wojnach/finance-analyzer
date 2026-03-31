"""Tests for orderbook_flow signal module."""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest


def _make_df(n=50):
    """Minimal OHLCV DataFrame for signal compatibility."""
    return pd.DataFrame({
        "open": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "close": [100.5] * n,
        "volume": [1000.0] * n,
    })


class TestComputeOrderbookFlowSignal:
    @patch("portfolio.signals.orderbook_flow._get_microstructure_context")
    def test_strong_buy_pressure(self, mock_ctx):
        from portfolio.signals.orderbook_flow import compute_orderbook_flow_signal
        mock_ctx.return_value = {
            "depth_imbalance": 1.5,
            "trade_imbalance_ratio": 0.7,
            "vpin": 0.8,
            "ofi": 50.0,
            "spread_zscore": -0.5,
        }
        result = compute_orderbook_flow_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        assert result["action"] == "BUY"
        assert result["confidence"] > 0.5

    @patch("portfolio.signals.orderbook_flow._get_microstructure_context")
    def test_strong_sell_pressure(self, mock_ctx):
        from portfolio.signals.orderbook_flow import compute_orderbook_flow_signal
        mock_ctx.return_value = {
            "depth_imbalance": -1.5,
            "trade_imbalance_ratio": -0.7,
            "vpin": 0.8,
            "ofi": -50.0,
            "spread_zscore": -0.5,
        }
        result = compute_orderbook_flow_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        assert result["action"] == "SELL"
        assert result["confidence"] > 0.5

    @patch("portfolio.signals.orderbook_flow._get_microstructure_context")
    def test_no_data_returns_hold(self, mock_ctx):
        from portfolio.signals.orderbook_flow import compute_orderbook_flow_signal
        mock_ctx.return_value = None
        result = compute_orderbook_flow_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    @patch("portfolio.signals.orderbook_flow._get_microstructure_context")
    def test_non_metals_returns_hold(self, mock_ctx):
        from portfolio.signals.orderbook_flow import compute_orderbook_flow_signal
        result = compute_orderbook_flow_signal(
            _make_df(), ticker="NVDA", config={}, macro={}
        )
        assert result["action"] == "HOLD"
        mock_ctx.assert_not_called()

    @patch("portfolio.signals.orderbook_flow._get_microstructure_context")
    def test_wide_spread_forces_hold(self, mock_ctx):
        from portfolio.signals.orderbook_flow import compute_orderbook_flow_signal
        mock_ctx.return_value = {
            "depth_imbalance": 1.5,
            "trade_imbalance_ratio": 0.7,
            "vpin": 0.3,
            "ofi": 50.0,
            "spread_zscore": 3.0,
        }
        result = compute_orderbook_flow_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        assert result["confidence"] < 0.5
