"""Tests for orderbook_flow signal module."""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd


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

    @patch("portfolio.signals.orderbook_flow._get_microstructure_context")
    def test_trade_throughs_contribute_to_vote(self, mock_ctx):
        from portfolio.signals.orderbook_flow import compute_orderbook_flow_signal
        mock_ctx.return_value = {
            "depth_imbalance": 0.0,
            "trade_imbalance_ratio": 0.0,
            "vpin": 0.3,
            "ofi": 0.0,
            "spread_zscore": 0.0,
            "spread_bps": 5.0,
            "trade_throughs": {
                "buy_throughs": 5,
                "sell_throughs": 0,
                "total_throughs": 5,
                "through_volume": 10.0,
                "max_gap_bps": 15.0,
            },
        }
        result = compute_orderbook_flow_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        assert result["sub_signals"].get("trade_pressure") == "BUY"

    @patch("portfolio.signals.orderbook_flow._get_microstructure_context")
    def test_vpin_toxicity_flag(self, mock_ctx):
        """VPIN > 0.7 should set high_toxicity=True in indicators."""
        from portfolio.signals.orderbook_flow import compute_orderbook_flow_signal
        mock_ctx.return_value = {
            "depth_imbalance": 0.0,
            "trade_imbalance_ratio": 0.0,
            "vpin": 0.85,
            "ofi": 0.0,
            "ofi_zscore": 0.0,
            "spread_zscore": 0.0,
        }
        result = compute_orderbook_flow_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        assert result["indicators"]["high_toxicity"] is True

    @patch("portfolio.signals.orderbook_flow._get_microstructure_context")
    def test_ofi_zscore_used_when_available(self, mock_ctx):
        """When ofi_zscore > 1.5, OFI sub should vote BUY."""
        from portfolio.signals.orderbook_flow import compute_orderbook_flow_signal
        mock_ctx.return_value = {
            "depth_imbalance": 0.0,
            "trade_imbalance_ratio": 0.0,
            "vpin": 0.3,
            "ofi": 2.0,         # below absolute threshold (5.0)
            "ofi_zscore": 2.0,  # above z-score threshold (1.5)
            "spread_zscore": 0.0,
        }
        result = compute_orderbook_flow_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        assert result["sub_signals"]["ofi"] == "BUY"

    @patch("portfolio.signals.orderbook_flow._get_microstructure_context")
    def test_ofi_fallback_to_absolute(self, mock_ctx):
        """When ofi_zscore=0.0 (cold start), should fall back to absolute threshold."""
        from portfolio.signals.orderbook_flow import compute_orderbook_flow_signal
        mock_ctx.return_value = {
            "depth_imbalance": 0.0,
            "trade_imbalance_ratio": 0.0,
            "vpin": 0.3,
            "ofi": 10.0,        # above absolute threshold (5.0)
            "ofi_zscore": 0.0,  # zero → cold start, use absolute
            "spread_zscore": 0.0,
        }
        result = compute_orderbook_flow_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        assert result["sub_signals"]["ofi"] == "BUY"
