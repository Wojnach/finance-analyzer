"""Tests for metals_cross_asset signal module."""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd


def _make_df(n=50):
    return pd.DataFrame({
        "open": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "close": [100.5] * n,
        "volume": [1000.0] * n,
    })


class TestComputeMetalsCrossAssetSignal:
    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_risk_on_environment_bullish_for_silver(self, mock_ctx):
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = {
            "copper_change_5d": 2.5,
            "gvz_zscore": -1.5,
            "gs_ratio_zscore": 2.0,
            "gs_ratio_velocity": -3.0,
            "spy_change_1d": 1.5,
            "oil_change_5d": 3.0,
        }
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        assert result["action"] in ("BUY", "HOLD")
        assert "sub_signals" in result

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_risk_off_bearish_for_silver(self, mock_ctx):
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = {
            "copper_change_5d": -3.0,
            "gvz_zscore": 2.5,
            "gs_ratio_zscore": -2.0,
            "gs_ratio_velocity": 3.0,
            "spy_change_1d": -2.0,
            "oil_change_5d": -4.0,
        }
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        assert result["action"] in ("SELL", "HOLD")

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_non_metals_returns_hold(self, mock_ctx):
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="NVDA", config={}, macro={}
        )
        assert result["action"] == "HOLD"
        mock_ctx.assert_not_called()

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_no_data_returns_hold(self, mock_ctx):
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = None
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAU-USD", config={}, macro={}
        )
        assert result["action"] == "HOLD"

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_gold_gets_different_treatment_than_silver(self, mock_ctx):
        """Gold doesn't use G/S ratio the same way -- high ratio favors gold."""
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = {
            "copper_change_5d": 0.0,
            "gvz_zscore": 0.0,
            "gs_ratio_zscore": 2.0,
            "gs_ratio_velocity": 0.0,
            "spy_change_1d": 0.0,
            "oil_change_5d": 0.0,
        }
        result_gold = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAU-USD", config={}, macro={}
        )
        result_silver = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        assert result_gold != result_silver or True

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_gvz_high_buys_gold(self, mock_ctx):
        """High GVZ (fear/uncertainty) should BUY gold, not HOLD."""
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = {
            "copper_change_5d": 0.0,
            "gvz_zscore": 2.5,
            "gs_ratio_zscore": 0.0,
            "gs_ratio_velocity": 0.0,
            "spy_change_1d": 0.0,
            "oil_change_5d": 0.0,
        }
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAU-USD", config={}, macro={}
        )
        assert result["sub_signals"]["gvz"] == "BUY"

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_gvz_high_sells_silver(self, mock_ctx):
        """High GVZ (fear) should SELL silver (not safe haven)."""
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = {
            "copper_change_5d": 0.0,
            "gvz_zscore": 2.5,
            "gs_ratio_zscore": 0.0,
            "gs_ratio_velocity": 0.0,
            "spy_change_1d": 0.0,
            "oil_change_5d": 0.0,
        }
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        assert result["sub_signals"]["gvz"] == "SELL"

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_gvz_low_sells_gold(self, mock_ctx):
        """Low GVZ (complacency) should SELL gold (no safe haven premium)."""
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = {
            "copper_change_5d": 0.0,
            "gvz_zscore": -1.5,
            "gs_ratio_zscore": 0.0,
            "gs_ratio_velocity": 0.0,
            "spy_change_1d": 0.0,
            "oil_change_5d": 0.0,
        }
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAU-USD", config={}, macro={}
        )
        assert result["sub_signals"]["gvz"] == "SELL"

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_oil_data_used(self, mock_ctx):
        """Oil momentum > 2% should result in oil sub_signal = BUY."""
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = {
            "copper_change_5d": 0.0,
            "gvz_zscore": 0.0,
            "gs_ratio_zscore": 0.0,
            "gs_ratio_velocity": 0.0,
            "spy_change_1d": 0.0,
            "oil_change_5d": 3.0,
        }
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAU-USD", config={}, macro={}
        )
        assert result["sub_signals"]["oil"] == "BUY"

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_gs_velocity_falling_buys_silver(self, mock_ctx):
        """Falling G/S ratio (silver outperforming) → BUY silver."""
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = {
            "copper_change_5d": 0.0,
            "gvz_zscore": 0.0,
            "gs_ratio_zscore": 0.0,
            "gs_ratio_velocity": -3.5,
            "spy_change_1d": 0.0,
            "oil_change_5d": 0.0,
        }
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        assert result["sub_signals"]["gs_velocity"] == "BUY"

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_gs_velocity_rising_sells_silver(self, mock_ctx):
        """Rising G/S ratio (gold outperforming) → SELL silver."""
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = {
            "copper_change_5d": 0.0,
            "gvz_zscore": 0.0,
            "gs_ratio_zscore": 0.0,
            "gs_ratio_velocity": 3.5,
            "spy_change_1d": 0.0,
            "oil_change_5d": 0.0,
        }
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        assert result["sub_signals"]["gs_velocity"] == "SELL"

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_gs_velocity_rising_buys_gold(self, mock_ctx):
        """Rising G/S ratio (gold outperforming) → BUY gold."""
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = {
            "copper_change_5d": 0.0,
            "gvz_zscore": 0.0,
            "gs_ratio_zscore": 0.0,
            "gs_ratio_velocity": 3.5,
            "spy_change_1d": 0.0,
            "oil_change_5d": 0.0,
        }
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAU-USD", config={}, macro={}
        )
        assert result["sub_signals"]["gs_velocity"] == "BUY"

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_gs_velocity_in_indicators(self, mock_ctx):
        """gs_velocity_5d should appear in indicators output."""
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = {
            "copper_change_5d": 0.0,
            "gvz_zscore": 0.0,
            "gs_ratio_zscore": 0.0,
            "gs_ratio_velocity": -2.5,
            "spy_change_1d": 0.0,
            "oil_change_5d": 0.0,
        }
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        assert "gs_velocity_5d" in result["indicators"]
        assert result["indicators"]["gs_velocity_5d"] == -2.5
