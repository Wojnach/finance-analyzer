"""Tests for portfolio.signals.dxy_cross_asset."""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd


def _make_df(n=30):
    return pd.DataFrame({
        "open": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "close": [100.5] * n,
        "volume": [1000.0] * n,
    })


class TestDxyCrossAssetSignal:
    @patch("portfolio.macro_context.get_dxy_intraday")
    def test_dxy_down_buys_silver(self, mock_dxy):
        from portfolio.signals.dxy_cross_asset import compute_dxy_cross_asset_signal
        mock_dxy.return_value = {
            "value": 104.0, "change_1h_pct": -0.25, "change_3h_pct": -0.4,
            "source": "DX-Y.NYB",
        }
        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
        assert result["action"] == "BUY"
        assert result["confidence"] > 0

    @patch("portfolio.macro_context.get_dxy_intraday")
    def test_dxy_up_sells_silver(self, mock_dxy):
        from portfolio.signals.dxy_cross_asset import compute_dxy_cross_asset_signal
        mock_dxy.return_value = {
            "value": 104.0, "change_1h_pct": 0.25, "change_3h_pct": 0.4,
            "source": "DX-Y.NYB",
        }
        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
        assert result["action"] == "SELL"

    @patch("portfolio.macro_context.get_dxy_intraday")
    def test_dxy_down_buys_gold(self, mock_dxy):
        from portfolio.signals.dxy_cross_asset import compute_dxy_cross_asset_signal
        mock_dxy.return_value = {
            "value": 104.0, "change_1h_pct": -0.3, "change_3h_pct": -0.5,
            "source": "DX-Y.NYB",
        }
        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAU-USD")
        assert result["action"] == "BUY"

    @patch("portfolio.macro_context.get_dxy_intraday")
    def test_below_threshold_holds(self, mock_dxy):
        """0.1% is below the 0.15% threshold → HOLD."""
        from portfolio.signals.dxy_cross_asset import compute_dxy_cross_asset_signal
        mock_dxy.return_value = {
            "value": 104.0, "change_1h_pct": -0.1, "change_3h_pct": -0.15,
            "source": "DX-Y.NYB",
        }
        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    @patch("portfolio.macro_context.get_dxy_intraday")
    def test_non_metals_returns_hold_without_fetching(self, mock_dxy):
        from portfolio.signals.dxy_cross_asset import compute_dxy_cross_asset_signal
        result = compute_dxy_cross_asset_signal(_make_df(), ticker="NVDA")
        assert result["action"] == "HOLD"
        mock_dxy.assert_not_called()

    @patch("portfolio.macro_context.get_dxy_intraday")
    def test_non_metals_btc_returns_hold(self, mock_dxy):
        from portfolio.signals.dxy_cross_asset import compute_dxy_cross_asset_signal
        result = compute_dxy_cross_asset_signal(_make_df(), ticker="BTC-USD")
        assert result["action"] == "HOLD"
        mock_dxy.assert_not_called()

    @patch("portfolio.macro_context.get_dxy_intraday")
    def test_missing_dxy_returns_hold(self, mock_dxy):
        from portfolio.signals.dxy_cross_asset import compute_dxy_cross_asset_signal
        mock_dxy.return_value = None
        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAU-USD")
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    @patch("portfolio.macro_context.get_dxy_intraday")
    def test_missing_change_1h_returns_hold(self, mock_dxy):
        """If DXY data exists but 1h field is None, vote HOLD."""
        from portfolio.signals.dxy_cross_asset import compute_dxy_cross_asset_signal
        mock_dxy.return_value = {
            "value": 104.0, "change_1h_pct": None, "change_3h_pct": -0.3,
            "source": "DX-Y.NYB",
        }
        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
        assert result["action"] == "HOLD"

    @patch("portfolio.macro_context.get_dxy_intraday")
    def test_confidence_scales_with_move(self, mock_dxy):
        """Confidence scales linearly with |change_1h| up to 0.5% cap."""
        from portfolio.signals.dxy_cross_asset import compute_dxy_cross_asset_signal

        mock_dxy.return_value = {
            "value": 104.0, "change_1h_pct": -0.25, "change_3h_pct": -0.5,
            "source": "DX-Y.NYB",
        }
        r1 = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")

        mock_dxy.return_value = {
            "value": 104.0, "change_1h_pct": -0.5, "change_3h_pct": -0.8,
            "source": "DX-Y.NYB",
        }
        r2 = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")

        assert r2["confidence"] > r1["confidence"]
        # A 0.5% move saturates at 1.0
        assert r2["confidence"] == 1.0

    @patch("portfolio.macro_context.get_dxy_intraday")
    def test_confidence_caps_at_one(self, mock_dxy):
        """Enormous DXY moves saturate at confidence=1.0."""
        from portfolio.signals.dxy_cross_asset import compute_dxy_cross_asset_signal
        mock_dxy.return_value = {
            "value": 104.0, "change_1h_pct": -2.0, "change_3h_pct": -3.0,
            "source": "DX-Y.NYB",
        }
        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
        assert result["confidence"] == 1.0

    @patch("portfolio.macro_context.get_dxy_intraday")
    def test_indicators_include_source(self, mock_dxy):
        from portfolio.signals.dxy_cross_asset import compute_dxy_cross_asset_signal
        mock_dxy.return_value = {
            "value": 104.0, "change_1h_pct": -0.2, "change_3h_pct": -0.3,
            "source": "EURUSD=X-synth",
        }
        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
        assert result["indicators"]["source"] == "EURUSD=X-synth"

    @patch("portfolio.macro_context.get_dxy_intraday")
    def test_sub_signals_present(self, mock_dxy):
        from portfolio.signals.dxy_cross_asset import compute_dxy_cross_asset_signal
        mock_dxy.return_value = {
            "value": 104.0, "change_1h_pct": -0.25, "change_3h_pct": -0.3,
            "source": "DX-Y.NYB",
        }
        result = compute_dxy_cross_asset_signal(_make_df(), ticker="XAG-USD")
        assert result["sub_signals"]["dxy_1h"] == "BUY"


class TestMacroContextIntradayDxy:
    """Tests for macro_context._fetch_dxy_intraday fallback logic."""

    @patch("yfinance.download")
    def test_primary_source_used_when_available(self, mock_dl):
        import pandas as pd
        from portfolio.macro_context import _fetch_dxy_intraday
        # Generate 5 hours of DX-Y.NYB bars
        idx = pd.date_range("2026-04-13 09:00", periods=5, freq="h")
        df = pd.DataFrame(
            {"Close": [104.0, 104.1, 104.2, 104.3, 104.4]}, index=idx,
        )
        mock_dl.return_value = df
        result = _fetch_dxy_intraday()
        assert result is not None
        assert result["source"] == "DX-Y.NYB"
        # change_1h: (104.4 / 104.3 - 1) * 100 ≈ +0.096%
        assert abs(result["change_1h_pct"] - 0.0959) < 0.01
        # change_3h: (104.4 / 104.1 - 1) * 100 ≈ +0.288%
        assert abs(result["change_3h_pct"] - 0.2882) < 0.01

    @patch("yfinance.download")
    def test_falls_back_to_eurusd_synth_on_empty(self, mock_dl):
        import pandas as pd
        from portfolio.macro_context import _fetch_dxy_intraday

        idx = pd.date_range("2026-04-13 09:00", periods=5, freq="h")
        # First call (DX-Y.NYB) returns empty; second (EURUSD=X) returns data
        empty_df = pd.DataFrame()
        eurusd_df = pd.DataFrame(
            {"Close": [1.170, 1.171, 1.172, 1.173, 1.174]}, index=idx,
        )
        mock_dl.side_effect = [empty_df, eurusd_df]
        result = _fetch_dxy_intraday()
        assert result is not None
        assert result["source"] == "EURUSD=X-synth"
        # Synth: 58 * eurusd^-0.576 — verify direction: eurusd went UP → synth DXY should go DOWN
        assert result["change_1h_pct"] < 0

    @patch("yfinance.download")
    def test_all_sources_fail_returns_none(self, mock_dl):
        import pandas as pd
        from portfolio.macro_context import _fetch_dxy_intraday
        mock_dl.return_value = pd.DataFrame()  # both DX-Y.NYB and EURUSD empty
        result = _fetch_dxy_intraday()
        assert result is None
