"""Tests for metals_cross_asset signal module.

2026-04-13: Context dict keys were renamed from horizon-specific
(`copper_change_5d`, `spy_change_1d`, `oil_change_5d`,
`gs_ratio_velocity`) to horizon-agnostic (`copper_change_pct`,
`spy_change_pct`, `oil_change_pct`, `gs_velocity_pct`) + a new
`_using_intraday` flag that tells the signal which threshold band to use.
Tests below set `_using_intraday: False` to preserve legacy daily-path
coverage; new tests cover the intraday path with tighter thresholds.
"""
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


def _daily_ctx(**overrides):
    """Build a daily-path context dict with zero defaults. Override as needed."""
    base = {
        "_using_intraday": False,
        "copper_change_pct": 0.0,
        "gvz_zscore": 0.0,
        "gs_ratio_zscore": 0.0,
        "gs_velocity_pct": 0.0,
        "spy_change_pct": 0.0,
        "oil_change_pct": 0.0,
    }
    base.update(overrides)
    return base


def _intraday_ctx(**overrides):
    """Build an intraday-path context dict with zero defaults."""
    base = {
        "_using_intraday": True,
        "copper_change_pct": 0.0,
        "gvz_zscore": 0.0,
        "gs_ratio_zscore": 0.0,
        "gs_velocity_pct": 0.0,
        "spy_change_pct": 0.0,
        "oil_change_pct": 0.0,
    }
    base.update(overrides)
    return base


class TestDailyPath:
    """Legacy daily-threshold path — used when intraday fetch unavailable."""

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_risk_on_environment_bullish_for_silver(self, mock_ctx):
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = _daily_ctx(
            copper_change_pct=2.5, gvz_zscore=-1.5, gs_ratio_zscore=2.0,
            gs_velocity_pct=-3.0, spy_change_pct=1.5, oil_change_pct=3.0,
        )
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        assert result["action"] in ("BUY", "HOLD")
        assert "sub_signals" in result

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_risk_off_bearish_for_silver(self, mock_ctx):
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = _daily_ctx(
            copper_change_pct=-3.0, gvz_zscore=2.5, gs_ratio_zscore=-2.0,
            gs_velocity_pct=3.0, spy_change_pct=-2.0, oil_change_pct=-4.0,
        )
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
    def test_gvz_high_buys_gold(self, mock_ctx):
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = _daily_ctx(gvz_zscore=2.5)
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAU-USD", config={}, macro={}
        )
        assert result["sub_signals"]["gvz"] == "BUY"

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_gvz_high_sells_silver(self, mock_ctx):
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = _daily_ctx(gvz_zscore=2.5)
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        assert result["sub_signals"]["gvz"] == "SELL"

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_gvz_low_sells_gold(self, mock_ctx):
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = _daily_ctx(gvz_zscore=-1.5)
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAU-USD", config={}, macro={}
        )
        assert result["sub_signals"]["gvz"] == "SELL"

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_oil_daily_threshold_2pct(self, mock_ctx):
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = _daily_ctx(oil_change_pct=3.0)
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAU-USD", config={}, macro={}
        )
        assert result["sub_signals"]["oil"] == "BUY"

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_gs_velocity_falling_buys_silver(self, mock_ctx):
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = _daily_ctx(gs_velocity_pct=-3.5)
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        assert result["sub_signals"]["gs_velocity"] == "BUY"

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_gs_velocity_rising_sells_silver(self, mock_ctx):
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = _daily_ctx(gs_velocity_pct=3.5)
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        assert result["sub_signals"]["gs_velocity"] == "SELL"

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_gs_velocity_rising_buys_gold(self, mock_ctx):
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = _daily_ctx(gs_velocity_pct=3.5)
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAU-USD", config={}, macro={}
        )
        assert result["sub_signals"]["gs_velocity"] == "BUY"

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_indicators_expose_intraday_flag(self, mock_ctx):
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = _daily_ctx(gs_velocity_pct=-2.5)
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        assert "gs_velocity" in result["indicators"]
        assert result["indicators"]["gs_velocity"] == -2.5
        assert result["indicators"]["using_intraday"] is False


class TestIntradayPath:
    """Intraday (60m bar) path — used when 3+ of 4 intraday fetches succeed."""

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_copper_threshold_intraday_tighter(self, mock_ctx):
        """Intraday copper threshold 0.4% should fire at +0.5%; daily 1.5% wouldn't."""
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = _intraday_ctx(copper_change_pct=0.5)
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAU-USD", config={}, macro={}
        )
        assert result["sub_signals"]["copper"] == "BUY"

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_copper_below_intraday_threshold_holds(self, mock_ctx):
        """0.3% is below the 0.4% intraday threshold → HOLD."""
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = _intraday_ctx(copper_change_pct=0.3)
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAU-USD", config={}, macro={}
        )
        assert result["sub_signals"]["copper"] == "HOLD"

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_spy_intraday_risk_on_bullish_silver(self, mock_ctx):
        """Intraday SPY 0.3% > 0.25% threshold → risk-on → BUY silver."""
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = _intraday_ctx(spy_change_pct=0.3)
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        assert result["sub_signals"]["spy_risk"] == "BUY"

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_spy_intraday_risk_off_sells_silver(self, mock_ctx):
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = _intraday_ctx(spy_change_pct=-0.35)
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        assert result["sub_signals"]["spy_risk"] == "SELL"

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_oil_intraday_threshold(self, mock_ctx):
        """Oil 0.6% > 0.5% intraday → BUY."""
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = _intraday_ctx(oil_change_pct=0.6)
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        assert result["sub_signals"]["oil"] == "BUY"

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_gs_velocity_intraday_tighter(self, mock_ctx):
        """Intraday G/S velocity -0.6% > 0.5% threshold magnitude → BUY silver."""
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = _intraday_ctx(gs_velocity_pct=-0.6)
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        assert result["sub_signals"]["gs_velocity"] == "BUY"

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_using_intraday_flag_in_indicators(self, mock_ctx):
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = _intraday_ctx(copper_change_pct=0.5)
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAU-USD", config={}, macro={}
        )
        assert result["indicators"]["using_intraday"] is True

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_balanced_vote_distribution_on_intraday(self, mock_ctx):
        """Regression guard against 178-BUY-1-SELL imbalance. With mixed
        intraday inputs the sub-signals should not all push the same way."""
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        # Copper up, SPY down, oil flat, GS velocity mixed — should produce
        # a mix of BUY/SELL/HOLD sub-signals, NOT all BUY.
        mock_ctx.return_value = _intraday_ctx(
            copper_change_pct=0.6,      # BUY
            spy_change_pct=-0.4,        # SELL silver (risk-off)
            oil_change_pct=0.0,         # HOLD
            gs_velocity_pct=0.7,        # SELL silver (gold outperforming)
            gvz_zscore=0.0,
            gs_ratio_zscore=0.0,
        )
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        sub = result["sub_signals"]
        votes = [sub["copper"], sub["spy_risk"], sub["oil"], sub["gs_velocity"]]
        buy_count = votes.count("BUY")
        sell_count = votes.count("SELL")
        assert buy_count > 0 and sell_count > 0, \
            f"votes={votes} — should have both BUY and SELL, regression"


class TestGetCrossAssetContextHealth:
    """Test the data layer's intraday/daily routing."""

    @patch("portfolio.metals_cross_assets.get_gvz")
    @patch("portfolio.metals_cross_assets.get_gold_silver_ratio")
    @patch("portfolio.metals_cross_assets.get_all_cross_asset_data")
    @patch("portfolio.metals_cross_assets.get_all_cross_asset_intraday")
    def test_uses_intraday_when_3_of_4_available(
        self, mock_intraday, mock_daily, mock_gsr_daily, mock_gvz,
    ):
        from portfolio.signals.metals_cross_asset import _get_cross_asset_context
        mock_intraday.return_value = {
            "copper": {"change_3h_pct": 0.5},
            "gold_silver_ratio": {"change_3h_pct": 0.2},
            "spy": {"change_3h_pct": 0.3},
            "oil": None,  # one failure — still 3/4 healthy
        }
        mock_gvz.return_value = {"zscore": 0.0}
        mock_gsr_daily.return_value = {"zscore": 0.5}
        ctx = _get_cross_asset_context("XAG-USD")
        assert ctx is not None
        assert ctx["_using_intraday"] is True
        assert ctx["copper_change_pct"] == 0.5
        mock_daily.assert_not_called()

    @patch("portfolio.metals_cross_assets.get_gold_silver_ratio")
    @patch("portfolio.metals_cross_assets.get_gvz")
    @patch("portfolio.metals_cross_assets.get_all_cross_asset_data")
    @patch("portfolio.metals_cross_assets.get_all_cross_asset_intraday")
    def test_degraded_intraday_source_logs_warning(
        self, mock_intraday, mock_daily, mock_gvz, mock_gsr_daily, caplog,
    ):
        """When 3 of 4 intraday sources are healthy, the missing one should
        be logged at WARNING so operators see the quiet vote loss."""
        import logging as _logging
        from portfolio.signals.metals_cross_asset import _get_cross_asset_context
        mock_intraday.return_value = {
            "copper": {"change_3h_pct": 0.5},
            "gold_silver_ratio": None,  # degraded
            "spy": {"change_3h_pct": 0.3},
            "oil": {"change_3h_pct": 0.2},
        }
        mock_gvz.return_value = {"zscore": 0.0}
        mock_gsr_daily.return_value = {"zscore": 0.5, "change_5d_pct": 0.0}
        with caplog.at_level(_logging.WARNING, logger="portfolio.signals.metals_cross_asset"):
            ctx = _get_cross_asset_context("XAG-USD")
        assert ctx is not None
        assert ctx["_using_intraday"] is True
        # Missing source's field collapsed to 0 (vote will be HOLD)
        assert ctx["gs_velocity_pct"] == 0.0
        # WARNING emitted naming the degraded source
        warnings = [r for r in caplog.records if r.levelno == _logging.WARNING]
        assert any("gold_silver_ratio" in r.getMessage() for r in warnings), \
            f"Expected WARNING naming 'gold_silver_ratio'; got: {[r.getMessage() for r in warnings]}"

    @patch("portfolio.metals_cross_assets.get_gold_silver_ratio")
    @patch("portfolio.metals_cross_assets.get_gvz")
    @patch("portfolio.metals_cross_assets.get_all_cross_asset_data")
    @patch("portfolio.metals_cross_assets.get_all_cross_asset_intraday")
    def test_falls_back_to_daily_when_intraday_mostly_missing(
        self, mock_intraday, mock_daily, mock_gvz, mock_gsr_daily,
    ):
        from portfolio.signals.metals_cross_asset import _get_cross_asset_context
        # Only 1 of 4 intraday sources OK — fall back
        mock_intraday.return_value = {
            "copper": {"change_3h_pct": 0.5},
            "gold_silver_ratio": None,
            "spy": None,
            "oil": None,
        }
        mock_daily.return_value = {
            "copper": {"change_5d_pct": 2.0},
            "gold_silver_ratio": {"change_5d_pct": -1.0, "zscore": 1.0},
            "spy": {"change_1d_pct": 0.5},
            "oil": {"change_5d_pct": 3.0},
        }
        mock_gvz.return_value = {"zscore": 0.0}
        mock_gsr_daily.return_value = {"change_5d_pct": -1.0, "zscore": 1.0}
        ctx = _get_cross_asset_context("XAG-USD")
        assert ctx is not None
        assert ctx["_using_intraday"] is False
        assert ctx["copper_change_pct"] == 2.0  # daily value
        mock_daily.assert_called_once()
        # gs_ratio_zscore still wired through the always-fetched daily source
        assert ctx["gs_ratio_zscore"] == 1.0
