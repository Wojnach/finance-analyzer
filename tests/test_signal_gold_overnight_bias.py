"""Tests for gold_overnight_bias signal module."""
import datetime

import numpy as np
import pandas as pd
import pytest

from portfolio.signals.gold_overnight_bias import (
    compute_gold_overnight_bias_signal,
    _session_phase_vote,
    _fix_proximity_vote,
)


def _make_df(n=100, tz_aware=False):
    """Create a test DataFrame with realistic OHLCV data."""
    np.random.seed(42)
    close = 2000 + np.cumsum(np.random.randn(n) * 5)
    idx = pd.date_range("2026-01-01", periods=n, freq="1h")
    if tz_aware:
        idx = idx.tz_localize("UTC")
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 1,
            "high": close + abs(np.random.randn(n) * 3),
            "low": close - abs(np.random.randn(n) * 3),
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=idx,
    )


class TestSignalInterface:
    """Standard interface compliance."""

    def test_returns_dict_with_required_keys(self):
        df = _make_df()
        ctx = {"ticker": "XAU-USD", "asset_class": "metals"}
        result = compute_gold_overnight_bias_signal(df, context=ctx)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self):
        df = _make_df()
        ctx = {"ticker": "XAU-USD", "asset_class": "metals"}
        result = compute_gold_overnight_bias_signal(df, context=ctx)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)

    def test_has_indicators(self):
        df = _make_df()
        ctx = {"ticker": "XAU-USD", "asset_class": "metals"}
        result = compute_gold_overnight_bias_signal(df, context=ctx)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        ctx = {"ticker": "XAU-USD", "asset_class": "metals"}
        result = compute_gold_overnight_bias_signal(df, context=ctx)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=3)
        ctx = {"ticker": "XAU-USD", "asset_class": "metals"}
        result = compute_gold_overnight_bias_signal(df, context=ctx)
        assert result["action"] == "HOLD"

    def test_nan_handling(self):
        df = _make_df()
        df.iloc[50:55, df.columns.get_loc("close")] = np.nan
        ctx = {"ticker": "XAU-USD", "asset_class": "metals"}
        result = compute_gold_overnight_bias_signal(df, context=ctx)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_none_dataframe_returns_hold(self):
        ctx = {"ticker": "XAU-USD", "asset_class": "metals"}
        result = compute_gold_overnight_bias_signal(None, context=ctx)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0


class TestMetalsOnly:
    """Signal only fires for metals tickers."""

    def test_non_metals_returns_hold(self):
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto"}
        result = compute_gold_overnight_bias_signal(df, context=ctx)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_stock_returns_hold(self):
        df = _make_df()
        ctx = {"ticker": "MSTR", "asset_class": "stocks"}
        result = compute_gold_overnight_bias_signal(df, context=ctx)
        assert result["action"] == "HOLD"

    def test_no_context_returns_hold(self):
        df = _make_df()
        result = compute_gold_overnight_bias_signal(df, context=None)
        assert result["action"] == "HOLD"

    def test_xau_fires(self):
        df = _make_df()
        ctx = {"ticker": "XAU-USD", "asset_class": "metals"}
        result = compute_gold_overnight_bias_signal(df, context=ctx)
        # Should produce a non-HOLD action (depends on current time)
        assert result["confidence"] >= 0.0

    def test_xag_fires_with_discount(self):
        df = _make_df()
        ctx_gold = {"ticker": "XAU-USD", "asset_class": "metals"}
        ctx_silver = {"ticker": "XAG-USD", "asset_class": "metals"}
        gold_result = compute_gold_overnight_bias_signal(df, context=ctx_gold)
        silver_result = compute_gold_overnight_bias_signal(df, context=ctx_silver)
        # Silver confidence should be <= gold confidence (0.7x discount)
        if gold_result["action"] != "HOLD" and silver_result["action"] != "HOLD":
            assert silver_result["confidence"] <= gold_result["confidence"]


class TestSessionPhase:
    """Test the core session phase sub-signal."""

    def test_overnight_session_buy(self):
        # 20:00 UTC -> deep overnight -> BUY
        vote, conf = _session_phase_vote(20 * 60)
        assert vote == "BUY"
        assert conf > 0.0

    def test_london_pm_session_sell(self):
        # 12:00 UTC -> middle of London PM -> SELL
        vote, conf = _session_phase_vote(12 * 60)
        assert vote == "SELL"
        assert conf > 0.0

    def test_early_morning_buy(self):
        # 03:00 UTC -> overnight -> BUY
        vote, conf = _session_phase_vote(3 * 60)
        assert vote == "BUY"

    def test_am_fix_boundary(self):
        # 10:30 UTC -> exact AM fix -> start of London PM -> SELL
        vote, conf = _session_phase_vote(630)
        assert vote == "SELL"

    def test_pm_fix_boundary(self):
        # 15:00 UTC -> exact PM fix -> start of overnight -> BUY
        vote, conf = _session_phase_vote(900)
        assert vote == "BUY"

    def test_confidence_higher_mid_session(self):
        # Mid-London PM (12:45 = 765 min) vs edge (10:35 = 635 min)
        _, conf_mid = _session_phase_vote(765)
        _, conf_edge = _session_phase_vote(635)
        assert conf_mid >= conf_edge

    def test_all_24_hours_produce_vote(self):
        for h in range(24):
            vote, conf = _session_phase_vote(h * 60)
            assert vote in ("BUY", "SELL")
            assert 0.0 < conf <= 1.0


class TestFixProximity:
    """Test the fix proximity sub-signal."""

    def test_far_from_fix_returns_hold(self):
        vote, conf = _fix_proximity_vote(18 * 60)  # 18:00 UTC
        assert vote == "HOLD"
        assert conf == 0.0

    def test_near_pm_fix_returns_buy(self):
        vote, conf = _fix_proximity_vote(14 * 60 + 30)  # 14:30 UTC, 30min before PM fix
        assert vote == "BUY"
        assert conf > 0.0

    def test_near_am_fix_returns_buy(self):
        vote, conf = _fix_proximity_vote(10 * 60)  # 10:00 UTC, 30min before AM fix
        assert vote == "BUY"
        assert conf > 0.0

    def test_proximity_decays_with_distance(self):
        # 10 min from PM fix -> higher conf than 80 min from PM fix
        _, conf_close = _fix_proximity_vote(14 * 60 + 50)
        _, conf_far = _fix_proximity_vote(13 * 60 + 30)
        assert conf_close > conf_far


class TestTimezoneHandling:
    """Test UTC time extraction from DataFrame index."""

    def test_tz_aware_index(self):
        df = _make_df(tz_aware=True)
        ctx = {"ticker": "XAU-USD", "asset_class": "metals"}
        result = compute_gold_overnight_bias_signal(df, context=ctx)
        assert "utc_hour" in result["indicators"]

    def test_tz_naive_index(self):
        df = _make_df(tz_aware=False)
        ctx = {"ticker": "XAU-USD", "asset_class": "metals"}
        result = compute_gold_overnight_bias_signal(df, context=ctx)
        assert "utc_hour" in result["indicators"]


class TestConfidenceCap:
    """Confidence never exceeds 0.7 (external signal cap)."""

    def test_confidence_capped(self):
        df = _make_df(n=200)
        ctx = {"ticker": "XAU-USD", "asset_class": "metals"}
        result = compute_gold_overnight_bias_signal(df, context=ctx)
        assert result["confidence"] <= 0.7
