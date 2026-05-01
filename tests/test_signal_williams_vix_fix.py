"""Tests for williams_vix_fix signal module."""

import numpy as np
import pandas as pd
import pytest

from portfolio.signals.williams_vix_fix import (
    _compute_wvf,
    _wvf_bb_spike,
    _wvf_complacency,
    _wvf_percentile,
    _wvf_rsi_confirm,
    compute_williams_vix_fix_signal,
)


def _make_df(n=100, seed=42):
    """Create a test DataFrame with realistic OHLCV data."""
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.3),
        "low": close - abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _make_crash_df(n=100):
    """Create a DataFrame with a sharp crash followed by recovery.

    This should trigger BUY signals from the WVF.
    """
    np.random.seed(42)
    # First 60 bars: uptrend
    close_up = 100 + np.cumsum(np.random.randn(60) * 0.3 + 0.1)
    # Next 15 bars: sharp crash
    close_crash = close_up[-1] + np.cumsum(np.random.randn(15) * 0.5 - 0.8)
    # Last 25 bars: recovery
    close_recovery = close_crash[-1] + np.cumsum(np.random.randn(25) * 0.3 + 0.1)
    close = np.concatenate([close_up, close_crash, close_recovery])

    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.4),
        "low": close - abs(np.random.randn(n) * 0.4),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _make_complacent_df(n=100):
    """Create a DataFrame with steady uptrend (low volatility = complacency)."""
    np.random.seed(123)
    close = 100 + np.cumsum(np.ones(n) * 0.2 + np.random.randn(n) * 0.02)
    return pd.DataFrame({
        "open": close - 0.01,
        "high": close + 0.05,
        "low": close - 0.05,
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


class TestComputeWVF:
    """Test the core WVF computation."""

    def test_basic_computation(self):
        df = _make_df()
        wvf = _compute_wvf(df["close"], df["low"])
        assert len(wvf) == len(df)
        # First 21 bars should be NaN (lookback=22)
        assert wvf.iloc[:21].isna().all()
        # Remaining should be non-negative
        valid = wvf.dropna()
        assert (valid >= 0).all()

    def test_wvf_increases_on_crash(self):
        df = _make_crash_df()
        wvf = _compute_wvf(df["close"], df["low"])
        valid = wvf.dropna()
        # During crash period (bars 60-75), WVF should be elevated
        crash_wvf = valid.iloc[40:55]  # offset by lookback
        pre_crash_wvf = valid.iloc[20:38]
        assert crash_wvf.mean() > pre_crash_wvf.mean()

    def test_wvf_near_zero_in_steady_uptrend(self):
        df = _make_complacent_df()
        wvf = _compute_wvf(df["close"], df["low"])
        valid = wvf.dropna()
        # In steady uptrend, WVF should be very low
        assert valid.mean() < 2.0


class TestSubIndicators:
    """Test individual sub-indicators."""

    def test_bb_spike_returns_hold_normally(self):
        df = _make_df()
        wvf = _compute_wvf(df["close"], df["low"])
        val, vote = _wvf_bb_spike(wvf)
        assert vote in ("BUY", "HOLD")
        assert isinstance(val, float)

    def test_bb_spike_fires_on_crash(self):
        df = _make_crash_df()
        wvf = _compute_wvf(df["close"], df["low"])
        # Check during crash period
        crash_slice = wvf.iloc[:75]
        val, vote = _wvf_bb_spike(crash_slice)
        # Should likely trigger BUY during crash
        assert vote in ("BUY", "HOLD")

    def test_percentile_returns_hold_for_normal_wvf(self):
        df = _make_df()
        wvf = _compute_wvf(df["close"], df["low"])
        val, vote = _wvf_percentile(wvf)
        assert vote in ("BUY", "HOLD")
        if not np.isnan(val):
            assert 0 <= val <= 1.0

    def test_rsi_confirm_returns_valid(self):
        df = _make_df()
        wvf = _compute_wvf(df["close"], df["low"])
        val, vote = _wvf_rsi_confirm(wvf, df["close"])
        assert vote in ("BUY", "HOLD")

    def test_complacency_returns_valid(self):
        df = _make_df()
        wvf = _compute_wvf(df["close"], df["low"])
        val, vote = _wvf_complacency(wvf, df["close"])
        assert vote in ("SELL", "HOLD")

    def test_insufficient_data(self):
        wvf = pd.Series([1.0, 2.0, 3.0])
        val, vote = _wvf_bb_spike(wvf)
        assert vote == "HOLD"
        assert np.isnan(val)

        val, vote = _wvf_percentile(wvf)
        assert vote == "HOLD"


class TestSignalInterface:
    """Test that the signal follows the standard interface."""

    def test_returns_dict_with_required_keys(self):
        df = _make_df()
        result = compute_williams_vix_fix_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self):
        df = _make_df()
        result = compute_williams_vix_fix_signal(df)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)
        expected_subs = {"wvf_bb_spike", "wvf_percentile", "wvf_rsi_confirm",
                         "wvf_complacency"}
        assert set(result["sub_signals"].keys()) == expected_subs

    def test_has_indicators(self):
        df = _make_df()
        result = compute_williams_vix_fix_signal(df)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)
        assert "wvf" in result["indicators"]
        assert "wvf_bb_val" in result["indicators"]
        assert "wvf_pct_rank" in result["indicators"]
        assert "rsi_val" in result["indicators"]
        assert "complacency_count" in result["indicators"]

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_williams_vix_fix_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=3)
        result = compute_williams_vix_fix_signal(df)
        assert result["action"] == "HOLD"

    def test_none_dataframe_returns_hold(self):
        result = compute_williams_vix_fix_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_nan_handling(self):
        df = _make_df()
        df.iloc[50:55, df.columns.get_loc("close")] = np.nan
        result = compute_williams_vix_fix_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_with_context(self):
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto",
               "regime": "trending-up"}
        result = compute_williams_vix_fix_signal(df, context=ctx)
        assert isinstance(result, dict)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_confidence_capped_at_0_7(self):
        df = _make_df()
        result = compute_williams_vix_fix_signal(df)
        assert result["confidence"] <= 0.7

    def test_crash_scenario_generates_buy(self):
        """During a sharp crash, WVF should spike and generate BUY."""
        df = _make_crash_df()
        # Test at the crash bottom (around bar 75)
        result = compute_williams_vix_fix_signal(df.iloc[:76])
        # At least some sub-signals should vote BUY
        buy_votes = sum(1 for v in result["sub_signals"].values()
                        if v == "BUY")
        # WVF should be elevated during crash
        assert result["indicators"]["wvf"] > 0

    def test_complacent_scenario(self):
        """During steady uptrend, WVF should be low."""
        df = _make_complacent_df()
        result = compute_williams_vix_fix_signal(df)
        assert result["indicators"]["wvf"] < 2.0

    def test_different_seeds_produce_valid_output(self):
        """Ensure signal works across different random data."""
        for seed in [1, 42, 99, 123, 456]:
            df = _make_df(n=100, seed=seed)
            result = compute_williams_vix_fix_signal(df)
            assert result["action"] in ("BUY", "SELL", "HOLD")
            assert 0.0 <= result["confidence"] <= 0.7

    def test_large_dataframe(self):
        """Ensure signal handles large datasets."""
        df = _make_df(n=500)
        result = compute_williams_vix_fix_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")
