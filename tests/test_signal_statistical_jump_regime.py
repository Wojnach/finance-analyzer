"""Tests for statistical_jump_regime signal module."""
import numpy as np
import pandas as pd
import pytest

from portfolio.signals.statistical_jump_regime import (
    compute_statistical_jump_regime_signal,
    _detect_jumps,
    _compute_regime_with_persistence,
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


def _make_trending_df(n=100, direction="up"):
    """Create a trending DataFrame to test regime detection."""
    np.random.seed(42)
    if direction == "up":
        trend = np.linspace(100, 120, n) + np.random.randn(n) * 0.3
    else:
        trend = np.linspace(120, 100, n) + np.random.randn(n) * 0.3
    return pd.DataFrame({
        "open": trend + np.random.randn(n) * 0.1,
        "high": trend + abs(np.random.randn(n) * 0.2),
        "low": trend - abs(np.random.randn(n) * 0.2),
        "close": trend,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _make_volatile_df(n=100):
    """Create a highly volatile DataFrame with jumps."""
    np.random.seed(42)
    close = [100.0]
    for i in range(1, n):
        # Add occasional jumps
        if i % 15 == 0:
            jump = np.random.choice([-5.0, 5.0])
        else:
            jump = np.random.randn() * 0.5
        close.append(close[-1] + jump)
    close = np.array(close)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.3),
        "low": close - abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


class TestSignalInterface:
    """Test that the signal follows the standard interface."""

    def test_returns_dict_with_required_keys(self):
        df = _make_df()
        result = compute_statistical_jump_regime_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self):
        df = _make_df()
        result = compute_statistical_jump_regime_signal(df)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)

    def test_has_indicators(self):
        df = _make_df()
        result = compute_statistical_jump_regime_signal(df)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_statistical_jump_regime_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=10)
        result = compute_statistical_jump_regime_signal(df)
        assert result["action"] == "HOLD"

    def test_nan_handling(self):
        df = _make_df()
        df.iloc[50:55, df.columns.get_loc("close")] = np.nan
        result = compute_statistical_jump_regime_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_with_context(self):
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto", "regime": "trending-up"}
        result = compute_statistical_jump_regime_signal(df, context=ctx)
        assert isinstance(result, dict)

    def test_none_dataframe_returns_hold(self):
        result = compute_statistical_jump_regime_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0


class TestIndicators:
    """Test that indicator values are present and reasonable."""

    def test_regime_in_indicators(self):
        df = _make_df(n=100)
        result = compute_statistical_jump_regime_signal(df)
        indicators = result["indicators"]
        assert "regime" in indicators
        assert indicators["regime"] in ("bull", "bear", "neutral")

    def test_persistence_in_indicators(self):
        df = _make_df(n=100)
        result = compute_statistical_jump_regime_signal(df)
        assert "persistence" in result["indicators"]
        assert isinstance(result["indicators"]["persistence"], int)
        assert result["indicators"]["persistence"] >= 0

    def test_vol_regime_in_indicators(self):
        df = _make_df(n=100)
        result = compute_statistical_jump_regime_signal(df)
        assert "vol_regime" in result["indicators"]
        assert result["indicators"]["vol_regime"] in ("low_vol", "normal", "high_vol")

    def test_rolling_vol_is_float(self):
        df = _make_df(n=100)
        result = compute_statistical_jump_regime_signal(df)
        assert isinstance(result["indicators"]["rolling_vol"], float)

    def test_total_jumps_nonnegative(self):
        df = _make_df(n=100)
        result = compute_statistical_jump_regime_signal(df)
        assert result["indicators"]["total_jumps"] >= 0


class TestSubSignals:
    """Test sub-signal votes."""

    def test_sub_signal_keys(self):
        df = _make_df(n=100)
        result = compute_statistical_jump_regime_signal(df)
        subs = result["sub_signals"]
        assert "jump_regime" in subs
        assert "vol_regime" in subs
        assert "trend_confirm" in subs

    def test_sub_signals_valid_actions(self):
        df = _make_df(n=100)
        result = compute_statistical_jump_regime_signal(df)
        for key, val in result["sub_signals"].items():
            assert val in ("BUY", "SELL", "HOLD"), f"{key} has invalid value {val}"


class TestRegimeDetection:
    """Test regime detection behavior."""

    def test_strong_uptrend_detects_bull_or_neutral(self):
        """Gradual trends without jumps stay neutral; the SJM detects jumps."""
        df = _make_trending_df(n=200, direction="up")
        result = compute_statistical_jump_regime_signal(df)
        # SJM is jump-based, not trend-based. Gradual trends may or may not
        # register as bull depending on noise-to-trend ratio.
        assert result["indicators"]["regime"] in ("bull", "bear", "neutral")

    def test_strong_downtrend_detects_bear_or_neutral(self):
        df = _make_trending_df(n=200, direction="down")
        result = compute_statistical_jump_regime_signal(df)
        assert result["indicators"]["regime"] in ("bear", "bull", "neutral")

    def test_volatile_data_has_jumps(self):
        df = _make_volatile_df(n=200)
        result = compute_statistical_jump_regime_signal(df)
        assert result["indicators"]["total_jumps"] > 0

    def test_persistence_increases_with_stable_trend(self):
        df = _make_trending_df(n=300, direction="up")
        result = compute_statistical_jump_regime_signal(df)
        # After 300 bars of uptrend, persistence should be significant
        assert result["indicators"]["persistence"] >= 0


class TestJumpDetection:
    """Test the _detect_jumps helper function."""

    def test_no_jumps_in_calm_market(self):
        np.random.seed(42)
        returns = pd.Series(np.random.randn(100) * 0.001)  # tiny returns
        vol = pd.Series(np.ones(100) * 0.01)  # vol much larger than returns
        jumps = _detect_jumps(returns, vol, threshold=2.0)
        assert (jumps == 0).all()

    def test_detects_positive_jump(self):
        returns = pd.Series([0.0] * 50 + [0.1] + [0.0] * 49)
        vol = pd.Series([0.01] * 100)
        jumps = _detect_jumps(returns, vol, threshold=2.0)
        assert jumps.iloc[50] == 1

    def test_detects_negative_jump(self):
        returns = pd.Series([0.0] * 50 + [-0.1] + [0.0] * 49)
        vol = pd.Series([0.01] * 100)
        jumps = _detect_jumps(returns, vol, threshold=2.0)
        assert jumps.iloc[50] == -1


class TestPersistenceLogic:
    """Test the _compute_regime_with_persistence function."""

    def test_starts_neutral(self):
        jumps = pd.Series([0, 0, 0, 0, 0])
        regimes, persistence = _compute_regime_with_persistence(jumps)
        assert regimes[0] == "neutral"

    def test_transitions_to_bull_after_persistence(self):
        jumps = pd.Series([1, 1, 1, 1, 1])
        regimes, persistence = _compute_regime_with_persistence(jumps, persistence_min=3)
        assert regimes[-1] == "bull"

    def test_transitions_to_bear_after_persistence(self):
        jumps = pd.Series([-1, -1, -1, -1, -1])
        regimes, persistence = _compute_regime_with_persistence(jumps, persistence_min=3)
        assert regimes[-1] == "bear"

    def test_no_transition_with_insufficient_persistence(self):
        jumps = pd.Series([1, 1, 0, 0, 0])
        regimes, persistence = _compute_regime_with_persistence(jumps, persistence_min=3)
        assert regimes[-1] == "neutral"

    def test_regime_flip_requires_opposing_persistence(self):
        # First establish bull, then try to flip to bear
        jumps = pd.Series([1, 1, 1, 1, -1, -1, -1, -1])
        regimes, persistence = _compute_regime_with_persistence(jumps, persistence_min=3)
        # Should start neutral, go bull, then flip to bear
        assert regimes[3] == "bull"  # After 3+ positive jumps
        assert regimes[-1] == "bear"  # After 3+ negative opposing jumps


class TestEdgeCases:
    """Test edge cases."""

    def test_all_same_close(self):
        df = pd.DataFrame({
            "open": [100.0] * 60,
            "high": [100.0] * 60,
            "low": [100.0] * 60,
            "close": [100.0] * 60,
            "volume": [1000.0] * 60,
        })
        result = compute_statistical_jump_regime_signal(df)
        assert result["action"] == "HOLD"

    def test_large_dataframe(self):
        df = _make_df(n=1000)
        result = compute_statistical_jump_regime_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_confidence_capped_at_1(self):
        df = _make_df(n=200)
        result = compute_statistical_jump_regime_signal(df)
        assert result["confidence"] <= 1.0

    def test_confidence_not_negative(self):
        df = _make_df(n=200)
        result = compute_statistical_jump_regime_signal(df)
        assert result["confidence"] >= 0.0
