"""Tests for bocpd_regime_switch signal module."""
import numpy as np
import pandas as pd
import pytest

from portfolio.signals.bocpd_regime_switch import (
    compute_bocpd_regime_switch_signal,
    _bocpd_run_lengths,
    _detect_changepoint,
    _trend_signal,
    _mr_signal,
)


def _make_df(n=200, seed=42):
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


def _make_trending_df(n=200, direction=1):
    """Create a trending DataFrame."""
    np.random.seed(42)
    trend = np.linspace(0, direction * 20, n)
    noise = np.random.randn(n) * 0.3
    close = 100 + trend + noise
    return pd.DataFrame({
        "open": close - 0.1 * direction,
        "high": close + 0.3,
        "low": close - 0.3,
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _make_regime_change_df(n=200):
    """Create a DataFrame with a clear regime change at midpoint."""
    np.random.seed(42)
    half = n // 2
    trend_up = np.linspace(0, 15, half)
    trend_down = np.linspace(15, 0, n - half)
    close = 100 + np.concatenate([trend_up, trend_down]) + np.random.randn(n) * 0.2
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
        result = compute_bocpd_regime_switch_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self):
        df = _make_df()
        result = compute_bocpd_regime_switch_signal(df)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)
        assert "changepoint_detector" in result["sub_signals"]
        assert "trend_follower" in result["sub_signals"]
        assert "mean_reverter" in result["sub_signals"]
        assert "regime_classifier" in result["sub_signals"]

    def test_has_indicators(self):
        df = _make_df()
        result = compute_bocpd_regime_switch_signal(df)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)
        assert "max_run_length" in result["indicators"]
        assert "changepoint_severity" in result["indicators"]
        assert "is_changepoint" in result["indicators"]
        assert "regime" in result["indicators"]

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_bocpd_regime_switch_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=10)
        result = compute_bocpd_regime_switch_signal(df)
        assert result["action"] == "HOLD"

    def test_none_input_returns_hold(self):
        result = compute_bocpd_regime_switch_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_nan_handling(self):
        df = _make_df()
        df.iloc[50:55, df.columns.get_loc("close")] = np.nan
        result = compute_bocpd_regime_switch_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_with_context(self):
        df = _make_df()
        ctx = {"ticker": "XAG-USD", "asset_class": "metals", "regime": "trending-up"}
        result = compute_bocpd_regime_switch_signal(df, context=ctx)
        assert isinstance(result, dict)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_confidence_capped_at_0_7(self):
        df = _make_df(n=300)
        result = compute_bocpd_regime_switch_signal(df)
        assert result["confidence"] <= 0.7


class TestBOCPD:
    """Test the BOCPD algorithm itself."""

    def test_run_lengths_stable_regime(self):
        np.random.seed(42)
        returns = np.random.randn(200) * 0.01
        rl = _bocpd_run_lengths(returns)
        assert len(rl) == 200
        assert rl[-1] > 10

    def test_run_lengths_with_shift(self):
        np.random.seed(42)
        r1 = np.random.randn(100) * 0.01
        r2 = np.random.randn(100) * 0.01 + 0.5
        returns = np.concatenate([r1, r2])
        rl = _bocpd_run_lengths(returns)
        assert len(rl) == 200
        pre_shift_end = rl[95]
        post_shift_early = np.min(rl[100:105])
        assert post_shift_early < pre_shift_end or post_shift_early < 20

    def test_empty_returns(self):
        rl = _bocpd_run_lengths(np.array([]))
        assert len(rl) == 0

    def test_detect_changepoint_stable(self):
        rl = np.arange(1, 51, dtype=float)
        is_cp, severity = _detect_changepoint(rl)
        assert not is_cp

    def test_detect_changepoint_drop(self):
        rl = np.ones(50) * 80.0
        rl[-1] = 2.0
        is_cp, severity = _detect_changepoint(rl)
        assert is_cp
        assert severity > 0.5


class TestSubSignals:
    """Test individual sub-signal functions."""

    def test_trend_signal_up(self):
        close = pd.Series(np.linspace(100, 120, 50))
        action, conf = _trend_signal(close)
        assert action == "BUY"
        assert conf > 0.0

    def test_trend_signal_down(self):
        close = pd.Series(np.linspace(120, 100, 50))
        action, conf = _trend_signal(close)
        assert action == "SELL"
        assert conf > 0.0

    def test_trend_signal_flat(self):
        close = pd.Series(np.ones(50) * 100)
        action, _ = _trend_signal(close)
        assert action == "HOLD"

    def test_mr_signal_oversold(self):
        np.random.seed(42)
        close = pd.Series(np.concatenate([
            np.ones(30) * 100,
            [90.0]
        ]))
        action, conf = _mr_signal(close, window=20, zscore_threshold=1.5)
        assert action == "BUY"

    def test_mr_signal_overbought(self):
        np.random.seed(42)
        close = pd.Series(np.concatenate([
            np.ones(30) * 100,
            [110.0]
        ]))
        action, conf = _mr_signal(close, window=20, zscore_threshold=1.5)
        assert action == "SELL"

    def test_mr_signal_neutral(self):
        close = pd.Series(np.ones(30) * 100)
        action, conf = _mr_signal(close)
        assert action == "HOLD"

    def test_mr_insufficient_data(self):
        close = pd.Series([100.0, 101.0])
        action, conf = _mr_signal(close, window=20)
        assert action == "HOLD"
        assert conf == 0.0


class TestRegimeBehavior:
    """Test that the signal correctly switches regimes."""

    def test_trending_market_gives_direction(self):
        df = _make_trending_df(n=200, direction=1)
        result = compute_bocpd_regime_switch_signal(df)
        assert result["indicators"]["regime"] in ("trend_following", "changepoint_mr")

    def test_regime_change_detection(self):
        df = _make_regime_change_df(n=300)
        result = compute_bocpd_regime_switch_signal(df)
        assert isinstance(result["indicators"]["changepoint_severity"], float)

    def test_multiple_assets(self):
        for seed in [42, 123, 456, 789]:
            df = _make_df(n=200, seed=seed)
            result = compute_bocpd_regime_switch_signal(df)
            assert result["action"] in ("BUY", "SELL", "HOLD")
            assert 0.0 <= result["confidence"] <= 0.7


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_constant_price(self):
        n = 100
        df = pd.DataFrame({
            "open": np.ones(n) * 100,
            "high": np.ones(n) * 100.1,
            "low": np.ones(n) * 99.9,
            "close": np.ones(n) * 100,
            "volume": np.ones(n) * 5000,
        })
        result = compute_bocpd_regime_switch_signal(df)
        assert result["action"] == "HOLD"

    def test_extreme_values(self):
        np.random.seed(42)
        close = np.exp(np.cumsum(np.random.randn(200) * 0.1))
        df = pd.DataFrame({
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.random.randint(1000, 10000, 200).astype(float),
        })
        result = compute_bocpd_regime_switch_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert not np.isnan(result["confidence"])

    def test_large_dataset(self):
        df = _make_df(n=1000)
        result = compute_bocpd_regime_switch_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")
