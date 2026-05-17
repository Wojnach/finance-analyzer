"""Tests for TSI + Choppiness MR signal module."""
import numpy as np
import pandas as pd
import pytest

from portfolio.signals.tsi_chop_mr import (
    MIN_ROWS,
    _compute_chop,
    _compute_tsi,
    _tsi_divergence,
    _tsi_extreme,
    _tsi_signal_cross,
    compute_tsi_chop_mr_signal,
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


def _make_ranging_df(n=100, seed=42):
    """DataFrame with sideways/ranging price action (high CHOP expected)."""
    np.random.seed(seed)
    close = 100 + np.sin(np.linspace(0, 8 * np.pi, n)) * 2 + np.random.randn(n) * 0.3
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.5),
        "low": close - abs(np.random.randn(n) * 0.5),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _make_trending_df(n=100, seed=42):
    """DataFrame with strong trending price action (low CHOP expected)."""
    np.random.seed(seed)
    close = 100 + np.arange(n) * 0.5 + np.random.randn(n) * 0.1
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.05,
        "high": close + abs(np.random.randn(n) * 0.1),
        "low": close - abs(np.random.randn(n) * 0.1),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


class TestSignalInterface:
    """Test that the signal follows the standard interface."""

    def test_returns_dict_with_required_keys(self):
        df = _make_df()
        result = compute_tsi_chop_mr_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self):
        df = _make_df()
        result = compute_tsi_chop_mr_signal(df)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)

    def test_has_indicators(self):
        df = _make_df()
        result = compute_tsi_chop_mr_signal(df)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)

    def test_indicators_contain_tsi_and_chop(self):
        df = _make_df()
        result = compute_tsi_chop_mr_signal(df)
        assert "tsi" in result["indicators"]
        assert "chop" in result["indicators"]
        assert "regime" in result["indicators"]

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_tsi_chop_mr_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_none_dataframe_returns_hold(self):
        result = compute_tsi_chop_mr_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=10)
        result = compute_tsi_chop_mr_signal(df)
        assert result["action"] == "HOLD"

    def test_nan_handling(self):
        df = _make_df()
        df.iloc[50:55, df.columns.get_loc("close")] = np.nan
        result = compute_tsi_chop_mr_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_with_context(self):
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto", "regime": "trending-up"}
        result = compute_tsi_chop_mr_signal(df, context=ctx)
        assert isinstance(result, dict)

    def test_confidence_capped_at_07(self):
        df = _make_df(n=200)
        result = compute_tsi_chop_mr_signal(df)
        assert result["confidence"] <= 0.7


class TestTSIComputation:
    """Test the TSI indicator."""

    def test_tsi_returns_series(self):
        close = pd.Series(np.random.randn(100).cumsum() + 100)
        tsi = _compute_tsi(close)
        assert isinstance(tsi, pd.Series)
        assert len(tsi) == len(close)

    def test_tsi_range(self):
        close = pd.Series(np.random.randn(200).cumsum() + 100)
        tsi = _compute_tsi(close)
        valid = tsi.dropna()
        assert (valid >= -100).all()
        assert (valid <= 100).all()

    def test_tsi_nan_at_start(self):
        close = pd.Series(np.random.randn(50).cumsum() + 100)
        tsi = _compute_tsi(close)
        assert len(tsi) == len(close)
        assert isinstance(tsi.iloc[-1], (float, np.floating))


class TestCHOPComputation:
    """Test the Choppiness Index."""

    def test_chop_returns_series(self):
        df = _make_df()
        chop = _compute_chop(df["high"], df["low"], df["close"])
        assert isinstance(chop, pd.Series)

    def test_chop_range(self):
        df = _make_df(n=200)
        chop = _compute_chop(df["high"], df["low"], df["close"])
        valid = chop.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_ranging_higher_chop_than_trending(self):
        df_range = _make_ranging_df(n=200)
        df_trend = _make_trending_df(n=200)
        chop_range = _compute_chop(df_range["high"], df_range["low"], df_range["close"]).iloc[-1]
        chop_trend = _compute_chop(df_trend["high"], df_trend["low"], df_trend["close"]).iloc[-1]
        assert chop_range > chop_trend, (
            f"Ranging CHOP ({chop_range:.1f}) should exceed trending CHOP ({chop_trend:.1f})"
        )

    def test_trending_market_low_chop(self):
        df = _make_trending_df(n=200)
        chop = _compute_chop(df["high"], df["low"], df["close"])
        last_chop = chop.iloc[-1]
        assert last_chop < 60, f"Expected low CHOP for trending data, got {last_chop}"


class TestSubIndicators:
    """Test individual sub-indicator functions."""

    def test_tsi_extreme_buy(self):
        assert _tsi_extreme(-35.0) == "BUY"

    def test_tsi_extreme_sell(self):
        assert _tsi_extreme(35.0) == "SELL"

    def test_tsi_extreme_hold(self):
        assert _tsi_extreme(0.0) == "HOLD"
        assert _tsi_extreme(20.0) == "HOLD"
        assert _tsi_extreme(-20.0) == "HOLD"

    def test_tsi_extreme_nan(self):
        assert _tsi_extreme(float("nan")) == "HOLD"

    def test_tsi_signal_cross(self):
        tsi = pd.Series(np.linspace(-30, 30, 50))
        result = _tsi_signal_cross(tsi)
        assert result in ("BUY", "SELL", "HOLD")

    def test_tsi_signal_cross_short_series(self):
        tsi = pd.Series([1.0, 2.0])
        assert _tsi_signal_cross(tsi) == "HOLD"

    def test_tsi_divergence(self):
        close = pd.Series(np.linspace(100, 90, 20))
        tsi = pd.Series(np.linspace(-20, -10, 20))
        result = _tsi_divergence(close, tsi)
        assert result in ("BUY", "SELL", "HOLD")


class TestRegimeGating:
    """Test that CHOP regime gating works correctly."""

    def test_trending_market_returns_hold(self):
        df = _make_trending_df(n=200)
        result = compute_tsi_chop_mr_signal(df)
        if result["indicators"].get("regime") == "trending":
            assert result["action"] == "HOLD"
            assert result["confidence"] == 0.0

    def test_sub_signals_populated_in_all_regimes(self):
        for make_fn in [_make_df, _make_ranging_df, _make_trending_df]:
            df = make_fn(n=200)
            result = compute_tsi_chop_mr_signal(df)
            assert "chop_regime" in result["sub_signals"]
