"""Tests for horizon-parameterized indicator computation."""

import numpy as np
import pandas as pd

from portfolio.indicators import compute_indicators


def _make_df(n=100, close_start=100.0):
    """Build minimal OHLCV DataFrame."""
    dates = pd.date_range("2026-01-01", periods=n, freq="h")
    np.random.seed(42)
    closes = close_start + np.cumsum(np.random.randn(n) * 0.5)
    highs = closes + np.random.rand(n) * 2
    lows = closes - np.random.rand(n) * 2
    volumes = np.random.randint(100, 10000, n).astype(float)
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=dates,
    )


class TestComputeIndicatorsDefault:
    """Default (1d) behavior is unchanged."""

    def test_default_rsi_period_14(self):
        df = _make_df(100)
        ind = compute_indicators(df)
        assert ind is not None
        assert "rsi" in ind

    def test_default_macd_12_26_9(self):
        df = _make_df(100)
        ind = compute_indicators(df)
        assert ind is not None
        assert "macd_hist" in ind
        assert "macd_hist_prev" in ind

    def test_no_horizon_same_as_1d(self):
        df = _make_df(100)
        ind_default = compute_indicators(df)
        ind_1d = compute_indicators(df, horizon="1d")
        assert ind_default["rsi"] == ind_1d["rsi"]
        assert ind_default["macd_hist"] == ind_1d["macd_hist"]


class TestComputeIndicators3H:
    """3h horizon uses RSI(7) and MACD(8,17,9)."""

    def test_3h_indicators_returned(self):
        df = _make_df(100)
        ind = compute_indicators(df, horizon="3h")
        assert ind is not None
        assert "rsi" in ind
        assert "macd_hist" in ind

    def test_3h_rsi_differs_from_default(self):
        df = _make_df(100)
        ind_1d = compute_indicators(df)
        ind_3h = compute_indicators(df, horizon="3h")
        assert ind_1d["rsi"] != ind_3h["rsi"]

    def test_3h_macd_differs_from_default(self):
        df = _make_df(100)
        ind_1d = compute_indicators(df)
        ind_3h = compute_indicators(df, horizon="3h")
        assert ind_1d["macd_hist"] != ind_3h["macd_hist"]

    def test_3h_rsi_thresholds(self):
        df = _make_df(100)
        ind_3h = compute_indicators(df, horizon="3h")
        assert "rsi_p20" in ind_3h
        assert "rsi_p80" in ind_3h

    def test_3h_min_rows_lower(self):
        """3h needs only 17 rows min (MACD slow=17), not 26."""
        df = _make_df(20)
        ind_3h = compute_indicators(df, horizon="3h")
        assert ind_3h is not None

    def test_1d_still_needs_26_rows(self):
        """Default still needs 26 rows."""
        df = _make_df(20)
        ind_1d = compute_indicators(df)
        assert ind_1d is None
