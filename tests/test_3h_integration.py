"""Integration test: 3h signal optimizations work end-to-end."""

from unittest import mock

import numpy as np
import pandas as pd
import pytest


def _make_df(n=100, close_start=100.0):
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


class TestGenerate3HSignal:
    """Test generate_signal with horizon='3h'."""

    @mock.patch("portfolio.market_timing.should_skip_gpu", return_value=True)
    @mock.patch("portfolio.shared_state._cached", return_value=None)
    def test_3h_returns_valid_signal(self, mock_cached, mock_skip):
        from portfolio.indicators import compute_indicators
        from portfolio.signal_engine import generate_signal

        df = _make_df(100)
        ind = compute_indicators(df, horizon="3h")
        assert ind is not None

        action, conf, extra = generate_signal(
            ind, ticker="BTC-USD", df=df, horizon="3h"
        )
        assert action in ("BUY", "SELL", "HOLD")
        assert 0.0 <= conf <= 0.75  # confidence cap

    @mock.patch("portfolio.market_timing.should_skip_gpu", return_value=True)
    @mock.patch("portfolio.shared_state._cached", return_value=None)
    def test_3h_stores_horizon_in_extra(self, mock_cached, mock_skip):
        from portfolio.indicators import compute_indicators
        from portfolio.signal_engine import generate_signal

        df = _make_df(100)
        ind = compute_indicators(df, horizon="3h")
        _, _, extra = generate_signal(ind, ticker="BTC-USD", df=df, horizon="3h")
        assert extra.get("_horizon") == "3h"

    @mock.patch("portfolio.market_timing.should_skip_gpu", return_value=True)
    @mock.patch("portfolio.shared_state._cached", return_value=None)
    def test_1d_not_capped(self, mock_cached, mock_skip):
        from portfolio.indicators import compute_indicators
        from portfolio.signal_engine import generate_signal

        df = _make_df(100)
        ind = compute_indicators(df)
        action, conf, extra = generate_signal(ind, ticker="BTC-USD", df=df)
        # 1d path: no horizon stored, no 0.75 cap enforced
        assert "_horizon" not in extra
        # conf can be > 0.75 in 1d mode (if signal is strong enough)
