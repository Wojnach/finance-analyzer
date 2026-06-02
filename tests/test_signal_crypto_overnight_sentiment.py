"""Tests for crypto_overnight_sentiment signal module."""
import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from portfolio.signals.crypto_overnight_sentiment import (
    compute_crypto_overnight_sentiment_signal,
    _compute_overnight_returns,
    _is_us_dst,
)


def _make_df(n=100):
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.3),
        "low": close - abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _make_hourly_df(n_days=40):
    """Create realistic hourly OHLCV data with UTC timestamps."""
    np.random.seed(42)
    n = n_days * 24
    base = datetime.datetime(2026, 5, 1, tzinfo=datetime.timezone.utc)
    idx = pd.DatetimeIndex([base + datetime.timedelta(hours=i) for i in range(n)])
    close = 50000 + np.cumsum(np.random.randn(n) * 50)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 10,
        "high": close + abs(np.random.randn(n) * 30),
        "low": close - abs(np.random.randn(n) * 30),
        "close": close,
        "volume": np.random.randint(100, 1000, n).astype(float),
    }, index=idx)


class TestSignalInterface:
    def test_returns_dict_with_required_keys(self):
        df = _make_df()
        hourly = _make_hourly_df()
        with patch(
            "portfolio.signals.crypto_overnight_sentiment._fetch_hourly_data",
            return_value=hourly,
        ):
            result = compute_crypto_overnight_sentiment_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self):
        df = _make_df()
        hourly = _make_hourly_df()
        with patch(
            "portfolio.signals.crypto_overnight_sentiment._fetch_hourly_data",
            return_value=hourly,
        ):
            result = compute_crypto_overnight_sentiment_signal(df)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)

    def test_has_indicators(self):
        df = _make_df()
        hourly = _make_hourly_df()
        with patch(
            "portfolio.signals.crypto_overnight_sentiment._fetch_hourly_data",
            return_value=hourly,
        ):
            result = compute_crypto_overnight_sentiment_signal(df)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_crypto_overnight_sentiment_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=3)
        result = compute_crypto_overnight_sentiment_signal(df)
        assert result["action"] == "HOLD"

    def test_nan_handling(self):
        df = _make_df()
        df.iloc[50:55, df.columns.get_loc("close")] = np.nan
        hourly = _make_hourly_df()
        with patch(
            "portfolio.signals.crypto_overnight_sentiment._fetch_hourly_data",
            return_value=hourly,
        ):
            result = compute_crypto_overnight_sentiment_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_no_hourly_data_returns_hold(self):
        df = _make_df()
        with patch(
            "portfolio.signals.crypto_overnight_sentiment._fetch_hourly_data",
            return_value=None,
        ):
            result = compute_crypto_overnight_sentiment_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0


class TestContext:
    def test_with_crypto_context(self):
        df = _make_df()
        hourly = _make_hourly_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto", "regime": "trending-up"}
        with patch(
            "portfolio.signals.crypto_overnight_sentiment._fetch_hourly_data",
            return_value=hourly,
        ):
            result = compute_crypto_overnight_sentiment_signal(df, context=ctx)
        assert isinstance(result, dict)

    def test_safe_haven_inverts_direction(self):
        """Safe-haven tickers should invert the signal direction."""
        df = _make_df()
        hourly = _make_hourly_df()
        crypto_ctx = {"ticker": "BTC-USD", "asset_class": "crypto"}
        metals_ctx = {"ticker": "XAG-USD", "asset_class": "metals"}
        with patch(
            "portfolio.signals.crypto_overnight_sentiment._fetch_hourly_data",
            return_value=hourly,
        ):
            crypto_result = compute_crypto_overnight_sentiment_signal(df, context=crypto_ctx)
            metals_result = compute_crypto_overnight_sentiment_signal(df, context=metals_ctx)

        if crypto_result["action"] != "HOLD" and metals_result["action"] != "HOLD":
            assert crypto_result["action"] != metals_result["action"]

    def test_confidence_capped_at_07(self):
        df = _make_df()
        hourly = _make_hourly_df()
        with patch(
            "portfolio.signals.crypto_overnight_sentiment._fetch_hourly_data",
            return_value=hourly,
        ):
            result = compute_crypto_overnight_sentiment_signal(df)
        assert result["confidence"] <= 0.7


class TestDST:
    def test_summer_dst(self):
        dt = datetime.datetime(2026, 7, 15, tzinfo=datetime.timezone.utc)
        assert _is_us_dst(dt) is True

    def test_winter_no_dst(self):
        dt = datetime.datetime(2026, 1, 15, tzinfo=datetime.timezone.utc)
        assert _is_us_dst(dt) is False

    def test_march_transition(self):
        dt_before = datetime.datetime(2026, 3, 7, tzinfo=datetime.timezone.utc)
        dt_after = datetime.datetime(2026, 3, 9, tzinfo=datetime.timezone.utc)
        assert _is_us_dst(dt_before) is False
        assert _is_us_dst(dt_after) is True


class TestOvernightReturns:
    def test_computes_returns_from_hourly(self):
        hourly = _make_hourly_df(n_days=10)
        returns = _compute_overnight_returns(hourly, close_hour=20, open_hour=14)
        assert isinstance(returns, list)
        assert len(returns) > 0
        for r in returns:
            assert isinstance(r, float)

    def test_empty_df_returns_empty(self):
        returns = _compute_overnight_returns(None, 20, 14)
        assert returns == []

    def test_short_df_returns_empty(self):
        hourly = _make_hourly_df(n_days=1)
        returns = _compute_overnight_returns(hourly[:5], 20, 14)
        assert returns == []
