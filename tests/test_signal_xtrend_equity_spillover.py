"""Tests for xtrend_equity_spillover signal module."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from portfolio.signals.xtrend_equity_spillover import (
    compute_xtrend_equity_spillover_signal,
    _compute_rsi,
    _compute_macd_hist,
    _sub_spy_rsi,
    _sub_spy_macd,
    _sub_qqq_rsi,
    _sub_spy_trend,
    _SAFE_HAVEN_TICKERS,
)


def _make_df(n=100):
    """Create a test DataFrame with realistic OHLCV data."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.3),
        "low": close - abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _mock_equity_data(spy_rsi=50.0, spy_macd=0.0, spy_above_ema=True,
                       spy_mom=0.0, qqq_rsi=50.0):
    """Create mock equity data dict."""
    return {
        "SPY": {
            "rsi": spy_rsi,
            "macd_hist": spy_macd,
            "above_ema50": spy_above_ema,
            "mom_5d": spy_mom,
            "close": 500.0,
        },
        "QQQ": {
            "rsi": qqq_rsi,
            "macd_hist": 0.0,
            "above_ema50": True,
            "mom_5d": 0.0,
            "close": 450.0,
        },
    }


class TestSignalInterface:
    """Test that the signal follows the standard interface."""

    @patch("portfolio.signals.xtrend_equity_spillover._fetch_equity_data")
    def test_returns_dict_with_required_keys(self, mock_fetch):
        mock_fetch.return_value = _mock_equity_data()
        df = _make_df()
        result = compute_xtrend_equity_spillover_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    @patch("portfolio.signals.xtrend_equity_spillover._fetch_equity_data")
    def test_has_sub_signals(self, mock_fetch):
        mock_fetch.return_value = _mock_equity_data()
        df = _make_df()
        result = compute_xtrend_equity_spillover_signal(df)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)
        assert "spy_rsi" in result["sub_signals"]
        assert "spy_macd" in result["sub_signals"]
        assert "qqq_rsi" in result["sub_signals"]
        assert "spy_trend" in result["sub_signals"]

    @patch("portfolio.signals.xtrend_equity_spillover._fetch_equity_data")
    def test_has_indicators(self, mock_fetch):
        mock_fetch.return_value = _mock_equity_data()
        df = _make_df()
        result = compute_xtrend_equity_spillover_signal(df)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)
        assert "spy_rsi" in result["indicators"]
        assert "spy_macd_hist" in result["indicators"]
        assert "qqq_rsi" in result["indicators"]

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_xtrend_equity_spillover_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=3)
        result = compute_xtrend_equity_spillover_signal(df)
        assert result["action"] == "HOLD"

    @patch("portfolio.signals.xtrend_equity_spillover._fetch_equity_data")
    def test_none_equity_data_returns_hold(self, mock_fetch):
        mock_fetch.return_value = None
        df = _make_df()
        result = compute_xtrend_equity_spillover_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0


class TestSafeHavenInversion:
    """Test that safe-haven assets get inverted signals."""

    @patch("portfolio.signals.xtrend_equity_spillover._fetch_equity_data")
    def test_bullish_equities_sell_gold(self, mock_fetch):
        """When equities are strongly bullish, gold should get SELL."""
        mock_fetch.return_value = _mock_equity_data(
            spy_rsi=30.0, spy_macd=2.0, spy_above_ema=True,
            spy_mom=0.02, qqq_rsi=30.0,
        )
        df = _make_df()
        result = compute_xtrend_equity_spillover_signal(
            df, context={"ticker": "XAU-USD", "asset_class": "metals"},
        )
        # All sub-signals vote BUY (equity bullish) -> inverted to SELL for gold
        assert result["action"] == "SELL"
        assert result["indicators"]["inverted"] == 1.0

    @patch("portfolio.signals.xtrend_equity_spillover._fetch_equity_data")
    def test_bearish_equities_buy_gold(self, mock_fetch):
        """When equities are strongly bearish, gold should get BUY."""
        mock_fetch.return_value = _mock_equity_data(
            spy_rsi=75.0, spy_macd=-2.0, spy_above_ema=False,
            spy_mom=-0.02, qqq_rsi=75.0,
        )
        df = _make_df()
        result = compute_xtrend_equity_spillover_signal(
            df, context={"ticker": "XAG-USD", "asset_class": "metals"},
        )
        # SELL equity -> inverted to BUY for silver
        assert result["action"] == "BUY"

    @patch("portfolio.signals.xtrend_equity_spillover._fetch_equity_data")
    def test_bullish_equities_buy_btc(self, mock_fetch):
        """When equities are strongly bullish, BTC should get BUY (not inverted)."""
        mock_fetch.return_value = _mock_equity_data(
            spy_rsi=30.0, spy_macd=2.0, spy_above_ema=True,
            spy_mom=0.02, qqq_rsi=30.0,
        )
        df = _make_df()
        result = compute_xtrend_equity_spillover_signal(
            df, context={"ticker": "BTC-USD", "asset_class": "crypto"},
        )
        assert result["action"] == "BUY"
        assert result["indicators"]["inverted"] == 0.0

    @patch("portfolio.signals.xtrend_equity_spillover._fetch_equity_data")
    def test_mstr_not_inverted(self, mock_fetch):
        """MSTR is a risk asset, should NOT be inverted."""
        mock_fetch.return_value = _mock_equity_data(
            spy_rsi=30.0, spy_macd=2.0, spy_above_ema=True,
            spy_mom=0.02, qqq_rsi=30.0,
        )
        df = _make_df()
        result = compute_xtrend_equity_spillover_signal(
            df, context={"ticker": "MSTR", "asset_class": "stocks"},
        )
        assert result["action"] == "BUY"


class TestSubIndicators:
    """Test individual sub-indicator logic."""

    def test_spy_rsi_oversold(self):
        data = _mock_equity_data(spy_rsi=25.0)
        assert _sub_spy_rsi(data) == "BUY"

    def test_spy_rsi_overbought(self):
        data = _mock_equity_data(spy_rsi=75.0)
        assert _sub_spy_rsi(data) == "SELL"

    def test_spy_rsi_neutral(self):
        data = _mock_equity_data(spy_rsi=50.0)
        assert _sub_spy_rsi(data) == "HOLD"

    def test_spy_macd_positive(self):
        data = _mock_equity_data(spy_macd=1.5)
        assert _sub_spy_macd(data) == "BUY"

    def test_spy_macd_negative(self):
        data = _mock_equity_data(spy_macd=-1.5)
        assert _sub_spy_macd(data) == "SELL"

    def test_spy_macd_near_zero(self):
        data = _mock_equity_data(spy_macd=0.2)
        assert _sub_spy_macd(data) == "HOLD"

    def test_qqq_rsi_oversold(self):
        data = _mock_equity_data(qqq_rsi=25.0)
        assert _sub_qqq_rsi(data) == "BUY"

    def test_spy_trend_bullish(self):
        data = _mock_equity_data(spy_above_ema=True, spy_mom=0.02)
        assert _sub_spy_trend(data) == "BUY"

    def test_spy_trend_bearish(self):
        data = _mock_equity_data(spy_above_ema=False, spy_mom=-0.02)
        assert _sub_spy_trend(data) == "SELL"

    def test_spy_trend_ambiguous(self):
        data = _mock_equity_data(spy_above_ema=True, spy_mom=-0.01)
        assert _sub_spy_trend(data) == "HOLD"

    def test_missing_spy_returns_hold(self):
        data = {"QQQ": {"rsi": 50.0}}
        assert _sub_spy_rsi(data) == "HOLD"
        assert _sub_spy_macd(data) == "HOLD"
        assert _sub_spy_trend(data) == "HOLD"


class TestHelpers:
    """Test RSI and MACD computation helpers."""

    def test_compute_rsi_trending_up(self):
        """Strongly trending up series should have high RSI."""
        np.random.seed(42)
        # Positive drift with realistic down days
        changes = np.random.randn(50) * 0.5 + 0.4
        close = pd.Series(100 + np.cumsum(changes))
        val = _compute_rsi(close, 14)
        assert val > 60

    def test_compute_rsi_trending_down(self):
        """Strongly trending down series should have low RSI."""
        close = pd.Series(np.linspace(120, 100, 50))
        val = _compute_rsi(close, 14)
        assert val < 40

    def test_compute_macd_hist_bullish(self):
        """Uptrending series should have positive MACD histogram."""
        close = pd.Series(np.linspace(100, 115, 60))
        val = _compute_macd_hist(close)
        assert val > 0

    def test_compute_macd_hist_bearish(self):
        """Downtrending series should have negative MACD histogram."""
        close = pd.Series(np.linspace(115, 100, 60))
        val = _compute_macd_hist(close)
        assert val < 0


class TestSafeHavenTickers:
    """Verify safe-haven ticker set."""

    def test_gold_is_safe_haven(self):
        assert "XAU-USD" in _SAFE_HAVEN_TICKERS

    def test_silver_is_safe_haven(self):
        assert "XAG-USD" in _SAFE_HAVEN_TICKERS

    def test_btc_not_safe_haven(self):
        assert "BTC-USD" not in _SAFE_HAVEN_TICKERS

    def test_mstr_not_safe_haven(self):
        assert "MSTR" not in _SAFE_HAVEN_TICKERS
