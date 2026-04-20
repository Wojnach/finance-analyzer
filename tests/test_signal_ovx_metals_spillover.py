"""Tests for ovx_metals_spillover signal module."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from portfolio.signals.ovx_metals_spillover import (
    compute_ovx_metals_spillover_signal,
    _percentile_rank,
    _ovx_level_signal,
    _ovx_momentum_signal,
    _ovx_zscore_signal,
    _ovx_reversion_signal,
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


def _make_ovx_data(current=25.0, n=300, trend="flat"):
    """Create mock OVX data dict."""
    np.random.seed(42)
    if trend == "high":
        base = np.linspace(20, 40, n) + np.random.randn(n) * 2
    elif trend == "low":
        base = np.linspace(30, 15, n) + np.random.randn(n) * 1.5
    elif trend == "spike":
        base = np.ones(n) * 22 + np.random.randn(n) * 1.5
        base[-10:] = np.linspace(22, 45, 10)
    else:
        base = np.ones(n) * current + np.random.randn(n) * 2

    base = np.maximum(base, 5)
    base[-1] = current
    return {"current": current, "series": base.tolist()}


class TestSubSignals:
    """Test individual sub-signal functions."""

    def test_percentile_rank_high(self):
        series = list(range(1, 101))  # 1..100
        pctile = _percentile_rank(series, 252)
        assert pctile > 95  # 100 is at the top

    def test_percentile_rank_low(self):
        series = list(range(1, 101))
        series[-1] = 1  # Set current to lowest
        pctile = _percentile_rank(series, 252)
        assert pctile < 5

    def test_percentile_rank_middle(self):
        series = list(range(1, 101))
        series[-1] = 50
        pctile = _percentile_rank(series, 252)
        assert 40 < pctile < 60

    def test_ovx_level_high(self):
        assert _ovx_level_signal(85) == "SELL"

    def test_ovx_level_low(self):
        assert _ovx_level_signal(15) == "BUY"

    def test_ovx_level_neutral(self):
        assert _ovx_level_signal(50) == "HOLD"

    def test_ovx_momentum_rising(self):
        series = [20.0] * 50 + [30.0]  # 50% rise
        val, vote = _ovx_momentum_signal(series)
        assert val > 10
        assert vote == "SELL"

    def test_ovx_momentum_falling(self):
        series = [30.0] * 50 + [20.0]  # 33% drop
        val, vote = _ovx_momentum_signal(series)
        assert val < -10
        assert vote == "BUY"

    def test_ovx_momentum_flat(self):
        series = [25.0] * 50
        val, vote = _ovx_momentum_signal(series)
        assert vote == "HOLD"

    def test_ovx_momentum_insufficient_data(self):
        series = [25.0, 26.0]
        val, vote = _ovx_momentum_signal(series)
        assert vote == "HOLD"

    def test_ovx_zscore_high(self):
        series = [25.0] * 20
        series[-1] = 40.0  # spike
        val, vote = _ovx_zscore_signal(series)
        assert val > 1.5
        assert vote == "SELL"

    def test_ovx_zscore_low(self):
        series = [25.0] * 20
        series[-1] = 10.0  # drop
        val, vote = _ovx_zscore_signal(series)
        assert val < -1.5
        assert vote == "BUY"

    def test_ovx_zscore_normal(self):
        series = [25.0] * 20
        val, vote = _ovx_zscore_signal(series)
        assert vote == "HOLD"

    def test_ovx_reversion_falling_from_high(self):
        # High percentile but falling = BUY (mean reversion)
        series = [30.0] * 50 + [25.0]  # falling
        vote = _ovx_reversion_signal(85.0, series)
        assert vote == "BUY"

    def test_ovx_reversion_rising_from_low(self):
        # Low percentile but rising = SELL
        series = [15.0] * 50 + [20.0]  # rising
        vote = _ovx_reversion_signal(15.0, series)
        assert vote == "SELL"

    def test_ovx_reversion_no_signal(self):
        series = [25.0] * 50
        vote = _ovx_reversion_signal(50.0, series)
        assert vote == "HOLD"


class TestSignalInterface:
    """Test that the signal follows the standard interface."""

    @patch("portfolio.signals.ovx_metals_spillover._cached")
    def test_returns_dict_with_required_keys(self, mock_cached):
        mock_cached.return_value = _make_ovx_data(current=25.0, n=300)
        df = _make_df()
        result = compute_ovx_metals_spillover_signal(
            df, context={"ticker": "XAU-USD"}
        )
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    @patch("portfolio.signals.ovx_metals_spillover._cached")
    def test_has_sub_signals(self, mock_cached):
        mock_cached.return_value = _make_ovx_data(current=25.0, n=300)
        result = compute_ovx_metals_spillover_signal(
            None, context={"ticker": "XAG-USD"}
        )
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)
        if result["action"] != "HOLD" or result["confidence"] > 0:
            assert "ovx_level" in result["sub_signals"]

    @patch("portfolio.signals.ovx_metals_spillover._cached")
    def test_has_indicators(self, mock_cached):
        mock_cached.return_value = _make_ovx_data(current=25.0, n=300)
        result = compute_ovx_metals_spillover_signal(
            None, context={"ticker": "XAU-USD"}
        )
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)

    def test_non_metals_returns_hold(self):
        result = compute_ovx_metals_spillover_signal(
            None, context={"ticker": "BTC-USD"}
        )
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_no_context_returns_hold(self):
        result = compute_ovx_metals_spillover_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    @patch("portfolio.signals.ovx_metals_spillover._cached")
    def test_none_data_returns_hold(self, mock_cached):
        mock_cached.return_value = None
        result = compute_ovx_metals_spillover_signal(
            None, context={"ticker": "XAU-USD"}
        )
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    @patch("portfolio.signals.ovx_metals_spillover._cached")
    def test_confidence_capped_at_0_7(self, mock_cached):
        mock_cached.return_value = _make_ovx_data(current=50.0, n=300, trend="spike")
        result = compute_ovx_metals_spillover_signal(
            None, context={"ticker": "XAU-USD"}
        )
        assert result["confidence"] <= 0.7

    @patch("portfolio.signals.ovx_metals_spillover._cached")
    def test_high_ovx_spike_sells(self, mock_cached):
        """OVX spiking to extreme should generate SELL for metals."""
        mock_cached.return_value = _make_ovx_data(current=50.0, n=300, trend="spike")
        result = compute_ovx_metals_spillover_signal(
            None, context={"ticker": "XAG-USD"}
        )
        # With a spike: level=SELL, momentum=SELL, zscore=SELL, reversion=HOLD
        assert result["action"] == "SELL"

    @patch("portfolio.signals.ovx_metals_spillover._cached")
    def test_low_falling_ovx_buys(self, mock_cached):
        """Low and falling OVX should generate BUY for metals."""
        mock_cached.return_value = _make_ovx_data(current=12.0, n=300, trend="low")
        result = compute_ovx_metals_spillover_signal(
            None, context={"ticker": "XAU-USD"}
        )
        # Low OVX: level=BUY, momentum=BUY (falling), zscore=BUY, reversion=HOLD
        assert result["action"] == "BUY"

    def test_with_kwargs_ticker(self):
        """Test ticker passed via kwargs instead of context."""
        result = compute_ovx_metals_spillover_signal(
            None, context={}, ticker="MSTR"
        )
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    @patch("portfolio.signals.ovx_metals_spillover._cached")
    def test_xag_same_as_xau(self, mock_cached):
        """Both metals tickers should get signals (not just gold)."""
        mock_cached.return_value = _make_ovx_data(current=25.0, n=300)
        r_gold = compute_ovx_metals_spillover_signal(
            None, context={"ticker": "XAU-USD"}
        )
        r_silver = compute_ovx_metals_spillover_signal(
            None, context={"ticker": "XAG-USD"}
        )
        # Same OVX data → same signal
        assert r_gold["action"] == r_silver["action"]
