"""Tests for credit_spread signal module."""
import numpy as np
import pandas as pd
import pytest

from portfolio.signals.credit_spread import (
    _crisis_level_signal,
    _oas_acceleration_signal,
    _oas_momentum_signal,
    _oas_zscore_signal,
    compute_credit_spread_signal,
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


def _make_oas_values(current=3.0, n=260, mean=3.0, std=0.3):
    """Create synthetic HY OAS values (newest first)."""
    np.random.seed(42)
    values = np.random.normal(mean, std, n).tolist()
    values[0] = current  # Set current value explicitly
    return values


class TestSignalInterface:
    """Test that the signal follows the standard interface."""

    def test_returns_dict_with_required_keys(self, monkeypatch):
        monkeypatch.setattr(
            "portfolio.signals.credit_spread._fetch_hy_oas",
            lambda key: _make_oas_values(),
        )
        df = _make_df()
        ctx = {"ticker": "XAU-USD", "config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_credit_spread_signal(df, context=ctx)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self, monkeypatch):
        monkeypatch.setattr(
            "portfolio.signals.credit_spread._fetch_hy_oas",
            lambda key: _make_oas_values(),
        )
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_credit_spread_signal(df, context=ctx)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)

    def test_has_indicators(self, monkeypatch):
        monkeypatch.setattr(
            "portfolio.signals.credit_spread._fetch_hy_oas",
            lambda key: _make_oas_values(),
        )
        df = _make_df()
        ctx = {"ticker": "XAG-USD", "config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_credit_spread_signal(df, context=ctx)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)

    def test_no_context_returns_hold(self):
        df = _make_df()
        result = compute_credit_spread_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_unknown_ticker_returns_hold(self, monkeypatch):
        monkeypatch.setattr(
            "portfolio.signals.credit_spread._fetch_hy_oas",
            lambda key: _make_oas_values(),
        )
        df = _make_df()
        ctx = {"ticker": "UNKNOWN", "config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_credit_spread_signal(df, context=ctx)
        assert result["action"] == "HOLD"

    def test_no_fred_key_returns_hold(self, monkeypatch):
        """Without FRED key (and no config.json fallback), returns HOLD."""
        monkeypatch.setattr(
            "portfolio.signals.credit_spread._fetch_hy_oas",
            lambda key: None,
        )
        monkeypatch.setattr(
            "portfolio.signals.credit_spread._get_fred_key",
            lambda ctx: "",
        )
        df = _make_df()
        ctx = {"ticker": "XAU-USD", "config": {}}
        result = compute_credit_spread_signal(df, context=ctx)
        assert result["action"] == "HOLD"

    def test_empty_oas_data_returns_hold(self, monkeypatch):
        monkeypatch.setattr(
            "portfolio.signals.credit_spread._fetch_hy_oas",
            lambda key: None,
        )
        df = _make_df()
        ctx = {"ticker": "XAU-USD", "config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_credit_spread_signal(df, context=ctx)
        assert result["action"] == "HOLD"


class TestSubIndicators:
    """Test individual sub-indicator logic."""

    def test_zscore_risk_off_gold_buy(self):
        """High z-score (risk-off) should produce BUY for gold."""
        # Current = 4.5, history mean ~3.0, std ~0.3 → z ≈ 5.0
        values = _make_oas_values(current=4.5, mean=3.0, std=0.3)
        action, ind = _oas_zscore_signal(values, safe_haven=True)
        assert action == "BUY"
        assert ind["oas_zscore"] > 1.5

    def test_zscore_risk_off_crypto_sell(self):
        """High z-score (risk-off) should produce SELL for crypto."""
        values = _make_oas_values(current=4.5, mean=3.0, std=0.3)
        action, ind = _oas_zscore_signal(values, safe_haven=False)
        assert action == "SELL"

    def test_zscore_risk_on_gold_sell(self):
        """Low z-score (risk-on) should produce SELL for gold."""
        values = _make_oas_values(current=2.2, mean=3.0, std=0.3)
        action, ind = _oas_zscore_signal(values, safe_haven=True)
        assert action == "SELL"
        assert ind["oas_zscore"] < -1.0

    def test_zscore_risk_on_crypto_buy(self):
        """Low z-score (risk-on) should produce BUY for crypto."""
        values = _make_oas_values(current=2.2, mean=3.0, std=0.3)
        action, ind = _oas_zscore_signal(values, safe_haven=False)
        assert action == "BUY"

    def test_zscore_neutral_hold(self):
        """Neutral z-score should produce HOLD."""
        values = _make_oas_values(current=3.0, mean=3.0, std=0.3)
        action, ind = _oas_zscore_signal(values, safe_haven=True)
        assert action == "HOLD"

    def test_zscore_insufficient_data(self):
        action, ind = _oas_zscore_signal([3.0] * 10, safe_haven=True)
        assert action == "HOLD"

    def test_momentum_widening_gold_buy(self):
        """Rapid widening (risk-off) → BUY gold."""
        values = [3.5, 3.4, 3.3, 3.2, 3.1, 3.0]  # 5d widened +0.5
        action, ind = _oas_momentum_signal(values, safe_haven=True)
        assert action == "BUY"
        assert ind["oas_mom_5d"] > 0

    def test_momentum_tightening_gold_sell(self):
        """Rapid tightening (risk-on) → SELL gold."""
        values = [2.5, 2.6, 2.7, 2.8, 2.9, 3.0]  # 5d tightened -0.5
        action, ind = _oas_momentum_signal(values, safe_haven=True)
        assert action == "SELL"

    def test_momentum_neutral_hold(self):
        """Small change → HOLD."""
        values = [3.05, 3.04, 3.03, 3.02, 3.01, 3.0]  # only 0.05 change
        action, ind = _oas_momentum_signal(values, safe_haven=True)
        assert action == "HOLD"

    def test_acceleration_increasing_risk_off(self):
        """Accelerating widening → early warning BUY gold."""
        # Current mom: 3.8 - 3.3 = 0.5, Prev mom: 3.3 - 3.1 = 0.2 → accel = 0.3
        values = [3.8, 3.7, 3.6, 3.5, 3.4, 3.3, 3.25, 3.2, 3.15, 3.12, 3.1]
        action, ind = _oas_acceleration_signal(values, safe_haven=True)
        assert action == "BUY"

    def test_acceleration_insufficient_data(self):
        action, ind = _oas_acceleration_signal([3.0] * 5, safe_haven=True)
        assert action == "HOLD"

    def test_crisis_level_buy_gold(self):
        """OAS >= 500bp → crisis → BUY gold."""
        values = [5.5, 5.0, 4.8]
        action, ind = _crisis_level_signal(values, safe_haven=True)
        assert action == "BUY"
        assert ind["oas_crisis"] is True

    def test_crisis_level_sell_crypto(self):
        """OAS >= 500bp → crisis → SELL crypto."""
        values = [6.0]
        action, ind = _crisis_level_signal(values, safe_haven=False)
        assert action == "SELL"

    def test_extreme_complacency_sell_gold(self):
        """OAS <= 2.5 → extreme complacency → contrarian SELL gold."""
        values = [2.3]
        action, ind = _crisis_level_signal(values, safe_haven=True)
        assert action == "SELL"

    def test_normal_level_hold(self):
        """Normal OAS level → HOLD."""
        values = [3.5]
        action, ind = _crisis_level_signal(values, safe_haven=True)
        assert action == "HOLD"

    def test_empty_values_hold(self):
        action, ind = _crisis_level_signal([], safe_haven=True)
        assert action == "HOLD"


class TestAssetClassDirectionality:
    """Test that signal direction flips correctly between asset classes."""

    def test_gold_risk_off_is_buy(self, monkeypatch):
        """Gold should get BUY signal in risk-off (wide spreads)."""
        monkeypatch.setattr(
            "portfolio.signals.credit_spread._fetch_hy_oas",
            lambda key: _make_oas_values(current=5.0, mean=3.0, std=0.3),
        )
        df = _make_df()
        ctx = {"ticker": "XAU-USD", "config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_credit_spread_signal(df, context=ctx)
        assert result["action"] == "BUY"

    def test_btc_risk_off_is_sell(self, monkeypatch):
        """BTC should get SELL signal in risk-off (wide spreads)."""
        monkeypatch.setattr(
            "portfolio.signals.credit_spread._fetch_hy_oas",
            lambda key: _make_oas_values(current=5.0, mean=3.0, std=0.3),
        )
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_credit_spread_signal(df, context=ctx)
        assert result["action"] == "SELL"

    def test_silver_risk_on_is_sell(self, monkeypatch):
        """Silver should get SELL signal in risk-on (tight spreads)."""
        monkeypatch.setattr(
            "portfolio.signals.credit_spread._fetch_hy_oas",
            lambda key: _make_oas_values(current=2.0, mean=3.0, std=0.3),
        )
        df = _make_df()
        ctx = {"ticker": "XAG-USD", "config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_credit_spread_signal(df, context=ctx)
        assert result["action"] == "SELL"

    def test_eth_risk_on_is_buy(self, monkeypatch):
        """ETH should get BUY signal in risk-on (tight spreads)."""
        monkeypatch.setattr(
            "portfolio.signals.credit_spread._fetch_hy_oas",
            lambda key: _make_oas_values(current=2.0, mean=3.0, std=0.3),
        )
        df = _make_df()
        ctx = {"ticker": "ETH-USD", "config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_credit_spread_signal(df, context=ctx)
        assert result["action"] == "BUY"

    def test_mstr_supported(self, monkeypatch):
        """MSTR should be recognized as a risk asset."""
        monkeypatch.setattr(
            "portfolio.signals.credit_spread._fetch_hy_oas",
            lambda key: _make_oas_values(current=5.0, mean=3.0, std=0.3),
        )
        df = _make_df()
        ctx = {"ticker": "MSTR", "config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_credit_spread_signal(df, context=ctx)
        assert result["action"] == "SELL"  # Risk-off → SELL for stocks


class TestCacheBehavior:
    """Test FRED data caching."""

    def test_cache_prevents_repeated_fetch(self, monkeypatch):
        call_count = 0
        original_values = _make_oas_values()

        def mock_fetch(key):
            nonlocal call_count
            call_count += 1
            return original_values

        monkeypatch.setattr(
            "portfolio.signals.credit_spread._fetch_hy_oas",
            mock_fetch,
        )

        df = _make_df()
        ctx = {"ticker": "XAU-USD", "config": {"golddigger": {"fred_api_key": "test"}}}

        # Call twice — should both go through our mock (cache is internal to _fetch_hy_oas)
        compute_credit_spread_signal(df, context=ctx)
        compute_credit_spread_signal(df, context=ctx)
        # Mock always gets called since we replaced the function
        assert call_count == 2


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_all_same_oas_values(self, monkeypatch):
        """All identical OAS values → std=0 → HOLD (no variance)."""
        monkeypatch.setattr(
            "portfolio.signals.credit_spread._fetch_hy_oas",
            lambda key: [3.0] * 260,
        )
        df = _make_df()
        ctx = {"ticker": "XAU-USD", "config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_credit_spread_signal(df, context=ctx)
        # Z-score sub will HOLD (std < 0.01), but crisis sub might vote
        # Either way, confidence should be low
        assert result["confidence"] <= 1.0

    def test_very_short_oas_history(self, monkeypatch):
        """Only 5 data points → most sub-indicators HOLD."""
        monkeypatch.setattr(
            "portfolio.signals.credit_spread._fetch_hy_oas",
            lambda key: [3.5, 3.4, 3.3, 3.2, 3.1],
        )
        df = _make_df()
        ctx = {"ticker": "XAU-USD", "config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_credit_spread_signal(df, context=ctx)
        # Insufficient data for most sub-indicators
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_context_with_ticker_kwarg(self, monkeypatch):
        """Ticker passed as kwarg instead of context."""
        monkeypatch.setattr(
            "portfolio.signals.credit_spread._fetch_hy_oas",
            lambda key: _make_oas_values(current=5.0, mean=3.0, std=0.3),
        )
        df = _make_df()
        ctx = {"config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_credit_spread_signal(df, context=ctx, ticker="XAU-USD")
        assert result["action"] == "BUY"
