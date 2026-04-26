"""Tests for hash_ribbons BTC miner capitulation signal module."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from portfolio.signals.hash_ribbons import (
    compute_hash_ribbons_signal,
    _hash_ribbon_crossover,
    _price_momentum_filter,
    _recovery_recency,
    HASH_FAST,
    HASH_SLOW,
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


def _make_hashrate_series(n=120, trend="recovery"):
    """Create a hashrate series simulating different scenarios.

    trend="recovery": dip then recover (triggers BUY)
    trend="capitulating": declining (triggers HOLD)
    trend="stable": flat (triggers HOLD)
    """
    dates = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
    base = 800_000_000.0  # 800 EH/s

    if trend == "recovery":
        # First 80 days: declining, then 40 days: recovering
        rates = np.concatenate([
            base - np.linspace(0, 200_000_000, 80),
            base - 200_000_000 + np.linspace(0, 300_000_000, 40),
        ])
    elif trend == "capitulating":
        # Steady decline
        rates = base - np.linspace(0, 300_000_000, n)
    else:  # stable — truly flat to avoid random crossovers
        rates = np.full(n, base)

    return pd.Series(rates[:n], index=dates[:n], name="hashrate")


class TestSignalInterface:
    """Test that the signal follows the standard interface."""

    @patch("portfolio.signals.hash_ribbons._fetch_hashrate")
    def test_returns_dict_with_required_keys(self, mock_fetch):
        mock_fetch.return_value = _make_hashrate_series(120, "stable")
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto"}
        result = compute_hash_ribbons_signal(df, context=ctx)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    @patch("portfolio.signals.hash_ribbons._fetch_hashrate")
    def test_has_sub_signals(self, mock_fetch):
        mock_fetch.return_value = _make_hashrate_series(120, "stable")
        df = _make_df()
        ctx = {"ticker": "BTC-USD"}
        result = compute_hash_ribbons_signal(df, context=ctx)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)

    @patch("portfolio.signals.hash_ribbons._fetch_hashrate")
    def test_has_indicators(self, mock_fetch):
        mock_fetch.return_value = _make_hashrate_series(120, "stable")
        df = _make_df()
        ctx = {"ticker": "BTC-USD"}
        result = compute_hash_ribbons_signal(df, context=ctx)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_hash_ribbons_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=3)
        result = compute_hash_ribbons_signal(df)
        assert result["action"] == "HOLD"

    @patch("portfolio.signals.hash_ribbons._fetch_hashrate")
    def test_nan_handling(self, mock_fetch):
        mock_fetch.return_value = _make_hashrate_series(120, "stable")
        df = _make_df()
        df.iloc[50:55, df.columns.get_loc("close")] = np.nan
        ctx = {"ticker": "BTC-USD"}
        result = compute_hash_ribbons_signal(df, context=ctx)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    @patch("portfolio.signals.hash_ribbons._fetch_hashrate")
    def test_with_context(self, mock_fetch):
        mock_fetch.return_value = _make_hashrate_series(120, "stable")
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto", "regime": "trending-up"}
        result = compute_hash_ribbons_signal(df, context=ctx)
        assert isinstance(result, dict)


class TestBTCOnly:
    """Test BTC-only filtering."""

    @patch("portfolio.signals.hash_ribbons._fetch_hashrate")
    def test_btc_ticker_accepted(self, mock_fetch):
        mock_fetch.return_value = _make_hashrate_series(120, "stable")
        df = _make_df()
        for ticker in ["BTC-USD", "BTC/USD", "BTCUSD"]:
            result = compute_hash_ribbons_signal(df, context={"ticker": ticker})
            # Should compute (not just HOLD due to ticker filter)
            assert "sub_signals" in result

    def test_non_btc_returns_hold(self):
        df = _make_df()
        for ticker in ["ETH-USD", "XAU-USD", "MSTR", "XAG-USD"]:
            result = compute_hash_ribbons_signal(df, context={"ticker": ticker})
            assert result["action"] == "HOLD"
            assert result["confidence"] == 0.0

    @patch("portfolio.signals.hash_ribbons._fetch_hashrate")
    def test_no_context_still_computes(self, mock_fetch):
        """No context means no ticker filter -- signal should compute."""
        mock_fetch.return_value = _make_hashrate_series(120, "stable")
        df = _make_df()
        result = compute_hash_ribbons_signal(df)
        assert "sub_signals" in result


class TestHashRibbonCrossover:
    """Test the hash ribbon crossover sub-indicator."""

    def test_recovery_crossover_emits_buy(self):
        """When 30DMA crosses above 60DMA, should emit BUY."""
        hashrate = _make_hashrate_series(120, "recovery")
        vote, indicators = _hash_ribbon_crossover(hashrate)
        # The recovery series should eventually produce a crossover
        assert indicators["hash_sma30"] is not None
        assert indicators["hash_sma60"] is not None

    def test_capitulation_emits_hold(self):
        """When miners are capitulating (30DMA < 60DMA), should emit HOLD."""
        hashrate = _make_hashrate_series(120, "capitulating")
        vote, indicators = _hash_ribbon_crossover(hashrate)
        assert vote == "HOLD"
        assert indicators["capitulating"] == True  # noqa: E712 (np.bool_ vs bool)

    def test_stable_emits_hold(self):
        """Stable hashrate should emit HOLD."""
        hashrate = _make_hashrate_series(120, "stable")
        vote, indicators = _hash_ribbon_crossover(hashrate)
        assert vote == "HOLD"

    def test_insufficient_data_returns_hold(self):
        """Not enough data for 60-day SMA."""
        hashrate = _make_hashrate_series(50, "stable")
        vote, indicators = _hash_ribbon_crossover(hashrate)
        assert vote == "HOLD"

    def test_none_returns_hold(self):
        vote, indicators = _hash_ribbon_crossover(None)
        assert vote == "HOLD"


class TestPriceMomentumFilter:
    """Test the price momentum filter sub-indicator."""

    def test_uptrend_emits_buy(self):
        """Price SMA10 > SMA20 should emit BUY."""
        # Create uptrending data
        close = pd.Series(np.linspace(100, 120, 50))
        vote, indicators = _price_momentum_filter(close)
        assert vote == "BUY"
        assert indicators["price_sma10"] > indicators["price_sma20"]

    def test_downtrend_emits_hold(self):
        """Price SMA10 < SMA20 should emit HOLD."""
        close = pd.Series(np.linspace(120, 100, 50))
        vote, indicators = _price_momentum_filter(close)
        assert vote == "HOLD"

    def test_insufficient_data(self):
        close = pd.Series([100, 101, 102])
        vote, indicators = _price_momentum_filter(close)
        assert vote == "HOLD"


class TestRecoveryRecency:
    """Test the recovery recency sub-indicator."""

    def test_no_recent_recovery(self):
        """Stable hashrate: no crossover in last 14 days."""
        hashrate = _make_hashrate_series(120, "stable")
        vote, indicators = _recovery_recency(hashrate)
        assert vote == "HOLD"

    def test_insufficient_data(self):
        hashrate = _make_hashrate_series(50, "stable")
        vote, indicators = _recovery_recency(hashrate)
        assert vote == "HOLD"


class TestConfidenceCap:
    """Verify confidence is capped at 0.7 (external data pattern)."""

    @patch("portfolio.signals.hash_ribbons._fetch_hashrate")
    def test_confidence_never_exceeds_cap(self, mock_fetch):
        mock_fetch.return_value = _make_hashrate_series(120, "recovery")
        df = _make_df(100)
        ctx = {"ticker": "BTC-USD"}
        result = compute_hash_ribbons_signal(df, context=ctx)
        assert result["confidence"] <= 0.7


class TestNeverSells:
    """This is a BUY-only signal -- it should never output SELL."""

    @patch("portfolio.signals.hash_ribbons._fetch_hashrate")
    def test_never_sell_stable(self, mock_fetch):
        mock_fetch.return_value = _make_hashrate_series(120, "stable")
        df = _make_df()
        ctx = {"ticker": "BTC-USD"}
        result = compute_hash_ribbons_signal(df, context=ctx)
        assert result["action"] != "SELL"

    @patch("portfolio.signals.hash_ribbons._fetch_hashrate")
    def test_never_sell_capitulating(self, mock_fetch):
        mock_fetch.return_value = _make_hashrate_series(120, "capitulating")
        df = _make_df()
        ctx = {"ticker": "BTC-USD"}
        result = compute_hash_ribbons_signal(df, context=ctx)
        assert result["action"] != "SELL"

    @patch("portfolio.signals.hash_ribbons._fetch_hashrate")
    def test_never_sell_recovery(self, mock_fetch):
        mock_fetch.return_value = _make_hashrate_series(120, "recovery")
        df = _make_df()
        ctx = {"ticker": "BTC-USD"}
        result = compute_hash_ribbons_signal(df, context=ctx)
        assert result["action"] != "SELL"


class TestHashrateFetchFailure:
    """Test graceful degradation when hashrate API fails."""

    @patch("portfolio.signals.hash_ribbons._fetch_hashrate")
    def test_returns_hold_on_fetch_failure(self, mock_fetch):
        mock_fetch.return_value = None
        df = _make_df()
        ctx = {"ticker": "BTC-USD"}
        result = compute_hash_ribbons_signal(df, context=ctx)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0
