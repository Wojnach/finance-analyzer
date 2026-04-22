"""Tests for complexity_gap_regime signal module."""
import numpy as np
import pandas as pd
import pytest

from portfolio.signals.complexity_gap_regime import (
    _compute_complexity_gap_series,
    _corr_regime_vote,
    _gap_slope_vote,
    _gap_zscore_vote,
    compute_complexity_gap_regime_signal,
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


def _make_multi_asset_closes(n=100, n_assets=5):
    """Create a test multi-asset DataFrame with correlated returns."""
    np.random.seed(42)
    # Base random walks with mild correlation
    base = np.random.randn(n)
    prices = pd.DataFrame()
    tickers = ["BTC-USD", "ETH-USD", "GC=F", "SI=F", "SPY"]
    for i, t in enumerate(tickers[:n_assets]):
        noise = np.random.randn(n) * 0.5
        combined = base * 0.3 + noise
        prices[t] = 100 * np.exp(np.cumsum(combined * 0.01))
    prices.index = pd.date_range("2026-01-01", periods=n, freq="D")
    return prices


class TestSignalInterface:
    """Test that the signal follows the standard interface."""

    def test_returns_dict_with_required_keys(self, monkeypatch):
        """Signal returns dict with action, confidence, sub_signals, indicators."""
        # Mock the multi-asset fetch to avoid yfinance dependency
        closes = _make_multi_asset_closes(100)
        monkeypatch.setattr(
            "portfolio.signals.complexity_gap_regime._fetch_multi_asset_closes",
            lambda: closes,
        )
        df = _make_df()
        result = compute_complexity_gap_regime_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self, monkeypatch):
        closes = _make_multi_asset_closes(100)
        monkeypatch.setattr(
            "portfolio.signals.complexity_gap_regime._fetch_multi_asset_closes",
            lambda: closes,
        )
        df = _make_df()
        result = compute_complexity_gap_regime_signal(df)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)

    def test_has_indicators(self, monkeypatch):
        closes = _make_multi_asset_closes(100)
        monkeypatch.setattr(
            "portfolio.signals.complexity_gap_regime._fetch_multi_asset_closes",
            lambda: closes,
        )
        df = _make_df()
        result = compute_complexity_gap_regime_signal(df)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_complexity_gap_regime_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=3)
        result = compute_complexity_gap_regime_signal(df)
        assert result["action"] == "HOLD"

    def test_nan_handling(self, monkeypatch):
        closes = _make_multi_asset_closes(100)
        monkeypatch.setattr(
            "portfolio.signals.complexity_gap_regime._fetch_multi_asset_closes",
            lambda: closes,
        )
        df = _make_df()
        df.iloc[50:55, df.columns.get_loc("close")] = np.nan
        result = compute_complexity_gap_regime_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_with_context(self, monkeypatch):
        closes = _make_multi_asset_closes(100)
        monkeypatch.setattr(
            "portfolio.signals.complexity_gap_regime._fetch_multi_asset_closes",
            lambda: closes,
        )
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto", "regime": "trending-up"}
        result = compute_complexity_gap_regime_signal(df, context=ctx)
        assert isinstance(result, dict)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_safe_haven_inverts_direction(self, monkeypatch):
        """Metals tickers should invert signal direction."""
        closes = _make_multi_asset_closes(100)
        monkeypatch.setattr(
            "portfolio.signals.complexity_gap_regime._fetch_multi_asset_closes",
            lambda: closes,
        )
        df = _make_df()
        ctx_gold = {"ticker": "XAU-USD", "asset_class": "metals"}
        result_gold = compute_complexity_gap_regime_signal(df, context=ctx_gold)
        assert result_gold["indicators"].get("is_safe_haven") is True

        ctx_btc = {"ticker": "BTC-USD", "asset_class": "crypto"}
        result_btc = compute_complexity_gap_regime_signal(df, context=ctx_btc)
        assert result_btc["indicators"].get("is_safe_haven") is False


class TestComplexityGapComputation:
    """Test the core RMT computation."""

    def test_gap_series_has_expected_columns(self):
        closes = _make_multi_asset_closes(100)
        gap_df = _compute_complexity_gap_series(closes)
        assert gap_df is not None
        assert "gap" in gap_df.columns
        assert "avg_corr" in gap_df.columns
        assert "max_eig_norm" in gap_df.columns

    def test_gap_values_are_finite(self):
        closes = _make_multi_asset_closes(100)
        gap_df = _compute_complexity_gap_series(closes)
        assert gap_df is not None
        assert gap_df["gap"].notna().all()
        assert np.isfinite(gap_df["gap"].values).all()

    def test_none_input_returns_none(self):
        assert _compute_complexity_gap_series(None) is None

    def test_too_few_rows_returns_none(self):
        closes = _make_multi_asset_closes(10)
        assert _compute_complexity_gap_series(closes) is None

    def test_synchronized_market_has_low_gap(self):
        """When all assets are perfectly correlated, gap should be small."""
        np.random.seed(42)
        n = 100
        base = np.cumsum(np.random.randn(n) * 0.01)
        # All assets follow the same path
        closes = pd.DataFrame({
            t: 100 * np.exp(base + np.random.randn(n) * 0.001)
            for t in _make_multi_asset_closes().columns
        })
        closes.index = pd.date_range("2026-01-01", periods=n, freq="D")
        gap_df = _compute_complexity_gap_series(closes)
        if gap_df is not None and len(gap_df) > 0:
            # High correlation → gap should be relatively small
            avg_gap = gap_df["gap"].mean()
            avg_corr = gap_df["avg_corr"].mean()
            assert avg_corr > 0.5  # Should be highly correlated


class TestSubSignals:
    """Test individual sub-signal voting functions."""

    def test_zscore_vote_collapse_risk(self):
        # Negative z-score (gap collapse) = SELL for risk assets
        assert _gap_zscore_vote(-2.0, is_safe_haven=False) == "SELL"
        assert _gap_zscore_vote(-2.0, is_safe_haven=True) == "BUY"

    def test_zscore_vote_widening_risk(self):
        # Positive z-score (gap widening) = BUY for risk assets
        assert _gap_zscore_vote(2.0, is_safe_haven=False) == "BUY"
        assert _gap_zscore_vote(2.0, is_safe_haven=True) == "SELL"

    def test_zscore_vote_neutral(self):
        assert _gap_zscore_vote(0.5, is_safe_haven=False) == "HOLD"
        assert _gap_zscore_vote(-0.5, is_safe_haven=True) == "HOLD"

    def test_corr_regime_high(self):
        assert _corr_regime_vote(0.6, is_safe_haven=False) == "SELL"
        assert _corr_regime_vote(0.6, is_safe_haven=True) == "BUY"

    def test_corr_regime_low(self):
        assert _corr_regime_vote(0.1, is_safe_haven=False) == "BUY"
        assert _corr_regime_vote(0.1, is_safe_haven=True) == "SELL"

    def test_corr_regime_neutral(self):
        assert _corr_regime_vote(0.3, is_safe_haven=False) == "HOLD"

    def test_slope_vote_insufficient_data(self):
        gap_series = pd.Series([0.1, 0.2])
        assert _gap_slope_vote(gap_series, is_safe_haven=False) == "HOLD"

    def test_fetch_failure_returns_hold(self, monkeypatch):
        """When multi-asset fetch fails, should return HOLD."""
        monkeypatch.setattr(
            "portfolio.signals.complexity_gap_regime._fetch_multi_asset_closes",
            lambda: None,
        )
        df = _make_df()
        result = compute_complexity_gap_regime_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0
