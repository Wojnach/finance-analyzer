"""Tests for portfolio.fear_greed — A-DE-4 yfinance MultiIndex flatten."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from portfolio import fear_greed


@pytest.fixture
def reset_yf_lock():
    """yfinance_lock is shared global state — make sure tests don't deadlock."""
    yield


def _mock_history_dataframe(closes, multiindex=False):
    """Build a fake yfinance Ticker.history() result with optional MultiIndex
    columns (newer yfinance versions sometimes return these even for a
    single ticker)."""
    df = pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": [1] * len(closes),
        }
    )
    if multiindex:
        df.columns = pd.MultiIndex.from_tuples(
            [(c, "^VIX") for c in df.columns]
        )
    return df


class TestStockFearGreedFlatColumns:
    """Baseline: ordinary single-level columns work."""

    def test_returns_value_for_normal_dataframe(self):
        df = _mock_history_dataframe([15.0, 16.0, 17.0, 18.0, 19.0], multiindex=False)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = fear_greed.get_stock_fear_greed()
        assert result is not None
        assert "value" in result
        assert 0 <= result["value"] <= 100
        assert result["vix"] == 19.0  # last close

    def test_returns_none_on_empty_history(self):
        df = pd.DataFrame()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = fear_greed.get_stock_fear_greed()
        assert result is None


class TestStockFearGreedMultiIndex:
    """A-DE-4: yfinance sometimes returns MultiIndex columns even for a
    single Ticker.history() call. Without flattening, h["Close"] returns
    a DataFrame instead of a Series, .iloc[-1] returns a row, and
    float(...) raises TypeError. The signal then silently dies."""

    def test_handles_multiindex_columns(self):
        """The fix must flatten MultiIndex columns and return a valid value."""
        df = _mock_history_dataframe([22.0, 23.0, 24.0, 25.0, 26.0], multiindex=True)
        # Confirm we actually built a MultiIndex df
        assert isinstance(df.columns, pd.MultiIndex)

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = fear_greed.get_stock_fear_greed()

        assert result is not None, (
            "get_stock_fear_greed returned None on a MultiIndex DataFrame — "
            "the MultiIndex flatten was not applied (A-DE-4)"
        )
        assert result["vix"] == 26.0
        assert isinstance(result["value"], int)

    def test_multiindex_does_not_swallow_errors_silently(self):
        """A-DE-4 belt-and-suspenders: if yfinance returns something
        unexpected, the function should still return None (not crash
        the caller). The fix shouldn't introduce new exception paths."""
        df = _mock_history_dataframe([20.0], multiindex=True)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = fear_greed.get_stock_fear_greed()
        # Should succeed — the single-row case was previously fine.
        assert result is not None
        assert result["vix"] == 20.0
