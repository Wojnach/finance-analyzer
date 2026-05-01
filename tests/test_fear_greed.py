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


class TestCryptoFearGreedMalformedAPI:
    """Adversarial review 05-01 P1-13 / 04-29 DE-P1-2: alternative.me API
    returns {"data": []} during maintenance windows. Previous unguarded
    body["data"][0] raised IndexError, killing every cycle's fear-greed
    signal computation silently."""

    def test_returns_none_on_fetch_failure(self):
        """fetch_json returning None must propagate to None (not crash)."""
        with patch("portfolio.fear_greed.fetch_json", return_value=None):
            assert fear_greed.get_crypto_fear_greed() is None

    def test_returns_none_on_empty_data_array(self):
        """Maintenance: API returns {"data": []}. Must return None, not crash."""
        with patch("portfolio.fear_greed.fetch_json", return_value={"data": []}):
            assert fear_greed.get_crypto_fear_greed() is None

    def test_returns_none_on_missing_data_key(self):
        """Malformed: body without "data" key. Must return None, not crash."""
        with patch("portfolio.fear_greed.fetch_json", return_value={"other": "x"}):
            assert fear_greed.get_crypto_fear_greed() is None

    def test_returns_none_on_data_not_list(self):
        """Malformed: data field is not a list. Must return None."""
        with patch("portfolio.fear_greed.fetch_json", return_value={"data": "oops"}):
            assert fear_greed.get_crypto_fear_greed() is None

    def test_returns_none_on_data_entry_not_dict(self):
        """Malformed: data[0] is not a dict. Must return None."""
        with patch("portfolio.fear_greed.fetch_json", return_value={"data": ["string"]}):
            assert fear_greed.get_crypto_fear_greed() is None

    def test_returns_none_on_missing_inner_keys(self):
        """Malformed: data[0] dict missing required keys (value/value_classification/timestamp)."""
        with patch("portfolio.fear_greed.fetch_json", return_value={"data": [{"foo": "bar"}]}):
            assert fear_greed.get_crypto_fear_greed() is None

    def test_returns_none_on_non_int_value(self):
        """Malformed: value is non-numeric string."""
        with patch("portfolio.fear_greed.fetch_json", return_value={
            "data": [{
                "value": "not-a-number",
                "value_classification": "Greed",
                "timestamp": "1700000000",
            }]
        }):
            assert fear_greed.get_crypto_fear_greed() is None

    def test_happy_path_still_works(self):
        """Sanity: a well-formed response still returns the parsed dict."""
        with patch("portfolio.fear_greed.fetch_json", return_value={
            "data": [{
                "value": "42",
                "value_classification": "Fear",
                "timestamp": "1700000000",
            }]
        }):
            result = fear_greed.get_crypto_fear_greed()
        assert result is not None
        assert result["value"] == 42
        assert result["classification"] == "Fear"
        assert "timestamp" in result

    def test_top_level_body_not_dict(self):
        """Defensive: if API ever returns a list at the top level."""
        with patch("portfolio.fear_greed.fetch_json", return_value=["not", "a", "dict"]):
            assert fear_greed.get_crypto_fear_greed() is None
